"""One WebSocket connection = one :class:`Session` (SPEC §6).

A session owns a seeded :class:`StreamingTable` (persistent shoe, strict bankroll), the selected agent,
running :class:`LabMetrics`, and three background activities:

* the **frame pump** sends queued ``state`` messages with pacing, so dealt cards animate (one state per
  card during the deal, crowd and dealer phases);
* the **autoplay loop** lets the agent decide whenever the pump is idle (``delay_ms`` pacing);
* a **simulation** (:class:`SimulationRun`, worker processes) streaming ``sim_progress``.

Every command is validated; problems are reported as ``{"type": "error"}`` (followed by a fresh state
so the client re-syncs) and never close the socket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..cards import parse_card
from ..config import GameConfig, apply_patch, load_config, to_dict
from ..crowd import Crowd
from ..engine import Action, Bets, Phase
from ..sidebets import SIDE_BETS
from . import agents as registry
from .metrics import LabMetrics
from .simulate import MAX_ROUNDS, SimJob, SimulationRun
from .stream import FrameRecorder, StreamingTable, visual_signature

log = logging.getLogger("onebj.server")

SendFn = Callable[[str], Awaitable[None]]

#: pause between streamed card frames when a human plays (seconds)
INTERACTIVE_FRAME_DELAY = 0.38
DEFAULT_DELAY_MS = 700
BUSY_MESSAGE = "Please wait — cards are still being dealt"
BET_KEYS = ("main", *SIDE_BETS)


class CommandError(Exception):
    """A client command that cannot be executed (reported to the client as an ``error``)."""


class Session:
    def __init__(self, send: SendFn, *, config: GameConfig | None = None, seed: int | None = None,
                 debug: bool = False, checkpoints: Path | None = None,
                 sim_processes: bool | None = None) -> None:
        self._send_text = send
        self.cfg = config or load_config()
        self.rng = random.Random(seed)
        self.debug = debug
        self.checkpoints = checkpoints or registry.checkpoints_dir()
        self.sim_processes = sim_processes
        self.table = self._new_table(self.cfg.bankroll.start)
        self.metrics = LabMetrics(start=self.table.balance)
        self._metrics_json: dict[str, Any] | None = None
        self._recorded_round = 0
        self.agent_name = "hilo"
        self.agent: Any = None
        self._agents: dict[str, Any] = {}
        self.network: dict[str, Any] | None = None
        self.autoplay = False
        self.delay_ms = DEFAULT_DELAY_MS
        self.advice = False
        self.last_trace: dict[str, Any] | None = None
        self.sim: SimulationRun | None = None
        self._sim_task: asyncio.Task | None = None
        self._auto_task: asyncio.Task | None = None
        self._frames: deque[tuple[dict[str, Any], float]] = deque()
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._sending = asyncio.Lock()
        self._pump_task: asyncio.Task | None = None
        self._closed = False
        self._handlers: dict[str, Callable[[dict[str, Any]], Awaitable[None]]] = {
            "place_bets": self._cmd_place_bets, "action": self._cmd_action, "ai_step": self._cmd_ai_step,
            "autoplay": self._cmd_autoplay, "advice": self._cmd_advice, "simulate": self._cmd_simulate,
            "stop": self._cmd_stop, "set_agent": self._cmd_set_agent, "new_shoe": self._cmd_new_shoe,
            "reset": self._cmd_reset, "set_config": self._cmd_set_config,
        }
        if debug:
            self._handlers["debug_rig"] = self._cmd_debug_rig

    # ================================================================== lifecycle
    async def start(self) -> None:
        """Pick the default agent (newest NN checkpoint, else Hi-Lo), send ``hello`` and the first state."""
        self._pump_task = asyncio.create_task(self._pump())
        names = registry.available_agents(self.checkpoints)
        default = names[0]
        try:
            await self._load_agent(default)
        except Exception as e:  # noqa: BLE001 - a broken checkpoint must not prevent connecting
            log.warning("could not load %s: %s", default, e)
            await self._error(f"Could not load {default}: {e} — using hilo")
            await self._load_agent("hilo")
        await self._send_hello()
        self._enqueue_state()

    async def close(self) -> None:
        self._closed = True
        self.autoplay = False
        if self.sim is not None:
            self.sim.stop()
        for task in (self._auto_task, self._pump_task):
            if task is not None:
                task.cancel()
        if self._sim_task is not None:
            try:
                await asyncio.wait_for(self._sim_task, timeout=6)
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - best effort on shutdown
                if self.sim is not None:
                    self.sim.terminate()

    @property
    def busy(self) -> bool:
        """True while streamed frames are still being sent."""
        return not self._idle.is_set()

    # ================================================================== transport
    async def _send(self, msg: dict[str, Any]) -> None:
        if self._closed:
            return
        text = _dumps(msg)
        async with self._sending:
            await self._send_text(text)

    async def _error(self, message: str) -> None:
        await self._send({"type": "error", "message": message})

    async def _send_hello(self) -> None:
        await self._send({"type": "hello", "config": to_dict(self.cfg), "network": self.network,
                          "agents": self._agent_names()})

    def _enqueue(self, msg: dict[str, Any], delay: float = 0.0) -> None:
        self._frames.append((msg, delay))
        self._idle.clear()
        self._wake.set()

    def _enqueue_state(self) -> None:
        self._enqueue(self._state_msg())

    async def _pump(self) -> None:
        try:
            while True:
                if not self._frames:
                    self._idle.set()
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                msg, delay = self._frames.popleft()
                await self._send(msg)
                if delay > 0 and self._frames:
                    await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - socket gone: stop quietly
            log.debug("pump stopped: %s", e)
            self._closed = True
            self._idle.set()

    # ================================================================== commands
    async def handle_text(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except (ValueError, TypeError):
            await self._error("Invalid message: not JSON")
            return
        if not isinstance(msg, dict) or not isinstance(msg.get("cmd"), str):
            await self._error('Invalid message: expected an object with a "cmd" string')
            return
        await self.handle(msg)

    async def handle(self, msg: dict[str, Any]) -> None:
        cmd = msg["cmd"]
        handler = self._handlers.get(cmd)
        if handler is None:
            await self._error(f"Unknown command '{cmd}' (expected one of: {', '.join(sorted(self._handlers))})")
            return
        try:
            await handler(msg)
        except (CommandError, ValueError, RuntimeError, KeyError, TypeError) as e:
            await self._error(str(e) or type(e).__name__)
            if not self.busy:
                self._enqueue_state()
        except Exception as e:
            log.exception("command %s failed", cmd)
            await self._error(f"Internal error while handling '{cmd}': {type(e).__name__}: {e}")
            if not self.busy:
                self._enqueue_state()

    def _require_idle(self) -> None:
        if self.busy:
            raise CommandError(BUSY_MESSAGE)

    def _require_between_rounds(self, what: str) -> None:
        if self.table.phase not in (Phase.BETTING, Phase.SETTLED):
            raise CommandError(f"Finish the current round before {what}")

    async def _cmd_place_bets(self, msg: dict[str, Any]) -> None:
        self._require_idle()
        self._require_between_rounds("placing new bets")
        bets = _parse_bets(msg.get("bets"))
        self._run(lambda: self.table.place_bets(bets))

    async def _cmd_action(self, msg: dict[str, Any]) -> None:
        self._require_idle()
        raw = msg.get("action")
        try:
            action = Action(raw)
        except ValueError:
            raise CommandError(f"Unknown action {raw!r} (expected one of: {', '.join(a.value for a in Action)})")
        t = self.table
        if t.phase not in (Phase.PLAYER, Phase.INSURANCE):
            raise CommandError("No decision pending — place your bets and deal")
        legal = t.legal_actions()
        if action not in legal:
            raise CommandError(f"'{action.value}' is not allowed now (legal: {', '.join(a.value for a in legal)})")
        if t.phase is Phase.PLAYER:
            self._record_decision(action)
        self._run(lambda: t.act(action))

    async def _cmd_ai_step(self, msg: dict[str, Any]) -> None:
        self._require_idle()
        self._ai_step()

    async def _cmd_autoplay(self, msg: dict[str, Any]) -> None:
        on = msg.get("on")
        if not isinstance(on, bool):
            raise CommandError('autoplay needs "on": true|false')
        if "delay_ms" in msg and msg["delay_ms"] is not None:
            d = msg["delay_ms"]
            if not _is_number(d):
                raise CommandError("delay_ms must be a number")
            self.delay_ms = int(min(max(d, 0), 10_000))
        self.autoplay = on
        if on and (self._auto_task is None or self._auto_task.done()):
            self._auto_task = asyncio.create_task(self._autoplay_loop())
        elif not on and self._auto_task is not None:
            self._auto_task.cancel()
            self._auto_task = None
        self._enqueue_state()

    async def _cmd_advice(self, msg: dict[str, Any]) -> None:
        on = msg.get("on")
        if not isinstance(on, bool):
            raise CommandError('advice needs "on": true|false')
        self.advice = on
        self._enqueue_state()

    async def _cmd_simulate(self, msg: dict[str, Any]) -> None:
        if self.sim is not None:
            raise CommandError("A simulation is already running — stop it first")
        rounds = msg.get("rounds")
        if not _is_number(rounds) or rounds != int(rounds) or not 1 <= rounds <= MAX_ROUNDS:
            raise CommandError(f"rounds must be an integer between 1 and {MAX_ROUNDS:,}")
        agent = msg.get("agent") or self.agent_name
        if agent not in self._agent_names():
            raise CommandError(f"Unknown agent {agent!r} (available: {', '.join(self._agent_names())})")
        if self.autoplay:
            self.autoplay = False
            if self._auto_task is not None:
                self._auto_task.cancel()
                self._auto_task = None
            self._enqueue_state()
        job = SimJob(config=to_dict(self.cfg), agent=agent, rounds=int(rounds), seed=self.rng.randrange(2**31),
                     start=float(self.cfg.bankroll.start), checkpoints=str(self.checkpoints))

        async def progress(done: int, total: int, metrics: dict[str, Any], finished: bool) -> None:
            await self._send({"type": "sim_progress", "done": done, "total": total, "metrics": metrics,
                              "finished": finished, "agent": agent})

        self.sim = SimulationRun(job, progress, use_processes=self.sim_processes)
        await progress(0, job.rounds, LabMetrics(start=job.start).to_dict(), False)
        self._sim_task = asyncio.create_task(self._run_sim(self.sim))

    async def _run_sim(self, run: SimulationRun) -> None:
        try:
            await run.run()
            if run.error:
                await self._error(f"Simulation failed: {run.error}")
        except asyncio.CancelledError:
            run.stop()
            raise
        except Exception as e:
            log.exception("simulation failed")
            await self._error(f"Simulation failed: {type(e).__name__}: {e}")
        finally:
            if self.sim is run:
                self.sim = None
            if not self._closed:
                self._enqueue_state()

    async def _cmd_stop(self, msg: dict[str, Any]) -> None:
        self.autoplay = False
        if self._auto_task is not None:
            self._auto_task.cancel()
            self._auto_task = None
        if self.sim is not None:
            self.sim.stop()
        self._enqueue_state()

    async def _cmd_set_agent(self, msg: dict[str, Any]) -> None:
        name = msg.get("agent")
        if not isinstance(name, str) or name not in self._agent_names():
            raise CommandError(f"Unknown agent {name!r} (available: {', '.join(self._agent_names())})")
        before = self.network
        await self._load_agent(name)
        self.last_trace = None
        if self.network != before:
            await self._send_hello()
        self._enqueue_state()

    async def _cmd_new_shoe(self, msg: dict[str, Any]) -> None:
        self._require_idle()
        self._require_between_rounds("shuffling a new shoe")
        self.table.new_shoe()
        self.table.messages = [f"New shoe #{self.table.shoe.shoe_number} — shuffled"
                               + (" and split into two halves" if self.cfg.shoe.split_halves else "")]
        self._enqueue_state()

    async def _cmd_reset(self, msg: dict[str, Any]) -> None:
        bankroll = msg.get("bankroll")
        if bankroll is None:
            bankroll = self.cfg.bankroll.start
        if not _is_number(bankroll) or bankroll <= 0:
            raise CommandError("bankroll must be a positive number")
        self.autoplay = False
        if self._auto_task is not None:
            self._auto_task.cancel()
            self._auto_task = None
        self._frames.clear()
        self.table.reset(float(bankroll))
        self.metrics = LabMetrics(start=float(bankroll))
        self._metrics_json = None
        self._recorded_round = 0
        self.last_trace = None
        self.table.messages = [f"Session reset — bankroll €{float(bankroll):,.2f}, new shoe"]
        self._enqueue_state()

    async def _cmd_set_config(self, msg: dict[str, Any]) -> None:
        patch = msg.get("patch")
        if not isinstance(patch, dict) or not patch:
            raise CommandError('set_config needs a non-empty "patch" object')
        self._require_idle()
        self._require_between_rounds("changing the configuration")
        new = apply_patch(self.cfg, patch)  # ValueError on unknown keys / invalid values
        old_d, new_d = to_dict(self.cfg), to_dict(new)
        changed = {k for k in new_d if new_d[k] != old_d[k]}
        if not changed:
            self._enqueue_state()
            return
        t = self.table
        if changed & {"shoe", "rules"}:
            nt = self._new_table(t.balance, new)
            nt.round_id = t.round_id
            self.table = nt
            self.table.messages = ["Rules changed — new table, new shoe"]
        else:
            t.cfg = new
            if "crowd" in changed:
                t.crowd = Crowd(new.crowd, t.strategy, new.rules.charlie_cards)
                t.crowd.resample(t.rng)
                t.messages = [f"Crowd {'enabled' if new.crowd.enabled else 'disabled'}"]
        self.cfg = new
        self._agents.clear()
        await self._load_agent(self.agent_name)
        await self._send_hello()
        self._enqueue_state()

    async def _cmd_debug_rig(self, msg: dict[str, Any]) -> None:
        """Debug only (``ONEBJ_DEBUG=1``): make the next cards drawn be ``cards`` (codes like
        ``"8_of_hearts"`` or ``"8h"``), by swapping cards within the unseen part of the shoe."""
        cards = msg.get("cards")
        if not isinstance(cards, list) or not all(isinstance(c, str) for c in cards):
            raise CommandError('debug_rig needs "cards": ["8_of_hearts", ...]')
        rig_shoe(self.table, [parse_card(c) for c in cards])

    # ================================================================== engine driving
    def _new_table(self, bankroll: float, cfg: GameConfig | None = None) -> StreamingTable:
        return StreamingTable(cfg or self.cfg, seed=self.rng.randrange(2**31), strict=True, bankroll=bankroll)

    def _frame_delay(self) -> float:
        if self.autoplay:
            return min(0.6, max(0.03, self.delay_ms * 0.45 / 1000))
        return INTERACTIVE_FRAME_DELAY

    def _run(self, mutate: Callable[[], None]) -> None:
        """Apply an engine mutation, then queue one state per visible change plus the final state."""
        before = self._metrics()
        with FrameRecorder(self.table) as rec:
            mutate()
        self._record_round()
        final = self._state_msg()
        frames = rec.frames
        if frames and visual_signature(frames[-1]["game"]) == visual_signature(final["game"]):
            frames.pop()
        delay = self._frame_delay()
        for f in frames:
            self._enqueue(self._state_msg(f, final=False, metrics=before), delay)
        self._enqueue(final)

    def _record_round(self) -> None:
        rr = self.table.round_result
        if rr is not None and self.table.phase is Phase.SETTLED and rr.round_id != self._recorded_round:
            self._recorded_round = rr.round_id
            self.metrics.add(rr)
            self._metrics_json = None

    def _record_decision(self, action: Action) -> None:
        try:
            self.metrics.decision(action.value == registry.basic_action(self.table))
            self._metrics_json = None
        except Exception as e:  # noqa: BLE001  # pragma: no cover - never block a move on bookkeeping
            log.debug("agreement bookkeeping failed: %s", e)

    def _ai_step(self) -> None:
        t = self.table
        if t.phase in (Phase.BETTING, Phase.SETTLED):
            if t.balance < t.cfg.limits.main_min:
                raise CommandError("Bankroll exhausted — reset the session")
            _, bets, trace = registry.decide(self.agent, t)
            bets = self._affordable(bets)
            if trace is not None:
                trace.setdefault("extra", {})["bets"] = bets.to_dict()
            self.last_trace = trace
            self._run(lambda: t.place_bets(bets))
        elif t.phase is Phase.INSURANCE:
            _, take, trace = registry.decide(self.agent, t)
            action = Action.INSURE if take and Action.INSURE in t.legal_actions() else Action.NO_INSURANCE
            self.last_trace = trace
            self._run(lambda: t.act(action))
        elif t.phase is Phase.PLAYER:
            _, action, trace = registry.decide(self.agent, t)
            if action not in t.legal_actions():
                raise CommandError(f"{self.agent_name} chose an illegal action '{action.value}'")
            self._record_decision(action)
            self.last_trace = trace
            self._run(lambda: t.act(action))
        else:  # pragma: no cover - transient phases never persist
            raise CommandError("Nothing to decide right now")

    def _affordable(self, bets: Bets) -> Bets:
        """Scale an agent's bets down to the bankroll (side bets dropped first)."""
        bal = self.table.balance
        if bets.total <= bal + 1e-9:
            return bets
        b = Bets(**bets.to_dict())
        for name in reversed(SIDE_BETS):
            if b.total > bal + 1e-9:
                setattr(b, name, 0)
        if b.main > bal:
            b.main = math.floor(bal * 100) / 100
        return b

    async def _autoplay_loop(self) -> None:
        try:
            while self.autoplay and not self._closed:
                await self._idle.wait()
                between = self.table.phase in (Phase.BETTING, Phase.SETTLED)
                wait = (self.delay_ms * 1.6 + 150) if between else (self.delay_ms * 0.8 + 60)
                await asyncio.sleep(wait / 1000)
                if not self.autoplay or self._closed:
                    break
                if self.busy:
                    continue
                try:
                    self._ai_step()
                except Exception as e:  # noqa: BLE001 - reported to the client, autoplay stops
                    self.autoplay = False
                    await self._error(f"Autoplay stopped: {e}")
                    self._enqueue_state()
        except asyncio.CancelledError:
            pass

    # ================================================================== agents
    def _agent_names(self) -> list[str]:
        return registry.available_agents(self.checkpoints)

    async def _load_agent(self, name: str) -> None:
        agent = self._agents.get(name)
        if agent is None:
            if registry.is_nn(name):
                agent = await asyncio.to_thread(registry.make_agent, name, self.cfg, self.checkpoints)
            else:
                agent = registry.make_agent(name, self.cfg, self.checkpoints)
            self._agents[name] = agent
        self.agent, self.agent_name = agent, name
        if registry.is_nn(name):
            self.network = registry.describe(agent)
        elif self.network is None or not any(n.startswith("nn:") for n in self._agent_names()):
            self.network = await self._latest_network()

    async def _latest_network(self) -> dict[str, Any] | None:
        """Topology of the newest checkpoint, shown (without activations) while a baseline plays."""
        nn = [n for n in self._agent_names() if registry.is_nn(n)]
        if not nn:
            return None
        try:
            agent = self._agents.get(nn[0])
            if agent is None:
                agent = await asyncio.to_thread(registry.make_agent, nn[0], self.cfg, self.checkpoints)
                self._agents[nn[0]] = agent
            return registry.describe(agent)
        except Exception as e:  # noqa: BLE001 - the topology is optional
            log.warning("could not describe %s: %s", nn[0], e)
            return None

    # ================================================================== state messages
    def _metrics(self) -> dict[str, Any]:
        if self._metrics_json is None:
            self._metrics_json = self.metrics.to_dict()
        return self._metrics_json

    def _advice(self) -> dict[str, Any] | None:
        if not self.advice or self.autoplay or self.agent is None:
            return None
        if self.table.phase not in (Phase.BETTING, Phase.SETTLED, Phase.INSURANCE, Phase.PLAYER):
            return None
        try:
            _, decision, trace = registry.decide(self.agent, self.table)
        except Exception as e:  # noqa: BLE001 - advice is optional
            log.debug("advice failed: %s", e)
            return None
        if trace is not None and isinstance(decision, Bets):
            trace.setdefault("extra", {})["bets"] = self._affordable(decision).to_dict()
        return trace

    def _state_msg(self, frame: dict[str, Any] | None = None, final: bool = True,
                   metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        snap = frame if frame is not None else self.table.snapshot()
        _normalize_side_results(snap["game"], self.table if frame is None else None)
        return {
            "type": "state", "game": snap["game"], "shoe": snap["shoe"],
            "ai": {"agent": self.agent_name, "advice": self._advice() if final else None,
                   "last_trace": self.last_trace, "autoplay": self.autoplay},
            "metrics": metrics if metrics is not None else self._metrics(),
        }


# ---------------------------------------------------------------------- helpers
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _parse_bets(raw: Any) -> Bets:
    if not isinstance(raw, dict):
        raise CommandError('place_bets needs "bets": {"main": 10, ...}')
    unknown = set(raw) - set(BET_KEYS)
    if unknown:
        raise CommandError(f"Unknown bet(s): {', '.join(sorted(unknown))}")
    for k, v in raw.items():
        if v is not None and (not _is_number(v) or v < 0):
            raise CommandError(f"Bet '{k}' must be a non-negative number")
    if not raw.get("main"):
        raise CommandError("Place a main bet — side bets require a main bet")
    return Bets.from_dict(raw)


def _normalize_side_results(game: dict[str, Any], table: StreamingTable | None) -> None:
    """Protocol details on top of the engine snapshot: a lost side bet has outcome ``"lose"`` (not
    null), and a settled insurance bet is listed in ``side_bet_results`` like the side bets."""
    for r in game["side_bet_results"]:
        if r.get("outcome") is None:
            r["outcome"] = "lose"
    rr = table.round_result if table is not None else None
    if rr is not None and rr.insurance_stake and table.phase is Phase.SETTLED:
        won = rr.insurance_payout > 0
        game["side_bet_results"].append({
            "bet": "insurance", "outcome": "dealer_blackjack" if won else "lose",
            "odds": table.cfg.rules.insurance_payout if won else 0, "stake": rr.insurance_stake,
            "payout": rr.insurance_payout})


def rig_shoe(table: StreamingTable, cards: list[int]) -> None:
    """Make the next cards drawn be ``cards`` by swapping within the not-yet-dealt part of the shoe."""
    shoe = table.shoe
    for k, c in enumerate(cards):
        i = shoe.pos + k
        try:
            j = next(j for j in range(i, shoe.total_cards) if shoe.cards[j] == c)
        except StopIteration:
            raise CommandError(f"card {c} is not left in the shoe") from None
        shoe.cards[i], shoe.cards[j] = shoe.cards[j], shoe.cards[i]


def _json_default(o: Any) -> Any:
    for attr in ("tolist", "item"):  # numpy arrays / scalars, torch tensors
        fn = getattr(o, attr, None)
        if callable(fn):
            return fn()
    if hasattr(o, "value"):  # enums
        return o.value
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _sanitize(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def _dumps(msg: dict[str, Any]) -> str:
    try:
        return json.dumps(msg, separators=(",", ":"), allow_nan=False, default=_json_default)
    except ValueError:  # NaN / inf somewhere: send null instead
        return json.dumps(_sanitize(json.loads(json.dumps(msg, default=_json_default))), separators=(",", ":"))
