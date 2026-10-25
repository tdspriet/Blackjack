"""Agent registry for the server: baselines + trained NN checkpoints, and UI traces for baselines.

Agent names (protocol strings): ``"nn:<checkpoint>"`` for every checkpoint found by
:func:`onebj.ai.agent.list_checkpoints`, then ``"hilo"`` and ``"basic"``. The NN package is imported
lazily and defensively: without PyTorch weights (or before ``onebj.ai`` exists) the server simply
offers the baselines and ``hello.network`` is ``null``.

The baseline agents are rule based, so :func:`decide` fills in informative traces for them: per-action
EVs from the fresh-shoe solver (the same numbers the cash-out offer is based on) for play decisions, and
the analytic composition EVs of every bet for bet / insurance decisions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..agents.basic import BasicStrategyAgent, HiLoAgent
from ..analytics import insurance_ev, main_edge_estimate, side_bet_evs
from ..cards import BUCKET_OF
from ..config import GameConfig
from ..engine import Action, Bets, Observation, Phase, Table
from ..sidebets import SIDE_BETS

log = logging.getLogger("onebj.server")

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINES: tuple[str, ...] = ("hilo", "basic")
PLAY_LABELS = ["stand", "hit", "double", "split", "cashout"]
BET_LABELS = ["main", *SIDE_BETS, "insurance"]
_SOLVER_CODE = {"stand": "S", "hit": "H", "double": "D", "split": "P"}


def checkpoints_dir() -> Path:
    return Path(os.environ.get("ONEBJ_CHECKPOINTS", REPO_ROOT / "checkpoints"))


def _ai_module() -> Any | None:
    try:
        from ..ai import agent as ai_agent
    except Exception as e:  # noqa: BLE001 - ImportError before the package exists, or a broken install
        log.debug("NN agent unavailable: %s", e)
        return None
    return ai_agent


def list_checkpoints(directory: Path | None = None) -> list[dict[str, Any]]:
    """``[{name, path, created, metrics}]`` (newest first); empty when no NN is available."""
    d = directory or checkpoints_dir()
    if not d.exists():
        return []
    mod = _ai_module()
    if mod is None or not hasattr(mod, "list_checkpoints"):
        return []
    try:
        cks = list(mod.list_checkpoints(str(d)))
    except Exception as e:  # noqa: BLE001 - checkpoints are optional
        log.warning("list_checkpoints failed: %s", e)
        return []
    out = []
    for c in cks:
        c = dict(c)
        if not c.get("name"):
            continue
        c.setdefault("path", str(d / f"{c['name']}.pt"))
        c.setdefault("created", None)
        c.setdefault("metrics", None)
        out.append(c)
    return out


def available_agents(directory: Path | None = None) -> list[str]:
    return [f"nn:{c['name']}" for c in list_checkpoints(directory)] + list(BASELINES)


def is_nn(spec: str) -> bool:
    return spec.startswith("nn:")


def _set_trace(agent: Any, on: bool) -> None:
    """Toggle trace generation on an NN agent, whatever the attribute is called."""
    if hasattr(agent, "set_trace") and callable(agent.set_trace):
        agent.set_trace(on)
        return
    for attr in ("trace", "traces", "with_trace", "trace_enabled"):
        if hasattr(agent, attr) and isinstance(getattr(agent, attr), bool):
            setattr(agent, attr, on)
            return


def make_agent(spec: str, cfg: GameConfig, directory: Path | None = None, trace: bool = True) -> Any:
    """Instantiate an agent from its protocol name. Raises ``ValueError`` for unknown names."""
    if spec == "basic":
        return BasicStrategyAgent(cfg)
    if spec == "hilo":
        return HiLoAgent(cfg)
    if is_nn(spec):
        name = spec[3:]
        mod = _ai_module()
        if mod is None:
            raise ValueError("the NN agent is not available (onebj.ai could not be imported)")
        match = next((c for c in list_checkpoints(directory) if c["name"] == name), None)
        if match is None:
            raise ValueError(f"unknown checkpoint {name!r}")
        agent = mod.NNAgent.load(match.get("path") or name, cfg)
        _set_trace(agent, trace)
        return agent
    raise ValueError(f"unknown agent {spec!r} (available: {', '.join(available_agents(directory))})")


def describe(agent: Any) -> dict[str, Any] | None:
    """``NNAgent.describe()`` for the hello message (None for agents without a network)."""
    fn = getattr(agent, "describe", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - the topology is optional
        log.warning("describe() failed: %s", e)
        return None


# ---------------------------------------------------------------------- decisions with traces
def decide(agent: Any, table: Table, obs: Observation | None = None) -> tuple[str, Any, dict | None]:
    """Ask ``agent`` for the pending decision without applying it.

    Returns ``(kind, decision, trace)``: kind ``"bet"`` (decision = :class:`Bets`), ``"insurance"``
    (decision = bool) or ``"play"`` (decision = :class:`Action`). Raises ``RuntimeError`` when nothing
    is pending.
    """
    obs = obs or table.observation()
    baseline = isinstance(agent, (BasicStrategyAgent, HiLoAgent))
    ph = table.phase
    if ph in (Phase.BETTING, Phase.SETTLED):
        bets, trace = agent.decide_bets(obs)
        if baseline:
            trace = _bet_trace(table, obs, bets, trace)
        elif trace is not None:
            trace.setdefault("extra", {}).setdefault("bets", bets.to_dict())
        return "bet", bets, trace
    if ph is Phase.INSURANCE:
        take, trace = agent.decide_insurance(obs)
        if baseline:
            trace = _insurance_trace(table, obs, take, trace)
        return "insurance", bool(take), trace
    if ph is Phase.PLAYER:
        action, trace = agent.decide_action(obs)
        action = Action(action)
        if baseline:
            trace = _play_trace(table, obs, action, trace)
        return "play", action, trace
    raise RuntimeError(f"nothing to decide in phase {ph.value}")


def basic_action(table: Table) -> str:
    """Basic-strategy action for the pending play decision."""
    total, soft, n, up, can_double, pair, can_split, _ = table.decision_state()
    return table.strategy.action(total, soft, n, up, can_double, pair, can_split)


def _solver(table: Table) -> Any:
    if table.cashout_table is not None:
        return table.cashout_table.solver
    from ..strategy.solver import SolverRules, solve
    return solve(SolverRules.from_config(table.cfg.rules))


def _play_trace(table: Table, obs: Observation, action: Action, trace: dict | None) -> dict:
    """Replace the one-hot outputs of the rule-based agents by fresh-shoe solver EVs per action."""
    trace = dict(trace or {"kind": "play", "features": [], "layers": [], "reference": {}, "extra": {}})
    h = obs.hand
    legal = {a.value for a in obs.legal_actions}
    values: list[float | None] = [None] * len(PLAY_LABELS)
    try:
        up = BUCKET_OF[obs.dealer_up]
        evs = _solver(table).evs(h.total, h.soft, h.num_cards, up, h.pair, h.can_double, h.can_split,
                                 h.is_split)
        for i, lab in enumerate(PLAY_LABELS):
            if lab == "cashout":
                if h.cashout_offer is not None and h.bet:
                    values[i] = round(h.cashout_offer / h.bet - 1, 4)
            elif lab in legal and _SOLVER_CODE[lab] in evs:
                values[i] = round(evs[_SOLVER_CODE[lab]], 4)
        unit = "EV per unit bet (fresh-shoe solver)"
    except Exception as e:  # noqa: BLE001  # pragma: no cover - defensive: keep the rule-based outputs
        log.debug("solver EVs unavailable: %s", e)
        values = [1.0 if lab == action.value else 0.0 for lab in PLAY_LABELS]
        unit = "chosen (rule-based)"
    trace["outputs"] = {"labels": PLAY_LABELS, "values": values, "legal": [lab in legal for lab in PLAY_LABELS],
                        "unit": unit}
    trace["chosen"] = action.value
    ref = dict(trace.get("reference") or {})
    ref.setdefault("basic_strategy", basic_action(table))
    ref.setdefault("hilo_deviation", None)
    trace["reference"] = ref
    return trace


def _bet_trace(table: Table, obs: Observation, bets: Bets, trace: dict | None) -> dict:
    """Bet decision trace: analytic EV per unit stake of every bet for the current composition."""
    cfg = table.cfg
    evs = side_bet_evs(obs.unseen_card_counts, cfg)
    main = main_edge_estimate(obs.unseen_counts, cfg.rules, obs.num_decks)
    ins = insurance_ev(obs.unseen_counts, cfg.rules.insurance_payout)
    pred = {"main": main, **evs, "insurance": ins}
    tc = obs.true_counts.get("hilo", 0.0)
    trace = dict(trace or {})
    feats = trace.get("features") or [
        {"name": "hilo_true_count", "value": tc / 10, "raw": round(tc, 2)},
        {"name": "decks_unseen", "value": obs.decks_unseen / obs.num_decks, "raw": round(obs.decks_unseen, 2)}]
    trace.update(
        kind="bet", features=feats, layers=trace.get("layers") or [],
        outputs={"labels": BET_LABELS, "values": [round(pred[k], 5) for k in BET_LABELS],
                 "legal": [True] * len(BET_LABELS), "unit": "EV per unit stake (analytic, current composition)"},
        chosen=f"{bets.main:g}",
        reference={"basic_strategy": None, "hilo_deviation": None},
    )
    extra = dict(trace.get("extra") or {})
    extra["predicted_ev"] = {k: round(v, 5) for k, v in pred.items()}
    extra["bets"] = bets.to_dict()
    extra["bet_fraction"] = bets.total / max(obs.bankroll, 1e-9)
    trace["extra"] = extra
    return trace


def _insurance_trace(table: Table, obs: Observation, take: bool, trace: dict | None) -> dict:
    ev = insurance_ev(obs.unseen_counts, table.cfg.rules.insurance_payout)
    tc = obs.true_counts.get("hilo", 0.0)
    trace = dict(trace or {})
    trace.update(
        kind="insurance",
        features=trace.get("features") or [{"name": "hilo_true_count", "value": tc / 10, "raw": round(tc, 2)}],
        layers=trace.get("layers") or [],
        outputs={"labels": ["insure", "no_insurance"], "values": [round(ev, 4), 0.0], "legal": [True, True],
                 "unit": "EV per unit insurance stake (current composition)"},
        chosen="insure" if take else "no_insurance",
    )
    ref = dict(trace.get("reference") or {})
    ref["basic_strategy"] = "no_insurance"
    ref.setdefault("hilo_deviation", None)
    trace["reference"] = ref
    return trace
