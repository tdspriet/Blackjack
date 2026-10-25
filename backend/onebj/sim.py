"""Fast simulation: play N rounds with an agent, optionally over several processes.

>>> from onebj.sim import simulate
>>> m = simulate(agent="basic", rounds=200_000, seed=1, workers=8)   # dict of metrics
>>> m["edge"], m["side_bets"]["perfect_pairs"]["rtp"]

``agent`` is ``"basic"``, ``"basic_sides"`` (flat minimum on every side bet), ``"hilo"``, ``"hilo_nosides"``
or a picklable factory ``f(config) -> Agent``.
Agents exposing ``fast_bets/fast_insurance/fast_action`` (the baselines) skip Observation building.
For in-process use with progress reporting (server ``simulate`` command) use :func:`run_rounds`.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import GameConfig, load_config, to_dict, from_dict
from .engine import Action, Phase, RoundResult, Table
from .sidebets import SIDE_BETS

AgentSpec = str | Callable[[GameConfig], Any]


def make_agent(spec: AgentSpec, cfg: GameConfig) -> Any:
    from .agents.basic import BasicStrategyAgent, HiLoAgent
    if callable(spec):
        return spec(cfg)
    if spec == "basic":
        return BasicStrategyAgent(cfg)
    if spec == "basic_sides":
        return BasicStrategyAgent(cfg, side_bets=SIDE_BETS, name="basic_sides")
    if spec == "hilo":
        return HiLoAgent(cfg)
    if spec == "hilo_nosides":
        return HiLoAgent(cfg, side_bets=(), name="hilo_nosides")
    raise ValueError(f"unknown agent {spec!r} (use 'basic', 'basic_sides', 'hilo', 'hilo_nosides' or a factory)")


def play_round(table: Table, agent: Any) -> RoundResult:
    """Play one full round with ``agent`` (fast path if available) and return its result."""
    if hasattr(agent, "fast_action"):
        table.place_bets(agent.fast_bets(table))
        while True:
            ph = table.phase
            if ph is Phase.PLAYER:
                table.act(agent.fast_action(table, table.decision_state()))
            elif ph is Phase.INSURANCE:
                table.act(Action.INSURE if agent.fast_insurance(table) else Action.NO_INSURANCE)
            else:
                break
    else:
        bets, _ = agent.decide_bets(table.observation())
        table.place_bets(bets)
        while True:
            ph = table.phase
            if ph is Phase.PLAYER:
                table.act(agent.decide_action(table.observation())[0])
            elif ph is Phase.INSURANCE:
                take, _ = agent.decide_insurance(table.observation())
                table.act(Action.INSURE if take else Action.NO_INSURANCE)
            else:
                break
    assert table.round_result is not None
    return table.round_result


@dataclass
class Metrics:
    """Running totals over simulated rounds. ``to_dict`` matches the SPEC §6 ``metrics`` object."""

    rounds: int = 0
    net: float = 0.0
    net_sq: float = 0.0
    wagered: float = 0.0
    main_bet: float = 0.0  # sum of original main stakes
    main_net: float = 0.0
    main_net_sq_unit: float = 0.0  # sum of (main_net / main_bet)^2
    main_wagered: float = 0.0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    blackjacks: int = 0
    charlies: int = 0
    cashouts: int = 0  # cashed-out hands (also classified as win/loss/push by payout vs stake)
    cashout_net: float = 0.0  # sum of (payout - stake) of cashed-out hands
    doubles: int = 0
    splits: int = 0
    dealer_busts: int = 0
    insurance_wagered: float = 0.0
    insurance_net: float = 0.0
    side: dict[str, list[float]] = field(default_factory=lambda: {k: [0.0, 0.0, 0.0, 0.0, 0.0] for k in SIDE_BETS})
    # side[name] = [wagered, returned, count, hits, sum of (net/stake)^2]
    balance: float = 0.0
    peak: float = 0.0
    max_drawdown: float = 0.0
    bankroll: list[list[float]] = field(default_factory=list)
    seconds: float = 0.0

    def add(self, r: RoundResult) -> None:
        self.rounds += 1
        n = r.net
        self.net += n
        self.net_sq += n * n
        self.wagered += r.wagered
        mb = r.main_bet
        mn = r.main_returned - r.main_wagered
        self.main_bet += mb
        self.main_net += mn
        u = mn / mb
        self.main_net_sq_unit += u * u
        self.main_wagered += r.main_wagered
        if r.dealer_bust:
            self.dealer_busts += 1
        if len(r.hands) > 1:
            self.splits += 1
        for h in r.hands:
            res = h.result
            if res == "cashout":
                self.cashouts += 1
                self.cashout_net += h.payout - h.bet
                res = "win" if h.payout > h.bet else ("lose" if h.payout < h.bet else "push")
            if res == "lose":
                self.losses += 1
            elif res == "push":
                self.pushes += 1
            else:
                self.wins += 1
                if res == "blackjack":
                    self.blackjacks += 1
                elif res == "charlie":
                    self.charlies += 1
            if h.doubled:
                self.doubles += 1
        if r.insurance_stake:
            self.insurance_wagered += r.insurance_stake
            self.insurance_net += r.insurance_payout - r.insurance_stake
        for s in r.side_bets:
            acc = self.side[s.bet]
            acc[0] += s.stake
            acc[1] += s.payout
            acc[2] += 1
            if s.payout > s.stake:
                acc[3] += 1
            x = (s.payout - s.stake) / s.stake
            acc[4] += x * x
        self.balance += n
        if self.balance > self.peak:
            self.peak = self.balance
        dd = self.peak - self.balance
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def merge(self, o: "Metrics") -> None:
        for k in ("rounds", "net", "net_sq", "wagered", "main_bet", "main_net", "main_net_sq_unit", "main_wagered",
                  "wins", "losses", "pushes", "blackjacks", "charlies", "cashouts", "cashout_net", "doubles", "splits", "dealer_busts",
                  "insurance_wagered", "insurance_net"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        for name, acc in o.side.items():
            mine = self.side[name]
            for i in range(5):
                mine[i] += acc[i]
        self.max_drawdown = max(self.max_drawdown, o.max_drawdown)  # per-worker drawdown (approximate)
        self.seconds = max(self.seconds, o.seconds)
        self.balance += o.balance

    def to_dict(self) -> dict[str, Any]:
        n = max(self.rounds, 1)
        mean = self.net / n
        var = max(self.net_sq / n - mean * mean, 0.0)
        edge = self.main_net / self.main_bet if self.main_bet else 0.0
        var_u = max(self.main_net_sq_unit / n - (self.main_net / max(self.main_bet, 1e-12)) ** 2, 0.0)
        sides = {}
        for name, (w, ret, cnt, hits, sq) in self.side.items():
            rtp = ret / w if w else None
            m = (ret - w) / w if w else 0.0
            sd = math.sqrt(max(sq / cnt - m * m, 0.0)) if cnt else 0.0
            sides[name] = {"wagered": w, "net": ret - w, "rtp": rtp, "bets": int(cnt), "wins": int(hits),
                           "rtp_se": sd / math.sqrt(cnt) if cnt else None}
        return {
            "rounds": self.rounds, "net": self.net, "wagered": self.wagered, "ev_per_round": mean,
            "std_per_round": math.sqrt(var), "edge": edge, "main_rtp": 1 + edge,
            "edge_se": math.sqrt(var_u / n), "main_std_per_unit": math.sqrt(var_u),
            "wins": self.wins, "losses": self.losses, "pushes": self.pushes, "blackjacks": self.blackjacks,
            "charlies": self.charlies, "cashouts": self.cashouts, "cashout_net": self.cashout_net,
            "doubles": self.doubles, "splits": self.splits,
            "dealer_bust_rate": self.dealer_busts / n,
            "insurance": {"wagered": self.insurance_wagered, "net": self.insurance_net},
            "side_bets": sides, "bankroll": self.bankroll, "max_drawdown": self.max_drawdown,
            "rounds_per_sec": self.rounds / self.seconds if self.seconds else None,
        }


def run_rounds(table: Table, agent: Any, rounds: int, metrics: Metrics | None = None,
               progress: Callable[[Metrics], Any] | None = None, progress_every: int = 5000,
               bankroll_every: int = 0, should_stop: Callable[[], bool] | None = None) -> Metrics:
    """Play ``rounds`` rounds on an existing table (single process). ``progress(metrics)`` is called
    every ``progress_every`` rounds; ``bankroll_every`` > 0 samples ``[round, balance]`` points."""
    m = metrics or Metrics()
    t0 = time.perf_counter()
    add = m.add
    for i in range(1, rounds + 1):
        add(play_round(table, agent))
        if bankroll_every and i % bankroll_every == 0:
            m.bankroll.append([m.rounds, table.balance])
        if progress is not None and i % progress_every == 0:
            m.seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            progress(m)
            if should_stop is not None and should_stop():
                break
    m.seconds += time.perf_counter() - t0
    return m


def _worker(args: tuple[dict, AgentSpec, int, int]) -> Metrics:
    cfg_dict, spec, rounds, seed = args
    cfg = from_dict(cfg_dict)
    table = Table(cfg, seed=seed, record_events=False)
    return run_rounds(table, make_agent(spec, cfg), rounds)


def simulate(config: GameConfig | None = None, agent: AgentSpec = "basic", rounds: int = 100_000,
             seed: int = 0, workers: int | None = None) -> dict[str, Any]:
    """Simulate ``rounds`` rounds split over ``workers`` processes (default: all cores) and return
    merged metrics (``to_dict`` form, plus ``workers``)."""
    cfg = config or load_config()
    workers = workers or mp.cpu_count()
    workers = max(1, min(workers, rounds))
    per = [rounds // workers + (1 if i < rounds % workers else 0) for i in range(workers)]
    jobs = [(to_dict(cfg), agent, per[i], seed * 1_000_003 + i) for i in range(workers)]
    t0 = time.perf_counter()
    if workers == 1:
        parts = [_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_worker, jobs)
    total = Metrics()
    for p in parts:
        total.merge(p)
    out = total.to_dict()
    out["workers"] = workers
    out["wall_seconds"] = time.perf_counter() - t0
    out["rounds_per_sec_per_core"] = out["rounds_per_sec"]
    out["rounds_per_sec"] = rounds / out["wall_seconds"]
    return out
