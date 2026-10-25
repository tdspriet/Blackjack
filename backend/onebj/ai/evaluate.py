"""Evaluate agents: full-game simulation, paired ("duplicate") play comparison, learned deviations.

CLI::

    uv run python -m onebj.ai.evaluate --agents nn_v1,basic,hilo --rounds 2000000
    uv run python -m onebj.ai.evaluate --agents nn_v1,nn_v1:flat,basic,hilo_nosides,hilo --rounds 2000000 \
        --duplicate 2000000 --analysis --json checkpoints/nn_v1.eval.json --markdown docs/nn_eval.md

Agent specs
    ``basic`` · ``basic_sides`` · ``hilo`` · ``hilo_nosides`` (see :mod:`onebj.sim`),
    ``<checkpoint>`` (NN: Kelly bets + side bets + insurance), ``<checkpoint>:flat`` (NN play only: flat
    minimum bet, no side bets, no insurance), ``<checkpoint>:nosides`` (Kelly bets + insurance, no side bets).

Full-game metrics per agent: EV per round (money and in units of the agent's minimum bet), edge on the
amount wagered on the main bet, std per round, side-bet and insurance results, agreement with basic
strategy (share of play decisions identical to basic strategy), cash-out usage and the most frequent
disagreements with basic strategy.

**Duplicate evaluation** (``--duplicate N``): every round is played twice from the same table state (same
card order) — once by the NN (flat bet, no sides, no insurance) and once by basic strategy — so the
difference is non-zero only where the decisions differ. This measures the play-quality gain with a
standard error ~10× smaller than comparing two independent simulations.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..cards import BUCKETS
from ..config import GameConfig, from_dict, load_config, to_dict
from ..counting import hilo_true
from ..engine import Action, Bets, Phase, Table
from ..sim import Metrics, make_agent
from .policy import BasicPolicy

BASELINES = ("basic", "basic_sides", "hilo", "hilo_nosides", "solver", "solver_nocash", "solver_plain")
FLAT = {"max_units": 1, "side_bets": False, "insurance_threshold": 1e9}
NOSIDES = {"side_bets": False}


# ---------------------------------------------------------------------- agent construction
def build_agent(spec: Any, cfg: GameConfig) -> Any:
    """``spec``: baseline name, ``"<ckpt>[:flat|:nosides]"`` or ``("weights", weights, meta, bet_cfg)``."""
    from .agent import NNAgent
    if isinstance(spec, tuple) and spec[0] == "weights":
        from .features import FeatureEncoder, FeatureSpec
        from .model import NumpyNet
        _, weights, spec_d, bet = spec
        return NNAgent(NumpyNet(weights), FeatureEncoder(FeatureSpec.from_dict(spec_d)), cfg, {}, "candidate",
                       None, bet, trace=False)
    if spec in ("solver", "solver_nocash", "solver_plain"):
        from .oracle import SolverAgent
        ag = SolverAgent(cfg, cashout=spec != "solver_nocash", name=spec)
        if spec == "solver_plain":  # cashes out now, but values hit/split without later cash-outs
            ag.rtp = None
        return ag
    if spec in BASELINES:
        return make_agent(spec, cfg)
    name, _, mode = str(spec).partition(":")
    bet = {"flat": FLAT, "flat_nocash": FLAT, "nosides": NOSIDES, "": None}.get(mode)
    if mode and bet is None:
        raise ValueError(f"unknown agent mode {mode!r} in {spec!r}")
    agent = NNAgent.load(name, cfg, trace=False, bet=bet)
    if mode == "flat_nocash":
        agent.allow_cashout = False
    return agent


def spec_label(spec: Any) -> str:
    return "candidate" if isinstance(spec, tuple) else str(spec)


# ---------------------------------------------------------------------- full-game evaluation
class Stats:
    """Decision-level statistics collected next to :class:`onebj.sim.Metrics`."""

    def __init__(self) -> None:
        self.decisions = 0
        self.agree = 0
        self.actions: Counter[str] = Counter()
        self.disagree: Counter[str] = Counter()
        self.disagree_tc: dict[str, float] = {}
        self.cashouts = 0
        self.cashout_offer = 0.0  # sum of offers per unit bet
        self.cash_ev = 0.0  # sum of cash-out EV (offer / bet - 1)
        self.cash_hold_solver = 0.0  # sum of the best hold EV (composition solver) in the same states
        self.cash_hold_mc = 0.0  # sum of a realised counterfactual "hold" return (1 resampled rollout each)
        self.dec_by_tc: Counter[int] = Counter()  # decisions per TC bucket
        self.cash_by_tc: Counter[int] = Counter()  # cash-outs per TC bucket
        self.dec_by_class: Counter[str] = Counter()  # decisions per hand class
        self.cash_by_class: Counter[str] = Counter()
        self.cash_tc_by_class: dict[str, float] = {}
        self.cash_gain_by_class: dict[str, float] = {}  # sum of (cash EV - best solver hold EV)
        self.ins_offered = 0
        self.ins_taken = 0
        self.units_sum = 0.0
        self.max_bet = 0.0
        self.unit = 1.0
        self.rounds_spread = Counter()

    def merge(self, o: Stats) -> None:
        self.decisions += o.decisions
        self.agree += o.agree
        self.actions.update(o.actions)
        self.disagree.update(o.disagree)
        for k, v in o.disagree_tc.items():
            self.disagree_tc[k] = self.disagree_tc.get(k, 0.0) + v
        self.cashouts += o.cashouts
        self.cashout_offer += o.cashout_offer
        self.cash_ev += o.cash_ev
        self.cash_hold_solver += o.cash_hold_solver
        self.cash_hold_mc += o.cash_hold_mc
        self.dec_by_tc.update(o.dec_by_tc)
        self.cash_by_tc.update(o.cash_by_tc)
        self.dec_by_class.update(o.dec_by_class)
        self.cash_by_class.update(o.cash_by_class)
        for k, v in o.cash_tc_by_class.items():
            self.cash_tc_by_class[k] = self.cash_tc_by_class.get(k, 0.0) + v
        for k, v in o.cash_gain_by_class.items():
            self.cash_gain_by_class[k] = self.cash_gain_by_class.get(k, 0.0) + v
        self.ins_offered += o.ins_offered
        self.ins_taken += o.ins_taken
        self.units_sum += o.units_sum
        self.max_bet = max(self.max_bet, o.max_bet)
        self.unit = o.unit
        self.rounds_spread.update(o.rounds_spread)


def _cell(state: tuple) -> str:
    total, soft, n, up, can_double, pair, can_split, is_split = state
    up_s = BUCKETS[up]
    if pair >= 0 and can_split:
        return f"P{BUCKETS[pair]}{BUCKETS[pair]}v{up_s}"
    tag = "S" if soft else "H"
    nc = "" if n == 2 else f"/{n}c"
    return f"{tag}{total}v{up_s}{nc}"


def hand_class(state: tuple) -> str:
    """Coarse class of a play decision for the cash-out breakdown."""
    total, soft, n, up, _cd, pair, can_split, _ = state
    if pair >= 0 and can_split:
        return "pair"
    if n >= 3:
        return "3+ cards"
    if soft:
        return "soft 2-card"
    if total <= 11:
        return "hard <=11"
    if total >= 17:
        return "hard 17-21"
    return "hard 12-16 v " + ("2-6" if 1 <= up <= 5 else "7-A")


def _cashout_diagnostic(table: Table, agent: Any, state: tuple, st: Stats, tc: float = 0.0) -> float:
    """Compare a cash-out with holding in the same state: the best hold EV from the composition solver
    and one realised counterfactual round (resampled unseen cards, best solver hold action first, then the
    agent's own policy), both per unit of the hand's original bet."""
    from ..strategy.solver import SolverRules
    from .oracle import composition_solver, state_evs
    h = table.hands[table.active]
    offer = table.cashout_offer() or 0.0
    st.cashouts += 1
    st.cashout_offer += offer / h.bet
    st.cash_ev += offer / h.bet - 1.0
    rules = SolverRules.from_config(table.cfg.rules)
    evs = state_evs(composition_solver(rules, tuple(table.shoe.unseen_buckets()), table.cfg.rules.cashout_rtp),
                    state)
    best = max(evs, key=evs.__getitem__)
    st.cash_hold_solver += evs[best]
    gain = offer / h.bet - 1.0 - evs[best]
    i0, nh0, bet0 = table.active, len(table.hands), h.bet
    c = table.clone(resample_unseen=True, seed=st.cashouts)
    c.act(best)
    while c.phase is Phase.PLAYER:
        c.act(agent.fast_action(c, c.decision_state()))
    rr = c.round_result
    nd = len(rr.hands) - nh0 + 1
    st.cash_hold_mc += sum(x.payout - x.bet for x in rr.hands[i0:i0 + nd]) / bet0
    return gain


def play_round_stats(table: Table, agent: Any, m: Metrics, st: Stats) -> None:
    bets = agent.fast_bets(table)
    table.place_bets(bets)
    st.units_sum += bets.main / st.unit
    st.max_bet = max(st.max_bet, bets.main)
    st.rounds_spread[int(round(bets.main / st.unit))] += 1
    strat = table.strategy
    while True:
        ph = table.phase
        if ph is Phase.PLAYER:
            state = table.decision_state()
            a = agent.fast_action(table, state)
            total, soft, n, up, can_double, pair, can_split, _ = state
            b = strat.action(total, soft, n, up, can_double, pair, can_split)
            st.decisions += 1
            st.actions[a] += 1
            if a == b:
                st.agree += 1
            else:
                key = f"{_cell(state)}: {a} (basic {b})"
                st.disagree[key] += 1
                st.disagree_tc[key] = st.disagree_tc.get(key, 0.0) + hilo_true(table.shoe.seen_bucket,
                                                                                table.shoe.num_decks)
            tc = hilo_true(table.shoe.seen_bucket, table.shoe.num_decks)
            tcb = max(-4, min(4, math.floor(tc + 0.5)))
            st.dec_by_tc[tcb] += 1
            hc = hand_class(state)
            st.dec_by_class[hc] += 1
            if a == "cashout":
                st.cash_by_tc[tcb] += 1
                st.cash_by_class[hc] += 1
                st.cash_tc_by_class[hc] = st.cash_tc_by_class.get(hc, 0.0) + tc
                g = _cashout_diagnostic(table, agent, state, st, tc)
                st.cash_gain_by_class[hc] = st.cash_gain_by_class.get(hc, 0.0) + g
            table.act(a)
        elif ph is Phase.INSURANCE:
            take = agent.fast_insurance(table)
            st.ins_offered += 1
            st.ins_taken += int(take)
            table.act(Action.INSURE if take else Action.NO_INSURANCE)
        else:
            break
    m.add(table.round_result)


def _eval_worker(args: tuple) -> tuple[Metrics, Stats]:
    cfg_d, spec, rounds, seed = args
    cfg = from_dict(cfg_d)
    agent = build_agent(spec, cfg)
    table = Table(cfg, seed=seed, record_events=False)
    m, st = Metrics(), Stats()
    st.unit = float(getattr(agent, "unit", None) or getattr(getattr(agent, "bet_cfg", None), "unit", 1.0) or 1.0)
    t0 = time.perf_counter()
    for _ in range(rounds):
        play_round_stats(table, agent, m, st)
    m.seconds = time.perf_counter() - t0
    return m, st


def _pool_map(fn: Any, jobs: list, workers: int) -> list:
    if workers <= 1 or len(jobs) == 1:
        return [fn(j) for j in jobs]
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        return pool.map(fn, jobs)


def evaluate_agent(spec: Any, cfg: GameConfig | None = None, rounds: int = 1_000_000, seed: int = 0,
                   workers: int | None = None) -> dict[str, Any]:
    """Full-game evaluation of one agent (multiprocessing). Same seeds for every agent."""
    cfg = cfg or load_config()
    workers = workers or mp.cpu_count()
    workers = max(1, min(workers, rounds))
    per = [rounds // workers + (1 if i < rounds % workers else 0) for i in range(workers)]
    jobs = [(to_dict(cfg), spec, per[i], seed * 1_000_003 + i) for i in range(workers)]
    t0 = time.perf_counter()
    parts = _pool_map(_eval_worker, jobs, workers)
    m, st = Metrics(), Stats()
    for pm, ps in parts:
        m.merge(pm)
        st.merge(ps)
    d = m.to_dict()
    n = max(m.rounds, 1)
    u = st.unit
    d.pop("bankroll", None)
    d["agent"] = spec_label(spec)
    d["unit"] = u
    d["ev_per_round_units"] = d["ev_per_round"] / u
    d["ev_se_units"] = d["std_per_round"] / u / math.sqrt(n)
    d["main_ev_per_round_units"] = m.main_net / n / u
    d["avg_bet_units"] = st.units_sum / n
    d["max_bet"] = st.max_bet
    d["edge_on_wagered_main"] = m.main_net / m.main_wagered if m.main_wagered else 0.0
    d["agreement_basic"] = st.agree / st.decisions if st.decisions else None
    d["decisions"] = st.decisions
    d["actions"] = dict(st.actions)
    d["cashouts_per_1000_rounds"] = 1000 * st.cashouts / n
    d["cashout_mean_offer"] = st.cashout_offer / st.cashouts if st.cashouts else None
    k = st.cashouts
    d["cashout_diagnostic"] = {
        "cashouts": k,
        "mean_cashout_ev": st.cash_ev / k if k else None,
        "mean_hold_ev_solver": st.cash_hold_solver / k if k else None,
        "mean_hold_return_mc": st.cash_hold_mc / k if k else None,
        "gain_per_cashout_vs_solver_hold": (st.cash_ev - st.cash_hold_solver) / k if k else None,
        "cashout_rate_by_tc": {str(t): st.cash_by_tc[t] / st.dec_by_tc[t] for t in sorted(st.dec_by_tc)
                               if st.dec_by_tc[t]},
        "cashout_rate_overall": k / st.decisions if st.decisions else None,
        "mean_tc_at_cashout": sum(st.cash_tc_by_class.values()) / k if k else None,
        "by_class": {c: {"decisions_per_1000_rounds": 1000 * st.dec_by_class[c] / n,
                         "cashout_rate": st.cash_by_class[c] / st.dec_by_class[c],
                         "mean_tc_at_cashout": (st.cash_tc_by_class[c] / st.cash_by_class[c])
                         if st.cash_by_class[c] else None,
                         "gain_per_cashout_vs_solver_hold": (st.cash_gain_by_class[c] / st.cash_by_class[c])
                         if st.cash_by_class[c] else None}
                     for c in sorted(st.dec_by_class)},
    }
    d["insurance_offered"] = st.ins_offered
    d["insurance_taken"] = st.ins_taken
    top = st.disagree.most_common(25)
    d["top_disagreements"] = [{"cell": k, "per_1000_rounds": 1000 * c / n, "mean_tc": st.disagree_tc[k] / c}
                              for k, c in top]
    d["bet_spread"] = {str(k): v / n for k, v in sorted(st.rounds_spread.items())}
    d["wall_seconds"] = time.perf_counter() - t0
    return d


# ---------------------------------------------------------------------- duplicate (paired) evaluation
def _dup_worker(args: tuple) -> dict[str, float]:
    cfg_d, spec_a, spec_b, rounds, seed = args
    cfg = from_dict(cfg_d)
    a = build_agent(spec_a, cfg)
    b = BasicPolicy() if spec_b == "basic" else build_agent(spec_b, cfg)
    table = Table(cfg, seed=seed, record_events=False)
    one = Bets(main=1)
    sa = sb = sd = sd2 = 0.0
    sa2 = 0.0
    dec = agree = differ_rounds = 0
    strat = table.strategy
    for _ in range(rounds):
        other = table.clone(resample_unseen=False, seed=0)
        table.place_bets(one)
        while True:
            ph = table.phase
            if ph is Phase.PLAYER:
                state = table.decision_state()
                act = a.fast_action(table, state)
                total, soft, n, up, cd, pair, cs, _ = state
                dec += 1
                if act == strat.action(total, soft, n, up, cd, pair, cs):
                    agree += 1
                table.act(act)
            elif ph is Phase.INSURANCE:
                table.act(Action.NO_INSURANCE)
            else:
                break
        ra = table.round_result.main_net
        other.place_bets(one)
        while True:
            ph = other.phase
            if ph is Phase.PLAYER:
                if isinstance(b, BasicPolicy):
                    other.act(b.action(other))
                else:
                    other.act(b.fast_action(other, other.decision_state()))
            elif ph is Phase.INSURANCE:
                other.act(Action.NO_INSURANCE)
            else:
                break
        rb = other.round_result.main_net
        d = ra - rb
        sa += ra
        sa2 += ra * ra
        sb += rb
        sd += d
        sd2 += d * d
        if d != 0:
            differ_rounds += 1
    return {"n": rounds, "sa": sa, "sa2": sa2, "sb": sb, "sd": sd, "sd2": sd2, "dec": dec, "agree": agree,
            "differ": differ_rounds}


def duplicate_eval(spec_a: Any, spec_b: Any = "basic", cfg: GameConfig | None = None, rounds: int = 500_000,
                   seed: int = 0, workers: int | None = None) -> dict[str, Any]:
    """Paired flat-bet comparison of ``spec_a`` (played flat: 1 unit, no sides, no insurance) vs ``spec_b``
    on identical card sequences. Returns EVs per round, the difference and its standard error."""
    cfg = cfg or load_config()
    if isinstance(spec_a, tuple):
        spec_a = (*spec_a[:3], FLAT)
    elif spec_a not in BASELINES and not str(spec_a).endswith(":flat_nocash"):
        spec_a = f"{str(spec_a).partition(':')[0]}:flat"
    workers = workers or mp.cpu_count()
    workers = max(1, min(workers, rounds))
    per = [rounds // workers + (1 if i < rounds % workers else 0) for i in range(workers)]
    jobs = [(to_dict(cfg), spec_a, spec_b, per[i], seed * 999_983 + i) for i in range(workers)]
    t0 = time.perf_counter()
    parts = _pool_map(_dup_worker, jobs, workers)
    tot = {k: sum(p[k] for p in parts) for k in parts[0]}
    n = tot["n"]
    mean_d = tot["sd"] / n
    var_d = max(tot["sd2"] / n - mean_d ** 2, 0.0)
    mean_a = tot["sa"] / n
    var_a = max(tot["sa2"] / n - mean_a ** 2, 0.0)
    return {"rounds": n, "ev_a": mean_a, "ev_b": tot["sb"] / n, "diff": mean_d, "diff_se": math.sqrt(var_d / n),
            "ev_a_se": math.sqrt(var_a / n), "agreement_basic": tot["agree"] / max(tot["dec"], 1),
            "rounds_differing": tot["differ"] / n, "a": spec_label(spec_a), "b": spec_label(spec_b),
            "wall_seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------------- reporting
def _pct(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{100 * x:+.{nd}f}%"


def markdown_table(results: list[dict[str, Any]]) -> str:
    sides = ("perfect_pairs", "twenty_one_plus_three", "crazy_7", "bust_bonus")
    short = {"perfect_pairs": "PP", "twenty_one_plus_three": "21+3", "crazy_7": "C7", "bust_bonus": "BB"}
    head = ("| agent | EV/round (units) | ± SE | main edge (on initial bet) | main edge on wagered | std/round (units) "
            "| avg bet (units) | agreement w/ basic | cash-outs /1000 | insured / offered | side bets (n, RTP) |")
    lines = [head, "|" + "---|" * 11]
    for r in results:
        sb = []
        for s in sides:
            x = r["side_bets"][s]
            if x["bets"]:
                sb.append(f"{short[s]} {x['bets']:,} @ {100 * x['rtp']:.1f}%")
        ag = r["agreement_basic"]
        lines.append(
            f"| {r['agent']} | {r['ev_per_round_units']:+.5f} | {r['ev_se_units']:.5f} | {_pct(r['edge'])} "
            f"| {_pct(r['edge_on_wagered_main'])} | {r['std_per_round'] / r['unit']:.3f} | {r['avg_bet_units']:.2f} "
            f"| {'—' if ag is None else f'{100 * ag:.2f}%'} | {r['cashouts_per_1000_rounds']:.2f} "
            f"| {r['insurance_taken']:,} / {r['insurance_offered']:,} | {'; '.join(sb) or '—'} |")
    return "\n".join(lines)


def cashout_table(results: list[dict[str, Any]]) -> str:
    rows = [r for r in results if r["cashout_diagnostic"]["cashouts"]]
    if not rows:
        return ""
    tcs = sorted({int(t) for r in rows for t in r["cashout_diagnostic"]["cashout_rate_by_tc"]})
    lines = ["\n### Cash-out diagnostic\n",
             "EV per unit of the hand's original bet in the states where the agent cashed out: the cash-out EV "
             "(exact) vs the best *hold* EV from the composition solver and vs one realised counterfactual hold "
             "round per cash-out (resampled unseen cards). Cash-out rate = share of play decisions, by Hi-Lo TC.\n",
             "| agent | cash-outs / 1000 rounds | mean cash-out EV | mean hold EV (solver) | mean hold return (MC) "
             "| gain per cash-out vs solver hold | " + " | ".join(f"TC {t:+d}" for t in tcs) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(tcs)]
    for r in rows:
        c = r["cashout_diagnostic"]
        rates = " | ".join(f"{100 * c['cashout_rate_by_tc'].get(str(t), 0):.1f}%" for t in tcs)
        lines.append(f"| {r['agent']} | {r['cashouts_per_1000_rounds']:.1f} | {c['mean_cashout_ev']:+.4f} "
                     f"| {c['mean_hold_ev_solver']:+.4f} | {c['mean_hold_return_mc']:+.4f} "
                     f"| {c['gain_per_cashout_vs_solver_hold']:+.4f} | {rates} |")
    for r in rows:
        c = r["cashout_diagnostic"]
        lines.append(f"\nBy hand class — {r['agent']} (overall cash-out rate {100 * c['cashout_rate_overall']:.1f}% of "
                     f"play decisions, mean Hi-Lo TC when cashing out {c['mean_tc_at_cashout']:+.2f}):\n")
        lines.append("| hand class | decisions /1000 rounds | cash-out rate | mean TC at cash-out | gain per cash-out vs solver hold |")
        lines.append("|---|---|---|---|---|")
        for cl, v in c["by_class"].items():
            tcs = "—" if v["mean_tc_at_cashout"] is None else f"{v['mean_tc_at_cashout']:+.2f}"
            gs = "—" if v["gain_per_cashout_vs_solver_hold"] is None else f"{v['gain_per_cashout_vs_solver_hold']:+.4f}"
            lines.append(f"| {cl} | {v['decisions_per_1000_rounds']:.1f} | {100 * v['cashout_rate']:.1f}% | {tcs} | {gs} |")
    return "\n".join(lines)


def save_metrics(name: str, out: dict[str, Any]) -> None:
    """Merge evaluation results for checkpoint ``name`` into its JSON sidecar."""
    from .agent import resolve_checkpoint
    js = resolve_checkpoint(name).with_suffix(".json")
    meta = json.loads(js.read_text())
    mine = [r for r in out["agents"] if r["agent"].partition(":")[0] == name]
    dup = [d for d in out.get("duplicate", []) if d["a"].partition(":")[0] == name]
    keep = ("agent", "rounds", "ev_per_round_units", "ev_se_units", "edge", "edge_se", "edge_on_wagered_main",
            "std_per_round", "avg_bet_units", "agreement_basic", "cashouts_per_1000_rounds", "cashout_diagnostic",
            "insurance_taken", "insurance_offered", "side_bets", "insurance", "top_disagreements")
    meta["evaluation"] = {"rounds": out["rounds"], "seed": out["seed"],
                          "agents": [{k: r.get(k) for k in keep} for r in mine], "duplicate": dup,
                          "baselines": [{k: r.get(k) for k in keep[:9]} for r in out["agents"]
                                        if r["agent"] in BASELINES]}
    m = meta.setdefault("metrics", {})
    for r in mine:
        tag = r["agent"].partition(":")[2] or "full"
        m[f"ev_per_round_{tag}"] = r["ev_per_round_units"]
        m[f"edge_{tag}"] = r["edge"]
    for d in dup:
        suffix = "_nocash" if d["a"].endswith(":flat_nocash") else ""
        m[f"dup_gain_vs_basic{suffix}"] = d["diff"]
        m[f"dup_gain_se{suffix}"] = d["diff_se"]
    js.write_text(json.dumps(meta, indent=1, default=float))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agents", default="nn_v1,basic,hilo")
    ap.add_argument("--rounds", type=int, default=2_000_000)
    ap.add_argument("--duplicate", type=int, default=0, help="rounds of paired NN-vs-basic play evaluation")
    ap.add_argument("--analysis", action="store_true", help="learned deviations, strategy charts, bet head vs exact")
    ap.add_argument("--dup-solver", action="store_true", help="also run the (slow) solver oracles in the paired eval")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--config", default=None, help="game config JSON (default config/default.json)")
    ap.add_argument("--game-patch", default=None, help='JSON patch of the game config, e.g. \'{"rules": {"cashout": false}}\'')
    ap.add_argument("--json", default=None, help="write all results as JSON")
    ap.add_argument("--markdown", default=None, help="write the report as Markdown")
    ap.add_argument("--save-metrics", action="store_true",
                    help="store the results in checkpoints/<name>.json ('evaluation' + headline 'metrics')")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, patch=json.loads(a.game_patch) if a.game_patch else None)
    workers = a.workers or mp.cpu_count()
    specs = [s.strip() for s in a.agents.split(",") if s.strip()]
    out: dict[str, Any] = {"rounds": a.rounds, "seed": a.seed, "agents": []}
    md: list[str] = []
    for s in specs:
        t0 = time.perf_counter()
        r = evaluate_agent(s, cfg, a.rounds, a.seed, workers)
        out["agents"].append(r)
        print(f"[{s}] EV/round {r['ev_per_round_units']:+.5f} ± {r['ev_se_units']:.5f} units, "
              f"edge {_pct(r['edge'])}, agreement {r['agreement_basic']}, {time.perf_counter() - t0:.0f}s", flush=True)
    md.append(f"### Full game, {a.rounds:,} rounds per agent (seed {a.seed})\n")
    md.append(markdown_table(out["agents"]))
    md.append(cashout_table(out["agents"]))
    nn_specs = sorted({s.partition(":")[0] for s in specs if s not in BASELINES})
    if a.duplicate:
        out["duplicate"] = []
        md.append(f"\n### Paired play comparison vs basic strategy (flat 1 unit, no side bets/insurance), "
                  f"{a.duplicate:,} rounds\n")
        md.append("| agent | EV/round | basic EV/round (same cards) | gain vs basic | ± SE | rounds that differ | agreement |")
        md.append("|---|---|---|---|---|---|---|")
        dup_specs = [x for n in nn_specs for x in (n, f"{n}:flat_nocash")] + ["hilo_nosides"]
        if a.dup_solver:
            dup_specs += ["solver", "solver_nocash"] if cfg.rules.cashout else ["solver_nocash"]
        if not cfg.rules.cashout:
            dup_specs = [x for x in dup_specs if not x.endswith(":flat_nocash")]
        for s in dup_specs:
            d = duplicate_eval(s, "basic", cfg, a.duplicate, a.seed + 1, workers)
            out["duplicate"].append(d)
            print(f"[dup {s}] gain {d['diff']:+.5f} ± {d['diff_se']:.5f} (EV {d['ev_a']:+.5f} vs {d['ev_b']:+.5f})",
                  flush=True)
            label = {True: f"{s.partition(':')[0]} (play only, never cashes out)", False: f"{s} (play only)"}[
                s.endswith(":flat_nocash")]
            md.append(f"| {label} | {d['ev_a']:+.5f} | {d['ev_b']:+.5f} | {d['diff']:+.5f} | {d['diff_se']:.5f} "
                      f"| {100 * d['rounds_differing']:.2f}% | {100 * d['agreement_basic']:.2f}% |")
    if a.analysis and nn_specs:
        from .analysis import analysis_report
        for s in nn_specs:
            rep, data = analysis_report(s, cfg, seed=a.seed, workers=workers)
            out.setdefault("analysis", {})[s] = data
            md.append(rep)
    for r in out["agents"]:
        if r["agent"] not in BASELINES:
            md.append(f"\n#### Most frequent disagreements with basic strategy — {r['agent']}\n")
            md.append("| decision cell: NN action (basic) | per 1000 rounds | mean TC |")
            md.append("|---|---|---|")
            for t in r["top_disagreements"][:15]:
                md.append(f"| {t['cell']} | {t['per_1000_rounds']:.2f} | {t['mean_tc']:+.2f} |")
    report = "\n".join(md)
    print(report)
    if a.save_metrics:
        for s in nn_specs:
            save_metrics(s, out)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    if a.markdown:
        Path(a.markdown).write_text(report + "\n")


if __name__ == "__main__":
    main()


__all__ = ["build_agent", "duplicate_eval", "evaluate_agent", "markdown_table"]
