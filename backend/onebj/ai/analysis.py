"""Inspection tools for a trained agent: solver regret, strategy charts, learned index plays, bet head vs
exact analytics. Used by :mod:`onebj.ai.evaluate` (``--analysis``), the training log and the notebook.

* :func:`benchmark_states` + :func:`solver_regret` — play quality on natural decision states, measured
  against the composition-dependent solver (:mod:`onebj.strategy.solver`, infinite deck at the current
  unseen composition — near-exact for 8 decks): regret = best solver EV − solver EV of the chosen action
  (cash-out EV is exact). Basic strategy has a small positive regret (it ignores the count); a perfect
  composition-dependent player has ~0.
* :func:`strategy_chart` — actions of the network by total × up card for a synthetic shoe at a given
  Hi-Lo true count.
* :func:`index_scan` / :func:`deviation_table` — true count at which the network switches between two
  actions (compare with the Hi-Lo Illustrious-18 indices) and the insurance index.
* :func:`bet_validation` — bet-head predictions vs exact composition EVs (side bets, insurance) and the
  EoR main-edge estimate.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..analytics import insurance_ev, main_edge_estimate, side_bet_evs
from ..cards import BUCKETS
from ..config import GameConfig, from_dict, load_config, to_dict
from ..engine import Action, Bets, Phase, Table
from ..strategy.basic import ILLUSTRIOUS_18, get_basic_strategy
from ..strategy.cashout import get_cashout_table
from ..strategy.solver import Solver, SolverRules
from .features import (
    KIND_BET,
    KIND_INSURANCE,
    KIND_PLAY,
    FeatureEncoder,
    HandState,
    cfg_half_size,
    hilo_composition,
    synthetic_shoe,
)
from .model import BET_LABELS, PLAY_LABELS, NumpyNet

_CODES = ("S", "H", "D", "P")


# ---------------------------------------------------------------------- solver regret
def solver_action_evs(rules: SolverRules, unseen_buckets: Sequence[int], state: tuple,
                      cash_value: float | None) -> list[float]:
    """EV per unit original bet of [stand, hit, double, split, cashout] (NaN = illegal) for the composition."""
    total, soft, n, up, can_double, pair, can_split, is_split = state
    tot = sum(unseen_buckets)
    s = Solver(rules, tuple(x / tot for x in unseen_buckets))
    e = s.evs(total, soft, n, up, pair if can_split else -1, can_double, can_split, is_split)
    out = [e.get(c, math.nan) for c in _CODES]
    out.append(math.nan if cash_value is None else cash_value)
    return out


def _bench_worker(args: tuple) -> dict[str, np.ndarray]:
    cfg_d, n, seed, tilt = args
    cfg = from_dict(cfg_d)
    from .datagen import tilt_shoe
    rng = random.Random(seed)
    t = Table(cfg, seed=seed, record_events=False)
    enc = FeatureEncoder()
    rules = SolverRules.from_config(cfg.rules)
    X, E, S, B = [], [], [], []
    last = -1
    while len(X) < n:
        if tilt and t.shoe.shoe_number != last:
            last = t.shoe.shoe_number
            if rng.random() < 0.5:
                tilt_shoe(t, rng, 0.08)
        t.place_bets(Bets(main=1))
        while True:
            ph = t.phase
            if ph is Phase.PLAYER:
                st = t.decision_state()
                h = t.hands[t.active]
                off = t.cashout_offer()
                cv = off / h.bet - 1 if off is not None else None
                X.append(enc.encode_table(t, KIND_PLAY))
                E.append(solver_action_evs(rules, t.shoe.unseen_buckets(), st, cv))
                S.append(st)
                total, soft, nc, up, cd, pair, cs, _ = st
                B.append(PLAY_LABELS.index(t.strategy.action(total, soft, nc, up, cd, pair, cs)))
                # behaviour: basic strategy with 10% random actions (reach multi-card states)
                if rng.random() < 0.1:
                    opts = [Action.STAND, Action.HIT] + ([Action.DOUBLE] if cd else []) + ([Action.SPLIT] if cs else [])
                    t.act(opts[rng.randrange(len(opts))])
                else:
                    t.act(PLAY_LABELS[B[-1]])
            elif ph is Phase.INSURANCE:
                t.act(Action.NO_INSURANCE)
            else:
                break
    return {"X": np.array(X[:n], np.float32), "E": np.array(E[:n], np.float64),
            "S": np.array(S[:n], np.int16), "basic": np.array(B[:n], np.int8)}


def benchmark_states(cfg: GameConfig | None = None, n: int = 40_000, seed: int = 99, tilt: bool = False,
                     workers: int | None = None) -> dict[str, np.ndarray]:
    """Natural play decision states (basic strategy + 10% exploration) with solver EVs per action."""
    cfg = cfg or load_config()
    workers = workers or mp.cpu_count()
    k = max(1, min(workers, n // 500))
    per = [n // k + (1 if i < n % k else 0) for i in range(k)]
    jobs = [(to_dict(cfg), per[i], seed * 7 + i, tilt) for i in range(k)]
    if k == 1:
        parts = [_bench_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(k) as pool:
            parts = pool.map(_bench_worker, jobs)
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def solver_regret(choices: np.ndarray, bench: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean regret (EV per decision, solver units) of action indices ``choices`` on ``bench`` states."""
    E = bench["E"]
    legal = ~np.isnan(E)
    best = np.where(legal, E, -np.inf).max(axis=1)
    got = E[np.arange(len(E)), choices]
    reg = best - got
    opt = np.where(legal, E, -np.inf).argmax(axis=1)
    return {"regret": float(np.nanmean(reg)), "agree_solver": float((choices == opt).mean()),
            "cashout_rate": float((choices == 4).mean()), "n": len(E)}


def net_choices(net: NumpyNet, bench: dict[str, np.ndarray]) -> np.ndarray:
    q = net.q(bench["X"])
    legal = ~np.isnan(bench["E"])
    return np.where(legal, q, -np.inf).argmax(axis=1)


def regret_report(net: NumpyNet | None, bench: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {"basic": solver_regret(bench["basic"].astype(int), bench)}
    if net is not None:
        out["net"] = solver_regret(net_choices(net, bench), bench)
        q = net.q(bench["X"])
        E = bench["E"]
        legal = ~np.isnan(E[:, :4])
        err = (q[:, :4] - E[:, :4])[legal]
        out["q_rmse_vs_solver"] = float(np.sqrt(np.mean(err ** 2)))
        out["q_bias_vs_solver"] = float(np.mean(err))
    return out


# ---------------------------------------------------------------------- synthetic states
def _play_x(enc: FeatureEncoder, cfg: GameConfig, total: int, soft: bool, up: int, tc: float, decks: float,
            pair: int = -1, ncards: int = 2, can_double: bool = True, is_split: bool = False) -> np.ndarray:
    ct = get_cashout_table(cfg.rules)
    can_split = pair >= 0
    cv = ct.offer(total, soft, ncards, up, can_double and ncards == 2, pair if can_split else -1, is_split) - 1.0
    hs = HandState(total, soft, ncards, pair, can_double and ncards == 2, can_split, is_split, cv)
    shoe = synthetic_shoe(hilo_composition(tc, decks, cfg.shoe.num_decks), cfg.shoe.num_decks,
                          half_size=cfg_half_size(cfg), cards_to_cut=max(0, int(decks * 52 - 104)))
    return enc.encode(KIND_PLAY, hs, up, shoe)


def strategy_chart(agent: Any, tc: float = 0.0, decks: float = 3.0, ncards: int = 2,
                   cashout: bool = True) -> dict[str, dict[str, list[str]]]:
    """Network action per (hard/soft total or pair) × up card (columns 2..9, T, A) at true count ``tc``."""
    enc, net, cfg = agent.enc, agent.net, agent.cfg
    ups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    out: dict[str, dict[str, list[str]]] = {"hard": {}, "soft": {}, "pairs": {}}

    def act(total: int, soft: bool, up: int, pair: int = -1) -> str:
        x = _play_x(enc, cfg, total, soft, up, tc, decks, pair, ncards)
        q = net.q(x)
        legal = [True, True, ncards == 2, pair >= 0, cashout]
        best = max((i for i in range(5) if legal[i]), key=lambda i: q[i])
        return PLAY_LABELS[best]

    # 21 is never a decision (auto-stand / blackjack), so it is not charted
    for t in range(5 if ncards == 2 else 8, 21):
        out["hard"][str(t)] = [act(t, False, u) for u in ups]
    for t in range(13, 21):
        out["soft"][str(t)] = [act(t, True, u) for u in ups]
    if ncards == 2:
        for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]:
            total, soft = (12, True) if p == 0 else (2 * p + 2, False)
            out["pairs"][BUCKETS[p] * 2] = [act(total, soft, u, p) for u in ups]
    return out


_ABBR = {"stand": "S", "hit": "H", "double": "D", "split": "P", "cashout": "C"}


def chart_text(chart: dict[str, dict[str, list[str]]], basic: dict[str, dict[str, list[str]]] | None = None) -> str:
    """Text chart; cells that differ from ``basic`` are marked with ``*``."""
    lines = ["       " + " ".join(f"{c:>3}" for c in "23456789TA")]
    for sec, tag in (("hard", "H"), ("soft", "S"), ("pairs", "P")):
        for row, cells in chart[sec].items():
            ref = basic[sec][row] if basic else None
            s = []
            for i, c in enumerate(cells):
                mark = "*" if ref and ref[i] != c else " "
                s.append(f"{_ABBR[c]:>2}{mark}")
            lines.append(f"{tag}{row:<5} " + " ".join(s))
    return "\n".join(lines)


def basic_chart(cfg: GameConfig, ncards: int = 2) -> dict[str, dict[str, list[str]]]:
    bs = get_basic_strategy(cfg.rules, cfg.shoe.num_decks)
    ups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    out: dict[str, dict[str, list[str]]] = {"hard": {}, "soft": {}, "pairs": {}}
    for t in range(5 if ncards == 2 else 8, 21):
        out["hard"][str(t)] = [bs.action(t, False, ncards, u, ncards == 2) for u in ups]
    for t in range(13, 21):
        out["soft"][str(t)] = [bs.action(t, True, ncards, u, ncards == 2) for u in ups]
    if ncards == 2:
        for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]:
            total, soft = (12, True) if p == 0 else (2 * p + 2, False)
            out["pairs"][BUCKETS[p] * 2] = [bs.action(total, soft, 2, u, True, p, True) for u in ups]
    return out


def index_scan(agent: Any, total: int, soft: bool, up: int, action: str, versus: str, pair: int = -1,
               tcs: Sequence[float] = tuple(np.arange(-8, 10.01, 0.25)), decks: float = 3.0) -> dict[str, Any]:
    """Q(action) − Q(versus) along the Hi-Lo true count; returns the crossing TC (None if no crossing)."""
    ia, ib = PLAY_LABELS.index(action), PLAY_LABELS.index(versus)
    diffs = []
    for tc in tcs:
        q = agent.net.q(_play_x(agent.enc, agent.cfg, total, soft, up, float(tc), decks, pair))
        diffs.append(float(q[ia] - q[ib]))
    cross = None
    for k in range(1, len(diffs)):
        if (diffs[k - 1] < 0) != (diffs[k] < 0):
            t0, t1 = tcs[k - 1], tcs[k]
            cross = float(t0 + (t1 - t0) * (-diffs[k - 1]) / (diffs[k] - diffs[k - 1]))
            break
    status = "crosses" if cross is not None else ("always" if diffs[0] > 0 else "never")
    return {"cross_tc": cross, "status": status, "tcs": [float(t) for t in tcs], "diffs": diffs}


def insurance_scan(agent: Any, tcs: Sequence[float] = tuple(np.arange(-6, 10.01, 0.25)), decks: float = 3.0,
                   total: int = 16, soft: bool = False) -> dict[str, Any]:
    """Predicted insurance EV along the true count (insurance-phase features, dealer ace)."""
    enc, cfg = agent.enc, agent.cfg
    evs, exact = [], []
    for tc in tcs:
        comp = hilo_composition(float(tc), decks, cfg.shoe.num_decks)
        shoe = synthetic_shoe(comp, cfg.shoe.num_decks, half_size=cfg_half_size(cfg))
        x = enc.encode(KIND_INSURANCE, HandState(total, soft, 2), 0, shoe)
        evs.append(float(agent.net.bet(x)[BET_LABELS.index("insurance")]))
        exact.append(insurance_ev(comp, cfg.rules.insurance_payout))
    cross = None
    for k in range(1, len(evs)):
        if evs[k - 1] < 0 <= evs[k]:
            cross = float(tcs[k - 1] + (tcs[k] - tcs[k - 1]) * (-evs[k - 1]) / (evs[k] - evs[k - 1]))
            break
    return {"cross_tc": cross, "tcs": [float(t) for t in tcs], "predicted": evs, "exact": exact}


_ACTION_NAME = {"stand": "stand", "hit": "hit", "double": "double", "split": "split"}


def deviation_table(agent: Any, decks: float = 3.0) -> list[dict[str, Any]]:
    """For each Illustrious-18 entry: the Hi-Lo index vs the network's crossing true count."""
    bs = agent.bs
    rows = []
    for d in ILLUSTRIOUS_18:
        if d.kind == "insurance":
            sc = insurance_scan(agent, decks=decks)
            rows.append({"name": d.name, "hilo_index": d.index, "nn_index": sc["cross_tc"],
                         "status": "crosses" if sc["cross_tc"] is not None else "never"})
            continue
        if d.kind == "pair":
            total, soft, pair = 2 * d.value + 2, False, d.value
        else:
            total, soft, pair = d.value, d.kind == "soft", -1
        basic = bs.action(total, soft, 2, d.up, True, pair, pair >= 0)
        a, b = (d.action, basic) if not d.below else (basic, d.action)
        sc = index_scan(agent, total, soft, d.up, a, b, pair, decks=decks)
        rows.append({"name": d.name, "hilo_index": d.index, "nn_index": sc["cross_tc"], "status": sc["status"],
                     "rule": f"{a} over {b} when TC >= index"})
    return rows


# ---------------------------------------------------------------------- bet head vs exact
def _betval_worker(args: tuple) -> dict[str, np.ndarray]:
    cfg_d, weights, spec_d, n, seed, tilt = args
    cfg = from_dict(cfg_d)
    from .datagen import tilt_shoe
    from .features import FeatureSpec
    enc = FeatureEncoder(FeatureSpec.from_dict(spec_d))
    net = NumpyNet(weights)
    t = Table(cfg, seed=seed, record_events=False)
    rng = random.Random(seed)
    P, X, TC = [], [], []
    last = -1
    while len(P) < n:
        if tilt and t.shoe.shoe_number != last:
            last = t.shoe.shoe_number
            if rng.random() < 0.5:
                tilt_shoe(t, rng, 0.08)
        x = enc.encode_table(t, KIND_BET)
        P.append(net.bet(x))
        cards = t.shoe.unseen_cards()
        b = t.shoe.unseen_buckets()
        ev = side_bet_evs(cards, cfg)
        X.append([main_edge_estimate(b, cfg.rules, cfg.shoe.num_decks), ev["perfect_pairs"],
                  ev["twenty_one_plus_three"], ev["crazy_7"], ev["bust_bonus"],
                  insurance_ev(b, cfg.rules.insurance_payout)])
        TC.append(x[enc.index["hilo_tc"]] * enc.spec.tc_scale)
        t.place_bets(Bets(main=1))
        while t.phase is not Phase.SETTLED:
            if t.phase is Phase.INSURANCE:
                t.act(Action.NO_INSURANCE)
            else:
                st = t.decision_state()
                t.act(t.strategy.action(*st[:4], st[4], st[5], st[6]))
    return {"pred": np.array(P), "exact": np.array(X), "tc": np.array(TC)}


def bet_validation(agent: Any, n: int = 20_000, seed: int = 5, tilt: bool = False,
                   workers: int | None = None) -> dict[str, Any]:
    """Bet-head predictions vs exact analytics on pre-deal states of natural (or tilted) shoes.
    Column 0 compares with the EoR *estimate* of the basic-strategy edge (not exact)."""
    cfg = agent.cfg
    workers = workers or mp.cpu_count()
    k = max(1, min(workers, n // 500))
    per = [n // k + (1 if i < n % k else 0) for i in range(k)]
    w = {"trunk_w": agent.net.Ws, "trunk_b": agent.net.bs, "play_w": agent.net.Wp, "play_b": agent.net.bp,
         "bet_w": agent.net.Wb, "bet_b": agent.net.bb, "bet_offset": agent.net.off, "bet_scale": agent.net.scale,
         "cash_index": agent.net.cash_index}
    jobs = [(to_dict(cfg), w, agent.enc.spec.to_dict(), per[i], seed * 13 + i, tilt) for i in range(k)]
    if k == 1:
        parts = [_betval_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(k) as pool:
            parts = pool.map(_betval_worker, jobs)
    pred = np.concatenate([p["pred"] for p in parts])
    exact = np.concatenate([p["exact"] for p in parts])
    tc = np.concatenate([p["tc"] for p in parts])
    stats = {}
    for i, lab in enumerate(BET_LABELS):
        p, e = pred[:, i], exact[:, i]
        c = float(np.corrcoef(p, e)[0, 1]) if p.std() > 0 and e.std() > 0 else float("nan")
        stats[lab] = {"mean_pred": float(p.mean()), "mean_exact": float(e.mean()), "corr": c,
                      "mae": float(np.abs(p - e).mean()), "share_pos_pred": float((p > 0).mean()),
                      "share_pos_exact": float((e > 0).mean())}
    return {"pred": pred, "exact": exact, "tc": tc, "stats": stats}


# ---------------------------------------------------------------------- report for evaluate --analysis
def analysis_report(spec: str, cfg: GameConfig, seed: int = 0, workers: int | None = None) -> tuple[str, dict]:
    from .agent import NNAgent
    agent = NNAgent.load(spec, cfg, trace=False)
    md = [f"\n### Learned strategy — {spec}\n"]
    bench = benchmark_states(cfg, 60_000, seed + 3, workers=workers)
    reg = regret_report(agent.net, bench)
    md.append("Solver regret on 60,000 natural decision states (basic strategy + 10% exploration; regret = EV "
              "lost per decision vs the composition-dependent solver, cash-out exact):\n")
    md.append("| policy | regret per decision | agreement with solver-optimal | cash-out rate |")
    md.append("|---|---|---|---|")
    for k in ("basic", "net"):
        r = reg[k]
        md.append(f"| {k} | {r['regret']:.5f} | {100 * r['agree_solver']:.2f}% | {100 * r['cashout_rate']:.2f}% |")
    md.append(f"\nQ-value RMSE vs solver EVs: {reg['q_rmse_vs_solver']:.4f} (bias {reg['q_bias_vs_solver']:+.4f}).\n")
    dev = deviation_table(agent)
    md.append("\nIndex plays (3 decks unseen, Hi-Lo-shifted composition): true count where the network switches\n")
    md.append("| play | Hi-Lo index | NN index |")
    md.append("|---|---|---|")
    for r in dev:
        nn = f"{r['nn_index']:+.1f}" if r["nn_index"] is not None else r["status"]
        md.append(f"| {r['name']} | {r['hilo_index']:+.0f} | {nn} |")
    charts = {}
    basic = basic_chart(cfg)
    for tc in (-3.0, 0.0, 3.0):
        ch = strategy_chart(agent, tc)
        charts[str(tc)] = ch
        md.append(f"\nStrategy chart at TC {tc:+.0f} (3 decks unseen; `*` = differs from basic strategy, C = cash out)\n")
        md.append("```\n" + chart_text(ch, basic) + "\n```")
    bv = bet_validation(agent, 20_000, seed + 5, workers=workers)
    md.append("\nBet head vs exact composition EVs (20,000 pre-deal states, natural shoes; `main` vs the EoR "
              "estimate for basic strategy)\n")
    md.append("| output | mean predicted | mean exact | corr | MAE | P(pred > 0) | P(exact > 0) |")
    md.append("|---|---|---|---|---|---|---|")
    for lab, s in bv["stats"].items():
        md.append(f"| {lab} | {s['mean_pred']:+.4f} | {s['mean_exact']:+.4f} | {s['corr']:.3f} | {s['mae']:.4f} "
                  f"| {100 * s['share_pos_pred']:.2f}% | {100 * s['share_pos_exact']:.2f}% |")
    data = {"regret": reg, "deviations": dev, "charts": charts, "bet_validation": bv["stats"]}
    return "\n".join(md), data


__all__ = [
    "analysis_report",
    "basic_chart",
    "benchmark_states",
    "bet_validation",
    "chart_text",
    "deviation_table",
    "index_scan",
    "insurance_scan",
    "regret_report",
    "solver_action_evs",
    "solver_regret",
    "strategy_chart",
]
