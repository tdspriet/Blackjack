"""How often is a side bet +EV, and what is it worth? Exact composition analytics over simulated shoes.

    uv run python -m onebj.ai.sidebet_study --shoes 4000            # two-halves shoe (default) + single shoe

For every round of ``--shoes`` simulated shoes (basic strategy, crowd on) the exact EV of each side bet
for the next round is computed from the unseen composition (:func:`onebj.analytics.side_bet_evs`; Crazy 7
and Bust Bonus as documented there), plus the insurance EV *if* an ace shows. Reported per bet:

* share of rounds with EV > 0 (and > 2 %, > 5 %, > 10 %), mean EV when positive, share of shoes with at
  least one +EV round, +EV rounds per shoe;
* the value of betting the table minimum (€1) exactly when EV > 0: expected € per 100 rounds;
* the Kelly stake for a €1000 bankroll at the mean positive EV (variance from the paytable);
* for Crazy 7: EV as a function of the number of 7s left at several shoe depths.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..analytics import crazy_7_ev, insurance_ev, side_bet_evs
from ..config import from_dict, load_config, to_dict
from ..engine import Action, Bets, Phase, Table
from ..sidebets import SIDE_BETS

BETS = (*SIDE_BETS, "insurance")


def _worker(args: tuple) -> dict[str, Any]:
    cfg_d, shoes, seed = args
    cfg = from_dict(cfg_d)
    t = Table(cfg, seed=seed, record_events=False)
    rows: list[list[float]] = []
    shoe_id: list[int] = []
    start = t.shoe.shoe_number
    while t.shoe.shoe_number - start < shoes:
        cards = t.shoe.unseen_cards()
        b = t.shoe.unseen_buckets()
        ev = side_bet_evs(cards, cfg)
        ins = insurance_ev(b, cfg.rules.insurance_payout)
        rows.append([ev["perfect_pairs"], ev["twenty_one_plus_three"], ev["crazy_7"], ev["bust_bonus"], ins,
                     t.shoe.half, t.shoe.cards_unseen, b[6]])
        shoe_id.append(t.shoe.shoe_number - start)
        t.place_bets(Bets(main=1))
        while t.phase is not Phase.SETTLED:
            if t.phase is Phase.INSURANCE:
                t.act(Action.NO_INSURANCE)
            else:
                st = t.decision_state()
                t.act(t.strategy.action(st[0], st[1], st[2], st[3], st[4], st[5], st[6]))
    return {"rows": np.array(rows), "shoe": np.array(shoe_id) + seed * 10_000_000}


def _side_variance(cfg) -> dict[str, float]:
    from .agent import _side_variances
    v = _side_variances(cfg)
    v["insurance"] = 2.0 ** 2 * (4 / 13) + (9 / 13) - (3 * 4 / 13 - 1) ** 2
    return v


def study(cfg, shoes: int, seed: int = 1, workers: int | None = None) -> dict[str, Any]:
    workers = workers or mp.cpu_count()
    per = [shoes // workers + (1 if i < shoes % workers else 0) for i in range(workers)]
    jobs = [(to_dict(cfg), per[i], seed * 101 + i) for i in range(workers) if per[i]]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = pool.map(_worker, jobs)
    R = np.concatenate([p["rows"] for p in parts])
    S = np.concatenate([p["shoe"] for p in parts])
    n_rounds, n_shoes = len(R), len(np.unique(S))
    var = _side_variance(cfg)
    out: dict[str, Any] = {"rounds": n_rounds, "shoes": n_shoes, "rounds_per_shoe": n_rounds / n_shoes, "bets": {}}
    for k, name in enumerate(BETS):
        ev = R[:, k]
        pos = ev > 0
        shoes_pos = len(np.unique(S[pos]))
        mean_pos = float(ev[pos].mean()) if pos.any() else None
        # insurance is only offered when an ace shows (1/13 of rounds)
        offer = 1 / 13 if name == "insurance" else 1.0
        d = {
            "mean_ev": float(ev.mean()), "p05": float(np.percentile(ev, 5)), "p95": float(np.percentile(ev, 95)),
            "max_ev": float(ev.max()), "share_pos": float(pos.mean()),
            "share_gt_2pct": float((ev > 0.02).mean()), "share_gt_5pct": float((ev > 0.05).mean()),
            "share_gt_10pct": float((ev > 0.10).mean()), "mean_ev_when_pos": mean_pos,
            "share_shoes_with_pos": shoes_pos / n_shoes, "pos_rounds_per_shoe": float(pos.sum() / n_shoes),
            "eur_per_100_rounds_min_stake": float(100 * np.clip(ev, 0, None).mean() * offer),
            "share_pos_half1": float(pos[R[:, 5] == 1].mean()) if (R[:, 5] == 1).any() else None,
            "share_pos_half2": float(pos[R[:, 5] == 2].mean()) if (R[:, 5] == 2).any() else None,
            "variance_per_unit": var[name],
            "kelly_stake_eur_bankroll_1000": (1000 * mean_pos / var[name]) if mean_pos else None,
        }
        out["bets"][name] = d
    # Crazy 7 vs sevens left
    sb = cfg.side_bets.crazy_7
    nd = cfg.shoe.num_decks
    table = {}
    for n_unseen in (312, 208, 104):
        exp7 = n_unseen / 13
        row = {}
        for k7 in range(int(exp7 * 0.5), int(exp7 * 1.6) + 1, max(1, int(exp7 / 8))):
            other = n_unseen - k7
            cards = [0] * 52
            per_card = other / 48
            for c in range(52):
                cards[c] = per_card
            for s in range(4):
                cards[24 + s] = k7 / 4
            row[str(k7)] = crazy_7_ev(cards, sb, cfg.rules.peek_on_ace)
        table[str(n_unseen)] = {"expected_sevens": exp7, "ev_by_sevens": row}
    out["crazy_7_by_sevens"] = table
    sev = R[:, 7] / R[:, 6] * 13 - 1
    out["sevens_relative_richness_p5_p95"] = [float(np.percentile(sev, 5)), float(np.percentile(sev, 95))]
    out["num_decks"] = nd
    return out


def markdown(res: dict[str, dict[str, Any]]) -> str:
    lines = []
    for label, r in res.items():
        lines.append(f"\n#### {label}: {r['shoes']:,} shoes, {r['rounds']:,} rounds ({r['rounds_per_shoe']:.1f} per shoe)\n")
        lines.append("| bet | mean EV | 5th–95th pct | P(EV>0) | P(EV>2%) | P(EV>5%) | mean EV when >0 | shoes with a +EV round "
                     "| +EV rounds/shoe | € per 100 rounds (€1 when +EV) | Kelly stake @ €1000 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, d in r["bets"].items():
            mp_ = d["mean_ev_when_pos"]
            ks = d["kelly_stake_eur_bankroll_1000"]
            lines.append(
                f"| {name} | {100 * d['mean_ev']:+.2f}% | {100 * d['p05']:+.1f}% … {100 * d['p95']:+.1f}% "
                f"| {100 * d['share_pos']:.2f}% | {100 * d['share_gt_2pct']:.2f}% | {100 * d['share_gt_5pct']:.2f}% "
                f"| {'—' if mp_ is None else f'{100 * mp_:+.2f}%'} | {100 * d['share_shoes_with_pos']:.1f}% "
                f"| {d['pos_rounds_per_shoe']:.2f} | €{d['eur_per_100_rounds_min_stake']:.2f} "
                f"| {'—' if ks is None else f'€{ks:.2f}'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Side-bet +EV frequency study (exact composition analytics)")
    ap.add_argument("--shoes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None)
    a = ap.parse_args(argv)
    res = {}
    for label, patch in (("two halves (Napoleon, default)", {}),
                         ("single 8-deck shoe, 75% penetration", {"shoe": {"split_halves": False}})):
        t0 = time.perf_counter()
        cfg = load_config(patch=patch)
        res[label] = study(cfg, a.shoes, a.seed, a.workers or None)
        print(f"{label}: {time.perf_counter() - t0:.0f}s", flush=True)
    md = markdown(res)
    c7 = res["two halves (Napoleon, default)"]["crazy_7_by_sevens"]
    md += "\n\nCrazy 7 EV by number of 7s left (other ranks at their expected share):\n\n"
    for n, row in c7.items():
        md += f"* {n} unseen cards (expected {row['expected_sevens']:.0f} sevens): " + ", ".join(
            f"{k}: {100 * v:+.1f}%" for k, v in row["ev_by_sevens"].items()) + "\n"
    print(md)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    if a.markdown:
        Path(a.markdown).write_text(md + "\n")


if __name__ == "__main__":
    main()


__all__ = ["markdown", "study"]
