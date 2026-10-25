"""Running metrics shared by interactive sessions and headless simulations (SPEC §6 ``metrics``).

:class:`LabMetrics` extends :class:`onebj.sim.Metrics` (all the per-round counters) with what the UI
needs on top: a downsampled bankroll curve, agreement with basic strategy and an insurance row in the
side-bet table. It is a plain picklable dataclass so simulation worker processes can ship it back.

Conventions (as in :mod:`onebj.sim`):

* ``edge`` = main-game net / original main stakes (EV per unit of the initial bet, the definition of
  the official 99.59% RTP); ``ev_per_round`` = mean total net per round incl. side bets and insurance.
* ``wins/losses/pushes`` count main hands (a split round counts two hands); a cashed-out hand counts as a win,
  loss or push depending on whether the offer was above, below or equal to its stake (``cashouts`` counts them).
* ``bankroll`` = ``[[round, balance], ...]``, at most ~``MAX_POINTS`` points: a point every ``stride``
  rounds, the stride doubling (and every other point dropped) whenever the curve gets too long.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..engine import RoundResult
from ..sim import Metrics

MAX_POINTS = 1000


@dataclass
class LabMetrics(Metrics):
    start: float = 0.0  # bankroll at round 0
    stride: int = 1
    curve: list[list[float]] = field(default_factory=lambda: [[0, 0.0]])  # [rounds, cumulative net]
    agree: int = 0
    decisions: int = 0

    def add(self, r: RoundResult) -> None:
        super().add(r)
        if self.rounds % self.stride == 0:
            self.curve.append([self.rounds, self.balance])
            if len(self.curve) > MAX_POINTS:
                self.curve = self.curve[::2]  # keeps the points at multiples of 2 * stride
                self.stride *= 2

    def decision(self, agrees: bool) -> None:
        """Record one play decision and whether it matches basic strategy."""
        self.decisions += 1
        if agrees:
            self.agree += 1

    def bankroll_curve(self) -> list[list[float]]:
        pts = [[int(r), round(self.start + n, 2)] for r, n in self.curve]
        if pts[-1][0] != self.rounds:
            pts.append([self.rounds, round(self.start + self.balance, 2)])
        return pts

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        sides = {k: _side_row(v) for k, v in d["side_bets"].items()}
        sides["insurance"] = {"wagered": round(self.insurance_wagered, 2), "net": round(self.insurance_net, 2),
                              "rtp": (1 + self.insurance_net / self.insurance_wagered) if self.insurance_wagered
                              else None}
        d.update(
            net=round(self.net, 2), wagered=round(self.wagered, 2), ev_per_round=_r(d["ev_per_round"]),
            std_per_round=_r(d["std_per_round"]), edge=_r(d["edge"]), main_rtp=_r(d["main_rtp"]),
            edge_se=_r(d["edge_se"]), main_std_per_unit=_r(d["main_std_per_unit"]),
            dealer_bust_rate=_r(d["dealer_bust_rate"]), side_bets=sides, bankroll=self.bankroll_curve(),
            max_drawdown=round(self.max_drawdown, 2),
            agreement_basic=(self.agree / self.decisions) if self.decisions else None, cashouts=self.cashouts,
            rounds_per_sec=round(d["rounds_per_sec"]) if d["rounds_per_sec"] else None,
        )
        d.pop("insurance", None)
        return d


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None else round(x, nd)


def _side_row(v: dict[str, Any]) -> dict[str, Any]:
    return {**v, "wagered": round(v["wagered"], 2), "net": round(v["net"], 2),
            "rtp": _r(v["rtp"]), "rtp_se": _r(v["rtp_se"])}


def merge_parts(parts: Sequence[LabMetrics], start: float) -> LabMetrics:
    """Combine per-worker metrics (each counted from net 0) into one, starting at bankroll ``start``.

    Counters are summed. The bankroll curve interleaves the workers: point k is the sum of every
    worker's k-th sample (each worker's samples are monotone in rounds, so this is a genuine sampled
    trajectory of the combined rounds); the drawdown is measured on that combined curve."""
    out = LabMetrics(start=start)
    if not parts:
        return out
    if len(parts) == 1:
        p = parts[0]
        out.merge(p)
        out.curve = [list(x) for x in p.curve]
        out.stride, out.max_drawdown = p.stride, p.max_drawdown
        out.agree, out.decisions = p.agree, p.decisions
        return out
    for p in parts:
        out.merge(p)
        out.agree += p.agree
        out.decisions += p.decisions
    k = min(len(p.curve) for p in parts)
    curve = [[sum(p.curve[i][0] for p in parts), sum(p.curve[i][1] for p in parts)] for i in range(k)]
    if curve[-1][0] != out.rounds:
        curve.append([out.rounds, out.balance])
    out.curve = curve
    peak = dd = 0.0
    for _, n in curve:
        peak = max(peak, n)
        dd = max(dd, peak - n)
    out.max_drawdown = dd
    return out
