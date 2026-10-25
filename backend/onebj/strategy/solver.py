"""Infinite-deck strategy solver for the ONE Blackjack rules.

Given a card-probability vector ``p`` over rank buckets (A, 2..9, T) — default: a full shoe — it computes
the optimal *total-dependent* strategy and its expected value, exactly for an infinite deck with that
composition. It models every rule that matters for decisions:

* S17/H17, peek on ace and/or ten; when the dealer does **not** peek, a dealer blackjack discovered at
  the end beats every live hand ("all": doubled/split stakes lost too; "original": only the original
  stake is lost) — decisions are optimised over the mixture of the "dealer has BJ" / "no BJ" worlds
  since the player cannot tell them apart;
* Six Card Charlie (state includes the number of cards) incl. "charlie beats dealer BJ";
* double on any two (or hard 9-11 only), DAS, one split (no re-split), split aces get one card.

Used to (1) derive and verify the basic strategy tables (:mod:`onebj.strategy.basic`), (2) compute
Hi-Lo index numbers by shifting the composition along the Hi-Lo direction, (3) compute effects of
removal for the main-game edge estimator (:mod:`onebj.analytics`) and (4) the theoretical RTP.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from ..config import RulesConfig

ACTIONS = ("S", "H", "D", "Ds", "P")


@dataclass(frozen=True)
class SolverRules:
    """The subset of :class:`RulesConfig` that affects strategy/EV."""

    h17: bool = False
    peek_ace: bool = True
    peek_ten: bool = False
    late_all: bool = True
    das: bool = True
    double_any: bool = True
    can_split: bool = True
    split_aces_one_card: bool = True
    charlie: int = 6
    charlie_beats_bj: bool = True
    bj_payout: float = 1.5

    @classmethod
    def from_config(cls, r: RulesConfig) -> "SolverRules":
        return cls(h17=r.dealer_hits_soft_17, peek_ace=r.peek_on_ace, peek_ten=r.peek_on_ten,
                   late_all=r.late_dealer_bj_takes == "all", das=r.double_after_split,
                   double_any=r.double_any_two, can_split=r.max_splits > 0,
                   split_aces_one_card=r.split_aces_one_card, charlie=r.charlie_cards,
                   charlie_beats_bj=r.charlie_beats_dealer_bj, bj_payout=r.blackjack_payout)


def full_shoe_probs() -> tuple[float, ...]:
    """Bucket probabilities of a full shoe (any number of decks): 1/13 each, T = 4/13."""
    return (1 / 13,) * 9 + (4 / 13,)


def dealer_outcomes(up: int, p: Sequence[float], h17: bool = False,
                    exclude_hole: int = -1) -> dict[tuple[int, int], float]:
    """Dealer final-hand distribution for up-card bucket ``up`` (infinite deck with probabilities ``p``).

    Returns ``{(final_total, num_cards): prob}`` with ``final_total`` 17..21 or 22 for bust. With
    ``exclude_hole`` = bucket, the hole card is conditioned not to be that bucket (peeked: no BJ).
    """
    q = list(p)
    if exclude_hole >= 0:
        z = 1.0 - q[exclude_hole]
        q = [0.0 if i == exclude_hole else x / z for i, x in enumerate(q)]
    out: dict[tuple[int, int], float] = {}
    frontier: dict[tuple[int, bool, int], float] = {(up + 1, up == 0, 1): 1.0}
    probs = q
    while frontier:
        nxt: dict[tuple[int, bool, int], float] = {}
        for (h, a, n), pr in frontier.items():
            n2 = n + 1
            for c in range(10):
                pc = probs[c]
                if pc == 0.0:
                    continue
                w = pr * pc
                h2 = h + c + 1
                if h2 > 21:
                    out[(22, n2)] = out.get((22, n2), 0.0) + w
                    continue
                a2 = a or c == 0
                soft = a2 and h2 <= 11
                t = h2 + 10 if soft else h2
                if t > 17 or (t == 17 and not (h17 and soft)):
                    out[(t, n2)] = out.get((t, n2), 0.0) + w
                else:
                    key = (h2, a2, n2)
                    nxt[key] = nxt.get(key, 0.0) + w
        frontier = nxt
        probs = p
    return out


def collapse_totals(outcomes: dict[tuple[int, int], float]) -> list[float]:
    """[P17, P18, P19, P20, P21, Pbust] from :func:`dealer_outcomes`."""
    d = [0.0] * 6
    for (t, _n), pr in outcomes.items():
        d[t - 17] += pr
    return d


class _Values:
    """Hit/stand value tables vs one dealer up card for one 'context' (main hand or split hand)."""

    def __init__(self, solver: "Solver", up: int, loss1: float, loss2: float) -> None:
        self.s = solver
        self.p = solver.p
        w = solver.w[up]
        self.w = w
        stand_nobj = solver.stand_nobj[up]  # set by Solver._prepare before construction
        self.stand = [(1 - w) * stand_nobj[t] - w * loss1 for t in range(22)]
        self.stand2 = [(1 - w) * 2 * stand_nobj[t] - w * loss2 for t in range(22)]  # doubled
        r = solver.rules
        self.charlie = (1 - w) + w * (1.0 if r.charlie_beats_bj else -loss1)
        self.memo: dict[tuple[int, bool, int], float] = {}

    def _nmax(self, n: int) -> int:
        return n if self.s.rules.charlie else 0

    def next_states(self, t: int, soft: bool):
        hard = t - 10 if soft else t
        for c in range(10):
            h2 = hard + c + 1
            if h2 > 21:
                yield c, -1, False
                continue
            a2 = soft or c == 0
            s2 = a2 and h2 <= 11
            yield c, (h2 + 10 if s2 else h2), s2

    def hit(self, t: int, soft: bool, n: int) -> float:
        p = self.p
        ch = self.s.rules.charlie
        ev = 0.0
        for c, t2, s2 in self.next_states(t, soft):
            if t2 < 0:
                ev -= p[c]
            elif ch and n + 1 >= ch:
                ev += p[c] * self.charlie
            else:
                ev += p[c] * self.best(t2, s2, n + 1)
        return ev

    def best(self, t: int, soft: bool, n: int) -> float:
        key = (t, soft, n if self.s.rules.charlie else 0)
        v = self.memo.get(key)
        if v is None:
            # 21 is auto-stood (as in the live game), so it is never hit even with Charlie in reach
            v = self.stand[t] if t == 21 else max(self.stand[t], self.hit(t, soft, n))
            self.memo[key] = v
        return v

    def double(self, t: int, soft: bool) -> float:
        p = self.p
        ch = self.s.rules.charlie
        ev = 0.0
        for c, t2, _s2 in self.next_states(t, soft):
            if t2 < 0:
                ev -= 2 * p[c]
            elif ch and ch <= 3:
                ev += 2 * p[c] * self.charlie
            else:
                ev += p[c] * self.stand2[t2]
        return ev


class Solver:
    """Optimal total-dependent strategy + EV for composition ``p`` under ``rules``."""

    def __init__(self, rules: SolverRules = SolverRules(), p: Sequence[float] | None = None) -> None:
        if p is None:
            p = full_shoe_probs()
        tot = sum(p)
        self.p = [x / tot for x in p]
        self.rules = rules
        p = self.p
        self.q = [0.0] * 10  # P(hole card completes a dealer BJ)
        self.q[0], self.q[9] = p[9], p[0]
        self.peeked = [False] * 10
        self.peeked[0], self.peeked[9] = rules.peek_ace, rules.peek_ten
        self.w = [0.0 if self.peeked[u] else self.q[u] for u in range(10)]  # decision-world BJ weight
        self.dealer_nobj: dict[int, list[float]] = {}
        self.stand_nobj: dict[int, list[float]] = {}
        self._main: dict[int, _Values] = {}
        self._split: dict[int, _Values] = {}

    def _prepare(self, u: int) -> None:
        """Lazily compute the dealer distribution / stand EVs / value tables vs up card ``u``."""
        p, rules = self.p, self.rules
        excl = 9 if u == 0 else (0 if u == 9 else -1)
        d = collapse_totals(dealer_outcomes(u, p, rules.h17, excl))
        self.dealer_nobj[u] = d
        st = []
        for t in range(22):
            if t < 17:
                st.append(d[5] - (1 - d[5]))
            else:
                win = d[5] + sum(d[i] for i in range(t - 17))
                lose = sum(d[i] for i in range(t - 16, 5))
                st.append(win - lose)
        self.stand_nobj[u] = st
        late_all = rules.late_all
        self._main[u] = _Values(self, u, 1.0, 2.0 if late_all else 1.0)
        self._split[u] = _Values(self, u, 1.0 if late_all else 0.5, 2.0 if late_all else 0.5)

    def values(self, u: int, split_hand: bool = False) -> _Values:
        if u not in self._main:
            self._prepare(u)
        return self._split[u] if split_hand else self._main[u]

    # -------------------------------------------------------------- decision EVs
    def can_double(self, t: int, soft: bool) -> bool:
        return self.rules.double_any or (not soft and 9 <= t <= 11)

    def split_ev(self, x: int, u: int) -> float:
        """EV (per original unit) of splitting a pair of bucket ``x`` vs up ``u`` (2 hands, no re-split)."""
        V = self.values(u, True)
        p = self.p
        r = self.rules
        ev = 0.0
        for c in range(10):
            hard = x + c + 2
            ace = x == 0 or c == 0
            soft = ace and hard <= 11
            t = hard + 10 if soft else hard
            if (x == 0 and r.split_aces_one_card) or t == 21:
                v = V.stand[t]
            else:
                v = max(V.stand[t], V.hit(t, soft, 2))
                if r.das and self.can_double(t, soft):
                    v = max(v, V.double(t, soft))
            ev += p[c] * v
        return 2 * ev

    def evs(self, t: int, soft: bool, n: int, u: int, pair: int = -1, can_double: bool = True,
            can_split: bool = True, split_hand: bool = False) -> dict[str, float]:
        """EV of each legal action for a hand (total ``t``, ``n`` cards) vs up ``u``."""
        V = self.values(u, split_hand)
        if t == 21:
            return {"S": V.stand[t]}
        out = {"S": V.stand[t], "H": V.hit(t, soft, n)}
        if n == 2 and can_double and self.can_double(t, soft):
            out["D"] = V.double(t, soft)
        if pair >= 0 and can_split and self.rules.can_split:
            out["P"] = self.split_ev(pair, u)
        return out

    def best_code(self, t: int, soft: bool, n: int, u: int, pair: int = -1) -> str:
        e = self.evs(t, soft, n, u, pair)
        best = max(e, key=e.__getitem__)
        if best == "D":
            return "D" if e["H"] >= e["S"] else "Ds"
        return best

    # -------------------------------------------------------------- whole game
    def initial_ev(self, c1: int, c2: int, u: int) -> float:
        """EV of a round given player buckets c1, c2 and up u (never insuring)."""
        q = self.q[u]
        if {c1, c2} == {0, 9}:
            return (1 - q) * self.rules.bj_payout
        hard = c1 + c2 + 2
        ace = c1 == 0 or c2 == 0
        soft = ace and hard <= 11
        t = hard + 10 if soft else hard
        best = max(self.evs(t, soft, 2, u, c1 if c1 == c2 else -1).values())
        if self.peeked[u]:
            return -q + (1 - q) * best
        return best

    def round_ev(self) -> float:
        """Expected net per round per unit of original bet under optimal strategy."""
        p = self.p
        ev = 0.0
        for u in range(10):
            s = 0.0
            for c1 in range(10):
                for c2 in range(c1, 10):
                    w = p[c1] * p[c2] * (1 if c1 == c2 else 2)
                    s += w * self.initial_ev(c1, c2, u)
            ev += p[u] * s
        return ev

    def strategy_tables(self, max_cards: int = 5) -> dict:
        """Optimal action codes: ``{"hard": {n: {t: [10 codes]}}, "soft": ..., "pairs": {x: [10]}}``."""
        hard: dict[int, dict[int, list[str]]] = {}
        soft: dict[int, dict[int, list[str]]] = {}
        for n in range(2, max_cards + 1):
            hard[n] = {t: [self.best_code(t, False, n, u) for u in range(10)] for t in range(4, 22)}
            soft[n] = {t: [self.best_code(t, True, n, u) for u in range(10)] for t in range(12, 22)}
        pairs = {}
        for x in range(10):
            hard_t = 2 * x + 2
            t, s = (12, True) if x == 0 else (hard_t, False)
            pairs[x] = ["P" if self.best_code(t, s, 2, u, x) == "P" else "-" for u in range(10)]
        return {"hard": hard, "soft": soft, "pairs": pairs}


@lru_cache(maxsize=64)
def solve(rules: SolverRules = SolverRules(), p: tuple[float, ...] | None = None) -> Solver:
    """Cached :class:`Solver` for ``rules`` and composition ``p`` (tuple, default full shoe)."""
    return Solver(rules, p)


def hilo_shift_probs(tc: float) -> tuple[float, ...]:
    """Composition of a 52-card deck at Hi-Lo true count ``tc``: low (2-6) and high (T, A) cards
    shifted by tc/2 cards each, spread proportionally over their ranks."""
    counts = [4.0] * 9 + [16.0]
    for b in (1, 2, 3, 4, 5):
        counts[b] -= tc / 2 / 5
    counts[0] += tc / 2 * 4 / 20
    counts[9] += tc / 2 * 16 / 20
    return tuple(c / 52 for c in counts)


def hilo_index(rules: SolverRules, total: int, soft: bool, up: int, action: str, versus: str,
               pair: int = -1, lo: float = -12.0, hi: float = 12.0, step: float = 0.25) -> float | None:
    """Hi-Lo true count at which ``action`` becomes better than ``versus`` (codes "S", "H", "D", "P")
    for a 2-card hand vs ``up``, scanning upward from ``lo``; None if they never cross in range.
    The composition at each true count is :func:`hilo_shift_probs` (infinite deck)."""
    prev = None
    tc = lo
    while tc <= hi + 1e-9:
        s = Solver(rules, hilo_shift_probs(tc))
        e = s.evs(total, soft, 2, up, pair)
        d = e[action] - e[versus]
        if prev is not None and (prev < 0) != (d < 0):
            return tc - step * d / (d - prev)  # linear interpolation of the crossing
        prev = d
        tc += step
    return None
