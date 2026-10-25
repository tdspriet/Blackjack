"""Basic strategy for the configured rules, plus Hi-Lo index plays (Illustrious 18 adapted).

The tables are *derived* (not typed in) with :mod:`onebj.strategy.solver`:

* 2-card decisions (incl. doubles and splits) are composition-weighted for the actual number of decks:
  for every player 2-card composition and up card the EVs are computed with those three cards removed
  from a full ``num_decks`` shoe, then averaged per (total, soft, up) — i.e. the classic multi-deck
  total-dependent basic strategy.
* 3-, 4- and 5-card hit/stand decisions come from the infinite-deck solver with the card count in the
  state, so Six Card Charlie is handled exactly (e.g. hit hard 16 with 5 cards vs 7, hit any 5-card soft).

For the default ONE Blackjack rules (8 decks, S17, DAS, split once, no surrender, peek on ace only,
late dealer BJ takes all, Six Card Charlie) the result differs from a standard 8-deck S17 DAS chart in:
11 vs T hit (no double), 8,8 vs T hit (no split) — both because a 10-up blackjack is only discovered at
the end and takes doubled/split stakes — and in the multi-card Charlie decisions. See docs/RULES.md.

Action codes: ``"S"`` stand, ``"H"`` hit, ``"D"`` double else hit, ``"Ds"`` double else stand, ``"P"`` split.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..config import RulesConfig
from .solver import Solver, SolverRules, solve

STAND, HIT, DOUBLE_HIT, DOUBLE_STAND = 0, 1, 2, 3
_CODE = {"S": STAND, "H": HIT, "D": DOUBLE_HIT, "Ds": DOUBLE_STAND}
_NAME = ("S", "H", "D", "Ds")
UP_ORDER = (1, 2, 3, 4, 5, 6, 7, 8, 9, 0)  # chart column order 2..9, T, A
MAX_TABLE_CARDS = 5


@dataclass(frozen=True)
class Deviation:
    """Hi-Lo index play: play ``action`` instead of basic when ``tc >= index`` (or ``tc < index``
    if ``below``). ``kind`` is "hard" | "soft" | "pair" | "insurance"; ``value`` is the total or the
    pair bucket; ``up`` the dealer up-card bucket (0 = A, 9 = T)."""

    name: str
    kind: str
    value: int
    up: int
    index: float
    action: str  # "stand" | "hit" | "double" | "split" | "insure"
    below: bool = False

    def applies(self, tc: float) -> bool:
        return tc < self.index if self.below else tc >= self.index


class BasicStrategy:
    """Lookup tables: ``hard[n][total][up]``, ``soft[n][total][up]`` (codes), ``split[pair][up]`` (bool)."""

    def __init__(self, rules: SolverRules, num_decks: int = 8, refine: bool = True) -> None:
        self.rules = rules
        self.num_decks = num_decks
        base = solve(rules)
        # n = 0/1 unused; n >= MAX_TABLE_CARDS uses the 5-card table
        self.hard: list[list[list[int]]] = [[[HIT] * 10 for _ in range(32)] for _ in range(MAX_TABLE_CARDS + 1)]
        self.soft: list[list[list[int]]] = [[[HIT] * 10 for _ in range(32)] for _ in range(MAX_TABLE_CARDS + 1)]
        for n in range(2, MAX_TABLE_CARDS + 1):
            for t in range(4, 32):
                for u in range(10):
                    if t > 21:
                        self.hard[n][t][u] = STAND
                        continue
                    self.hard[n][t][u] = _CODE[base.best_code(t, False, n, u)]
                    if t >= 12:
                        self.soft[n][t][u] = _CODE[base.best_code(t, True, n, u)]
        self.split = [[base.best_code(12 if x == 0 else 2 * x + 2, x == 0, 2, u, x) == "P"
                       for u in range(10)] for x in range(10)]
        if refine and num_decks > 0:
            self._refine_two_card(num_decks)

    def _refine_two_card(self, num_decks: int) -> None:
        """Recompute 2-card decisions with the actual removal of both player cards and the up card."""
        r = self.rules
        full = [4.0 * num_decks] * 9 + [16.0 * num_decks]
        for u in range(10):
            acc: dict[tuple[int, bool], dict[str, float]] = {}
            for c1 in range(10):
                for c2 in range(c1, 10):
                    counts = full[:]
                    for c in (c1, c2, u):
                        counts[c] -= 1
                    if min(counts) < 0:
                        continue
                    # P(c1, c2 | up) without replacement, both orders
                    n_tot = sum(full) - 1
                    cu = full[:]
                    cu[u] -= 1
                    w = cu[c1] * (cu[c2] - (1 if c1 == c2 else 0)) / (n_tot * (n_tot - 1))
                    if c1 != c2:
                        w *= 2
                    if w <= 0:
                        continue
                    s = Solver(r, tuple(counts))
                    hard = c1 + c2 + 2
                    soft = (c1 == 0 or c2 == 0) and hard <= 11
                    t = hard + 10 if soft else hard
                    if soft and t == 21:
                        continue  # blackjack
                    evs = s.evs(t, soft, 2, u)
                    a = acc.setdefault((t, soft), {})
                    for k, v in evs.items():
                        a[k] = a.get(k, 0.0) + w * v
                    a["_w"] = a.get("_w", 0.0) + w
                    if c1 == c2 and r.can_split:
                        ev_split = s.split_ev(c1, u)
                        self.split[c1][u] = ev_split > max(evs.values())
            for (t, soft), a in acc.items():
                evs = {k: v for k, v in a.items() if k != "_w"}
                best = max(evs, key=evs.__getitem__)
                if best == "D":
                    code = DOUBLE_HIT if evs["H"] >= evs["S"] else DOUBLE_STAND
                else:
                    code = _CODE[best]
                (self.soft if soft else self.hard)[2][t][u] = code

    # ------------------------------------------------------------------ lookups
    def code(self, total: int, soft: bool, ncards: int, up: int) -> int:
        n = ncards if ncards < MAX_TABLE_CARDS else MAX_TABLE_CARDS
        if n < 2:
            n = 2
        return (self.soft if soft else self.hard)[n][total][up]

    def action(self, total: int, soft: bool, ncards: int, up: int, can_double: bool = True,
               pair: int = -1, can_split: bool = False) -> str:
        """Basic-strategy action name: "stand" | "hit" | "double" | "split"."""
        if pair >= 0 and can_split and self.split[pair][up]:
            return "split"
        c = self.code(total, soft, ncards, up)
        if c == STAND:
            return "stand"
        if c == HIT:
            return "hit"
        if can_double and ncards == 2:
            return "double"
        return "hit" if c == DOUBLE_HIT else "stand"

    def chart(self, ncards: int = 2) -> str:
        """Human-readable chart (rows = totals, columns = dealer 2..9, T, A)."""
        head = "      " + " ".join(f"{b:>2}" for b in "23456789TA")
        rows = [f"{ncards}-card hands", head]
        for t in range(5, 22):
            rows.append(f"H{t:<4} " + " ".join(f"{_NAME[self.hard[ncards][t][u]]:>2}" for u in UP_ORDER))
        for t in range(13, 22):
            rows.append(f"S{t:<4} " + " ".join(f"{_NAME[self.soft[ncards][t][u]]:>2}" for u in UP_ORDER))
        if ncards == 2:
            for x in UP_ORDER:
                lab = "AA" if x == 0 else ("TT" if x == 9 else f"{x + 1}{x + 1}")
                rows.append(f"P{lab:<4} " + " ".join(f"{'P' if self.split[x][u] else '-':>2}" for u in UP_ORDER))
        return "\n".join(rows)


@lru_cache(maxsize=16)
def _cached(rules: SolverRules, num_decks: int) -> BasicStrategy:
    return BasicStrategy(rules, num_decks)


def get_basic_strategy(rules: RulesConfig | SolverRules, num_decks: int = 8) -> BasicStrategy:
    """Cached :class:`BasicStrategy` for the given rules / number of decks."""
    sr = rules if isinstance(rules, SolverRules) else SolverRules.from_config(rules)
    return _cached(sr, num_decks)


# ---------------------------------------------------------------------- Hi-Lo deviations
#: Illustrious 18 adapted to these rules. Every index was recomputed with
#: :func:`onebj.strategy.solver.hilo_index` (composition shifted along the Hi-Lo direction, infinite
#: deck); the same method reproduces the published I18 for standard peek rules within ±0.6. For the ONE
#: Blackjack rules (late dealer BJ on tens takes doubles) "10 v T double" never becomes correct and is
#: dropped; "11 v T double" (basic: hit) is added at +3 (solver +3.4); "12 v 3 stand" is +1 (solver
#: +1.3); "11 v A double" is +2 (solver +1.6). Indices apply to 2-card hands only.
ILLUSTRIOUS_18: tuple[Deviation, ...] = (
    Deviation("insurance", "insurance", 0, 0, 3.0, "insure"),
    Deviation("16 v T stand", "hard", 16, 9, 0.0, "stand"),
    Deviation("15 v T stand", "hard", 15, 9, 4.0, "stand"),
    Deviation("TT v 5 split", "pair", 9, 4, 5.0, "split"),
    Deviation("TT v 6 split", "pair", 9, 5, 4.0, "split"),
    Deviation("11 v T double", "hard", 11, 9, 3.0, "double"),
    Deviation("12 v 3 stand", "hard", 12, 2, 1.0, "stand"),
    Deviation("12 v 2 stand", "hard", 12, 1, 3.0, "stand"),
    Deviation("11 v A double", "hard", 11, 0, 2.0, "double"),
    Deviation("9 v 2 double", "hard", 9, 1, 1.0, "double"),
    Deviation("10 v A double", "hard", 10, 0, 4.0, "double"),
    Deviation("9 v 7 double", "hard", 9, 6, 3.0, "double"),
    Deviation("16 v 9 stand", "hard", 16, 8, 5.0, "stand"),
    Deviation("13 v 2 hit", "hard", 13, 1, -1.0, "hit", below=True),
    Deviation("12 v 4 hit", "hard", 12, 3, 0.0, "hit", below=True),
    Deviation("12 v 5 hit", "hard", 12, 4, -2.0, "hit", below=True),
    Deviation("12 v 6 hit", "hard", 12, 5, -1.0, "hit", below=True),
    Deviation("13 v 3 hit", "hard", 13, 2, -2.0, "hit", below=True),
)


def hilo_action(bs: BasicStrategy, total: int, soft: bool, ncards: int, up: int, can_double: bool,
                pair: int, can_split: bool, tc: float,
                deviations: tuple[Deviation, ...] = ILLUSTRIOUS_18) -> tuple[str, Deviation | None]:
    """Basic strategy with Hi-Lo index plays. Returns (action, deviation applied or None)."""
    basic = bs.action(total, soft, ncards, up, can_double, pair, can_split)
    if ncards != 2:
        return basic, None
    for d in deviations:
        if d.up != up or d.kind == "insurance":
            continue
        if d.kind == "pair":
            if pair != d.value or not can_split:
                continue
        elif basic == "split":
            continue  # total-based indices never override a basic-strategy split
        elif d.kind == "soft" and (not soft or total != d.value):
            continue
        elif d.kind == "hard" and (soft or total != d.value):
            continue
        if d.action == "double" and not can_double:
            continue
        if d.applies(tc) and d.action != basic:
            return d.action, d
        return basic, None
    return basic, None


def should_insure(tc: float, deviations: tuple[Deviation, ...] = ILLUSTRIOUS_18) -> bool:
    """Hi-Lo insurance decision (take insurance at TC >= +3 by default)."""
    for d in deviations:
        if d.kind == "insurance":
            return d.applies(tc)
    return False
