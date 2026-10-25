"""Cash-out ("Uitbetalen") offers.

Official rule: the offer is constantly recalculated from the hand value, the dealer up card and the
player's decisions; the option's RTP is 98%. We model it as

    offer = cashout_rtp × fair value × hand bet

where *fair value* = expected return (incl. the stake) per unit of the hand's bet under optimal
composition-independent play vs that up card, for a full fresh shoe (infinite-deck probabilities
1/13 per rank, the default composition of :mod:`onebj.strategy.solver`). It includes Six Card Charlie,
no peek on tens (late dealer blackjack per ``late_dealer_bj_takes``), the conditioning on "no dealer
blackjack" after an ace peek, DAS and the one-split rule. The live shoe composition is NOT used — which
is exactly what can make a cash-out +EV for a counter in a negative shoe.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import RulesConfig
from .solver import SolverRules, solve


class CashoutTable:
    """Fair values (EV + 1 per unit bet) of hand states vs up cards, memoised."""

    def __init__(self, rules: SolverRules, rtp: float = 0.98) -> None:
        self.rules = rules
        self.rtp = rtp
        self.solver = solve(rules)
        self._cache: dict[tuple, float] = {}

    def fair_value(self, total: int, soft: bool, ncards: int, up: int, can_double: bool = False,
                   pair: int = -1, split_hand: bool = False) -> float:
        """Expected return per unit bet (stake included) of continuing optimally."""
        key = (total, soft, ncards, up, can_double, pair, split_hand)
        v = self._cache.get(key)
        if v is None:
            if total > 21:
                v = 0.0
            else:
                evs = self.solver.evs(total, soft, ncards, up, pair, can_double, pair >= 0, split_hand)
                v = 1.0 + max(evs.values())
            self._cache[key] = v
        return v

    def offer(self, total: int, soft: bool, ncards: int, up: int, can_double: bool = False, pair: int = -1,
              split_hand: bool = False) -> float:
        """Cash-out offer per unit of the hand's bet."""
        return self.rtp * self.fair_value(total, soft, ncards, up, can_double, pair, split_hand)


@lru_cache(maxsize=16)
def _cached(rules: SolverRules, rtp: float) -> CashoutTable:
    return CashoutTable(rules, rtp)


def get_cashout_table(rules: RulesConfig) -> CashoutTable:
    return _cached(SolverRules.from_config(rules), rules.cashout_rtp)
