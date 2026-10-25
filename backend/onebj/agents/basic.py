"""Baseline agents: flat basic strategy, and Hi-Lo (index plays + bet spread + positive-EV side bets).

Both also offer ``fast_bets(table)`` / ``fast_action(state)`` / ``fast_insurance(table)`` used by
:mod:`onebj.sim` to skip building an :class:`Observation` per decision (same decisions, much faster).
"""

from __future__ import annotations

import math
from typing import Sequence

from ..analytics import side_bet_evs
from ..config import GameConfig, load_config
from ..counting import hilo_true
from ..engine import Action, Bets, Observation, Table
from ..sidebets import SIDE_BETS
from ..strategy.basic import get_basic_strategy, hilo_action, should_insure
from .base import Trace

_ACT = {"hit": Action.HIT, "stand": Action.STAND, "double": Action.DOUBLE, "split": Action.SPLIT}
PLAY_LABELS = ["stand", "hit", "double", "split", "cashout"]


def _play_trace(chosen: str, basic: str, deviation: str | None, obs: Observation, tc: float | None) -> Trace:
    h = obs.hand
    feats = [{"name": "player_total", "value": h.total / 21, "raw": h.total},
             {"name": "soft", "value": float(h.soft), "raw": h.soft},
             {"name": "num_cards", "value": h.num_cards / 6, "raw": h.num_cards},
             {"name": "dealer_up", "value": 0.0 if obs.dealer_up is None else (obs.dealer_up // 4 + 1) / 13,
              "raw": obs.dealer_total}]
    if tc is not None:
        feats.append({"name": "hilo_true_count", "value": tc / 10, "raw": round(tc, 2)})
    legal = [a.value for a in obs.legal_actions]
    return {"kind": "play", "features": feats, "layers": [],
            "outputs": {"labels": PLAY_LABELS, "values": [1.0 if a == chosen else 0.0 for a in PLAY_LABELS],
                        "legal": [a in legal for a in PLAY_LABELS], "unit": "chosen (rule-based)"},
            "chosen": chosen, "reference": {"basic_strategy": basic, "hilo_deviation": deviation}, "extra": {}}


class BasicStrategyAgent:
    """Flat minimum (or ``unit``) main bet, never insures, plays basic strategy. ``side_bets`` lists side
    bets to place flat at the minimum every round (default none; used for RTP measurements)."""

    def __init__(self, config: GameConfig | None = None, unit: float | None = None, name: str = "basic",
                 side_bets: Sequence[str] = ()) -> None:
        self.cfg = config or load_config()
        self.name = name
        self.unit = unit if unit is not None else self.cfg.limits.main_min
        self.side_bets = tuple(side_bets)
        self.bs = get_basic_strategy(self.cfg.rules, self.cfg.shoe.num_decks)

    def _bets(self) -> Bets:
        b = Bets(main=self.unit)
        for name in self.side_bets:
            setattr(b, name, self.cfg.limits.side_min)
        return b

    def decide_bets(self, obs: Observation) -> tuple[Bets, Trace | None]:
        return self._bets(), None

    def decide_insurance(self, obs: Observation) -> tuple[bool, Trace | None]:
        return False, None

    def decide_action(self, obs: Observation) -> tuple[Action, Trace | None]:
        h = obs.hand
        a = self.bs.action(h.total, h.soft, h.num_cards, min(obs.dealer_up // 4, 9), h.can_double, h.pair,
                           h.can_split)
        return _ACT[a], _play_trace(a, a, None, obs, None)

    # fast paths (sim.py)
    def fast_bets(self, table: Table) -> Bets:
        return self._bets()

    def fast_insurance(self, table: Table) -> bool:
        return False

    def fast_action(self, table: Table, state: tuple) -> str:
        total, soft, n, up, can_double, pair, can_split, _ = state
        return self.bs.action(total, soft, n, up, can_double, pair, can_split)


#: default Hi-Lo bet ramp: (minimum floor(true count), units)
DEFAULT_SPREAD: tuple[tuple[int, int], ...] = ((6, 12), (5, 8), (4, 6), (3, 4), (2, 2))


class HiLoAgent:
    """Basic strategy + Illustrious-18 index plays, insurance at TC >= +3, 1-12 spread by true count,
    side bets (minimum stake) whenever their analytic EV for the composition exceeds ``side_threshold``."""

    def __init__(self, config: GameConfig | None = None, unit: float | None = None,
                 spread: Sequence[tuple[int, int]] = DEFAULT_SPREAD, side_bets: Sequence[str] = SIDE_BETS,
                 side_threshold: float = 0.0, name: str = "hilo") -> None:
        self.cfg = config or load_config()
        self.name = name
        self.unit = unit if unit is not None else self.cfg.limits.main_min
        self.spread = tuple(sorted(spread, reverse=True))
        self.side_bets = tuple(side_bets)
        self.side_threshold = side_threshold
        self.bs = get_basic_strategy(self.cfg.rules, self.cfg.shoe.num_decks)

    def units_for(self, tc: float) -> int:
        f = math.floor(tc)
        for min_tc, units in self.spread:
            if f >= min_tc:
                return units
        return 1

    def _bets(self, tc: float, unseen_cards: Sequence[int]) -> tuple[Bets, dict[str, float]]:
        lim = self.cfg.limits
        main = min(max(self.unit * self.units_for(tc), lim.main_min), lim.main_max)
        b = Bets(main=main)
        evs: dict[str, float] = {}
        if self.side_bets:
            evs = side_bet_evs(unseen_cards, self.cfg)
            for name in self.side_bets:
                if evs[name] > self.side_threshold:
                    setattr(b, name, lim.side_min)
        return b, evs

    def decide_bets(self, obs: Observation) -> tuple[Bets, Trace | None]:
        tc = obs.true_counts["hilo"]
        b, evs = self._bets(tc, obs.unseen_card_counts)
        trace: Trace = {"kind": "bet", "features": [{"name": "hilo_true_count", "value": tc / 10, "raw": round(tc, 2)},
                                                    {"name": "decks_unseen", "value": obs.decks_unseen / 8,
                                                     "raw": round(obs.decks_unseen, 2)}],
                        "layers": [], "outputs": {"labels": ["main_units"], "values": [b.main / self.unit],
                                                  "legal": [True], "unit": "units"},
                        "chosen": f"{b.main:g}", "reference": {"basic_strategy": None, "hilo_deviation": None},
                        "extra": {"predicted_ev": evs, "bet_fraction": b.main / max(obs.bankroll, 1e-9)}}
        return b, trace

    def decide_insurance(self, obs: Observation) -> tuple[bool, Trace | None]:
        tc = obs.true_counts["hilo"]
        take = should_insure(tc)
        trace: Trace = {"kind": "insurance", "features": [{"name": "hilo_true_count", "value": tc / 10,
                                                           "raw": round(tc, 2)}],
                        "layers": [], "outputs": {"labels": ["insure", "no_insurance"],
                                                  "values": [float(take), float(not take)], "legal": [True, True],
                                                  "unit": "chosen (rule-based)"},
                        "chosen": "insure" if take else "no_insurance",
                        "reference": {"basic_strategy": "no_insurance",
                                      "hilo_deviation": "insurance" if take else None},
                        "extra": {}}
        return take, trace

    def decide_action(self, obs: Observation) -> tuple[Action, Trace | None]:
        h = obs.hand
        tc = obs.true_counts["hilo"]
        up = min(obs.dealer_up // 4, 9)
        basic = self.bs.action(h.total, h.soft, h.num_cards, up, h.can_double, h.pair, h.can_split)
        a, dev = hilo_action(self.bs, h.total, h.soft, h.num_cards, up, h.can_double, h.pair, h.can_split, tc)
        return _ACT[a], _play_trace(a, basic, dev.name if dev else None, obs, tc)

    # fast paths (sim.py)
    def fast_bets(self, table: Table) -> Bets:
        shoe = table.shoe
        tc = hilo_true(shoe.seen_bucket, shoe.num_decks)
        return self._bets(tc, shoe.unseen_cards() if self.side_bets else ())[0]

    def fast_insurance(self, table: Table) -> bool:
        shoe = table.shoe
        return should_insure(hilo_true(shoe.seen_bucket, shoe.num_decks))

    def fast_action(self, table: Table, state: tuple) -> str:
        total, soft, n, up, can_double, pair, can_split, _ = state
        if n != 2:
            return self.bs.action(total, soft, n, up, can_double, pair, can_split)
        shoe = table.shoe
        tc = hilo_true(shoe.seen_bucket, shoe.num_decks)
        return hilo_action(self.bs, total, soft, n, up, can_double, pair, can_split, tc)[0]
