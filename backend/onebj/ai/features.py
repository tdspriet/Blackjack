"""Observation → named, normalised feature vector for the network (SPEC §7).

One encoder serves the three decision kinds, so the whole agent is a single network with a single input
layer (which keeps the visualisation readable):

* ``play``      — a pending hit/stand/double/split/cash-out decision (hand + dealer + shoe features),
* ``insurance`` — the insurance question (initial two cards, dealer ace, shoe features),
* ``bet``       — before the deal (only shoe features; hand/dealer features are zero).

Only legal information is used: everything comes from :class:`onebj.engine.Observation` (or the
equivalent fast path on a :class:`~onebj.engine.Table`, which reads exactly the same public state — the
hole card is never touched). ``tests/test_ai.py`` checks that both paths give identical vectors.

Default feature set (61 features)::

    hand         player_total, total_4..total_21 (one-hot), soft, num_cards, pair_A..pair_T (one-hot of a
                 splittable pair), can_double, can_split, is_split, cashout_value (= offer / bet - 1, the
                 exact EV of cashing out; 0 when unavailable)
    dealer       dealer_up_A..dealer_up_T (one-hot)
    composition  unseen_A..unseen_T: relative richness of the unseen cards per rank bucket,
                 (unseen_r / unseen - expected_r) / expected_r * 2.5, clipped to [-2, 2]
                 (per-rank, so the 7s needed by Crazy 7 have their own input)
    count/shoe   hilo_tc (true count / 6), decks_unseen (/ num_decks), shoe_half2, cards_to_cut
                 (cards left before the cut card / half size)
    context      phase_bet, phase_ins (both 0 for play decisions)

The feature *set* is versioned through :class:`FeatureSpec` and stored in every checkpoint, so old
checkpoints keep working if defaults change.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..cards import BUCKET_OF, BUCKETS
from ..config import GameConfig

if TYPE_CHECKING:
    from ..engine import Observation, Table

#: expected fraction of each rank bucket in a full shoe (A, 2..9, T)
EXPECTED_FRAC: tuple[float, ...] = (1 / 13,) * 9 + (4 / 13,)
HILO_TAGS: tuple[int, ...] = (-1, 1, 1, 1, 1, 1, 0, 0, 0, -1)

KIND_PLAY, KIND_INSURANCE, KIND_BET = "play", "insurance", "bet"


@dataclass(frozen=True)
class FeatureSpec:
    """Knobs of the feature set (stored in checkpoints).

    * ``total_onehot``: inclusive range of the one-hot player-total block (totals outside the range only
      appear through ``player_total``).
    * ``composition_scale``: multiplier of the relative richness inputs (``unseen_*``).
    * ``tc_scale``: the Hi-Lo true count is divided by this.
    """

    version: int = 1
    total_onehot: tuple[int, int] = (4, 21)
    composition_scale: float = 2.5
    tc_scale: float = 6.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_onehot"] = list(self.total_onehot)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FeatureSpec:
        if not d:
            return cls()
        d = dict(d)
        if "total_onehot" in d:
            d["total_onehot"] = tuple(d["total_onehot"])
        return cls(**d)


def feature_names(spec: FeatureSpec = FeatureSpec()) -> list[str]:
    lo, hi = spec.total_onehot
    names = ["player_total"]
    names += [f"total_{t}" for t in range(lo, hi + 1)]
    names += ["soft", "num_cards"]
    names += [f"pair_{b}" for b in BUCKETS]
    names += ["can_double", "can_split", "is_split", "cashout_value"]
    names += [f"dealer_up_{b}" for b in BUCKETS]
    names += [f"unseen_{b}" for b in BUCKETS]
    names += ["hilo_tc", "decks_unseen", "shoe_half2", "cards_to_cut"]
    names += ["phase_bet", "phase_ins"]
    return names


@dataclass
class HandState:
    """Public description of the hand a play/insurance decision is about."""

    total: int
    soft: bool
    num_cards: int
    pair: int = -1  # bucket of a splittable pair, else -1
    can_double: bool = False
    can_split: bool = False
    is_split: bool = False
    cashout_value: float | None = None  # offer / bet - 1, None when not available


@dataclass
class ShoeState:
    """Public shoe knowledge: seen cards per bucket and the shoe position."""

    seen_buckets: Sequence[int]
    num_decks: int = 8
    half: int = 1
    cards_to_cut: int = 0
    half_size: int = 208
    extra: dict[str, Any] = field(default_factory=dict)


class FeatureEncoder:
    """Builds feature vectors (``np.float32``) plus, on request, human-readable raw values for traces."""

    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self.spec = spec or FeatureSpec()
        self.names = feature_names(self.spec)
        self.size = len(self.names)
        ix = {n: i for i, n in enumerate(self.names)}
        self.index = ix
        lo, hi = self.spec.total_onehot
        self._tlo, self._thi = lo, hi
        self.i_total = ix["player_total"]
        self.i_tot0 = ix[f"total_{lo}"]
        self.i_soft = ix["soft"]
        self.i_n = ix["num_cards"]
        self.i_pair0 = ix["pair_A"]
        self.i_cd = ix["can_double"]
        self.i_cs = ix["can_split"]
        self.i_split = ix["is_split"]
        self.i_cash = ix["cashout_value"]
        self.i_up0 = ix["dealer_up_A"]
        self.i_un0 = ix["unseen_A"]
        self.i_tc = ix["hilo_tc"]
        self.i_decks = ix["decks_unseen"]
        self.i_half = ix["shoe_half2"]
        self.i_cut = ix["cards_to_cut"]
        self.i_pbet = ix["phase_bet"]
        self.i_pins = ix["phase_ins"]
        self._cscale = self.spec.composition_scale
        self._tcscale = self.spec.tc_scale

    # ------------------------------------------------------------------ core
    def encode(self, kind: str, hand: HandState | None, up_bucket: int | None, shoe: ShoeState,
               raw: bool = False) -> np.ndarray | tuple[np.ndarray, list[Any]]:
        """Feature vector for a decision of ``kind`` ("play" | "insurance" | "bet")."""
        v = [0.0] * self.size
        r: list[Any] | None = [None] * self.size if raw else None
        if hand is not None:
            t = hand.total
            v[self.i_total] = t / 21.0
            if self._tlo <= t <= self._thi:
                v[self.i_tot0 + t - self._tlo] = 1.0
            v[self.i_soft] = 1.0 if hand.soft else 0.0
            v[self.i_n] = hand.num_cards / 6.0
            if hand.pair >= 0 and hand.can_split:
                v[self.i_pair0 + hand.pair] = 1.0
            v[self.i_cd] = 1.0 if hand.can_double else 0.0
            v[self.i_cs] = 1.0 if hand.can_split else 0.0
            v[self.i_split] = 1.0 if hand.is_split else 0.0
            cv = hand.cashout_value
            v[self.i_cash] = 0.0 if cv is None else cv
            if r is not None:
                r[self.i_total] = t
                r[self.i_soft] = hand.soft
                r[self.i_n] = hand.num_cards
                r[self.i_cd] = hand.can_double
                r[self.i_cs] = hand.can_split
                r[self.i_split] = hand.is_split
                r[self.i_cash] = None if cv is None else round(cv + 1.0, 4)
        if up_bucket is not None:
            v[self.i_up0 + up_bucket] = 1.0
        # composition / count
        seen = shoe.seen_buckets
        nd = shoe.num_decks
        full_t = 16 * nd
        full_o = 4 * nd
        n_seen = 0
        for x in seen:
            n_seen += x
        unseen = 52 * nd - n_seen
        n = unseen if unseen > 0 else 1
        cs = self._cscale
        i0 = self.i_un0
        for b in range(10):
            u = (full_t if b == 9 else full_o) - seen[b]
            rel = (u / n / EXPECTED_FRAC[b] - 1.0) * cs
            v[i0 + b] = 2.0 if rel > 2.0 else (max(rel, -2.0))
            if r is not None:
                r[i0 + b] = u
        rc = seen[1] + seen[2] + seen[3] + seen[4] + seen[5] - seen[0] - seen[9]
        tc = rc * 52.0 / n
        v[self.i_tc] = tc / self._tcscale
        v[self.i_decks] = unseen / 52.0 / nd
        v[self.i_half] = 1.0 if shoe.half == 2 else 0.0
        cut = shoe.cards_to_cut
        v[self.i_cut] = (max(0, cut)) / shoe.half_size
        if kind == KIND_BET:
            v[self.i_pbet] = 1.0
        elif kind == KIND_INSURANCE:
            v[self.i_pins] = 1.0
        arr = np.asarray(v, dtype=np.float32)
        if r is None:
            return arr
        r[self.i_tc] = round(tc, 2)
        r[self.i_decks] = round(unseen / 52.0, 2)
        r[self.i_half] = shoe.half
        r[self.i_cut] = cut
        r[self.i_pbet] = kind == KIND_BET
        r[self.i_pins] = kind == KIND_INSURANCE
        if hand is not None:
            if hand.pair >= 0 and hand.can_split:
                r[self.i_pair0 + hand.pair] = BUCKETS[hand.pair]
            if self._tlo <= hand.total <= self._thi:
                r[self.i_tot0 + hand.total - self._tlo] = hand.total
        if up_bucket is not None:
            r[self.i_up0 + up_bucket] = BUCKETS[up_bucket]
        return arr, r

    # ------------------------------------------------------------------ from an Observation
    @staticmethod
    def shoe_from_obs(obs: Observation, half_size: int) -> ShoeState:
        return ShoeState(obs.seen_counts, obs.num_decks, obs.half, obs.cards_left_in_half_to_cut, half_size)

    def encode_obs(self, obs: Observation, kind: str, half_size: int, raw: bool = False):
        """Encode an :class:`Observation` (``half_size`` = cards per shoe half, from the config)."""
        shoe = self.shoe_from_obs(obs, half_size)
        if kind == KIND_BET:
            return self.encode(kind, None, None, shoe, raw)
        h = obs.hand
        up = BUCKET_OF[obs.dealer_up]
        if kind == KIND_INSURANCE:
            hs = HandState(h.total, h.soft, h.num_cards)
            return self.encode(kind, hs, up, shoe, raw)
        cv = None
        if h.cashout_offer is not None and h.bet:
            cv = h.cashout_offer / h.bet - 1.0
        hs = HandState(h.total, h.soft, h.num_cards, h.pair, h.can_double, h.can_split, h.is_split, cv)
        return self.encode(kind, hs, up, shoe, raw)

    # ------------------------------------------------------------------ fast path from a Table
    @staticmethod
    def shoe_from_table(table: Table) -> ShoeState:
        s = table.shoe
        return ShoeState(s.seen_bucket, s.num_decks, s.half, s.cut_index - s.pos, s.half_size)

    def encode_table(self, table: Table, kind: str, raw: bool = False):
        """Same vector as :meth:`encode_obs` on ``table.observation()``, without building it."""
        shoe = self.shoe_from_table(table)
        if kind == KIND_BET:
            return self.encode(kind, None, None, shoe, raw)
        h = table.hands[table.active]
        up = BUCKET_OF[table.dealer.cards[0]]
        if kind == KIND_INSURANCE:
            return self.encode(kind, HandState(h.total, h.soft, len(h.cards)), up, shoe, raw)
        total, soft, n, _up, can_double, pair, can_split, is_split = table.decision_state()
        offer = table.cashout_offer()
        cv = offer / h.bet - 1.0 if offer is not None and h.bet else None
        return self.encode(kind, HandState(total, soft, n, pair, can_double, can_split, is_split, cv), up,
                           shoe, raw)

    def encode_play_state(self, table: Table, state: tuple) -> np.ndarray:
        """Fast path for agents that already hold ``table.decision_state()``."""
        total, soft, n, up, can_double, pair, can_split, is_split = state
        h = table.hands[table.active]
        offer = table.cashout_offer()
        cv = offer / h.bet - 1.0 if offer is not None and h.bet else None
        return self.encode(KIND_PLAY, HandState(total, soft, n, pair, can_double, can_split, is_split, cv), up,
                           self.shoe_from_table(table))


def synthetic_shoe(unseen_buckets: Sequence[float], num_decks: int = 8, half: int = 2,
                   cards_to_cut: int = 40, half_size: int = 208) -> ShoeState:
    """ShoeState for a hypothetical unseen composition (strategy charts / deviation scans).
    ``unseen_buckets`` may be fractional; seen = full - unseen."""
    full = [4 * num_decks] * 9 + [16 * num_decks]
    seen = [full[b] - unseen_buckets[b] for b in range(10)]
    return ShoeState(seen, num_decks, half, cards_to_cut, half_size)


def hilo_composition(tc: float, decks_unseen: float, num_decks: int = 8) -> list[float]:
    """Unseen bucket counts for ``decks_unseen`` decks at Hi-Lo true count ``tc``: the low cards (2-6)
    are depleted and the high cards (T, A) enriched by tc/2 cards per deck each (same convention as
    :func:`onebj.strategy.solver.hilo_shift_probs`)."""
    counts = [4.0] * 9 + [16.0]
    for b in (1, 2, 3, 4, 5):
        counts[b] -= tc / 2 / 5
    counts[0] += tc / 2 * 4 / 20
    counts[9] += tc / 2 * 16 / 20
    return [c * decks_unseen for c in counts]


def cfg_half_size(cfg: GameConfig) -> int:
    total = 52 * cfg.shoe.num_decks
    return total // 2 if cfg.shoe.split_halves else total
