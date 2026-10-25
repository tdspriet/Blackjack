"""Virtual crowd sharing the hand (SPEC §1.4).

Every crowd player of a given type plays deterministically, so the crowd is represented by the number
of players per type (sampled at each shoe start) and, per round, the stop state of each type:

* ``basic`` — basic strategy (doubles where basic doubles, never splits; pairs played as totals)
* ``cautious`` — the official timeout default: stand on 12 or more, hit on 11 or less; never doubles
* ``dealer_mimic`` — hits below 17, never doubles
* ``reckless`` — hits below 19, never doubles

Before every card of the shared hand each still-active type decides stand / hit / double. A doubling
type takes exactly that one card. A type at ``charlie_cards`` cards stops with a Charlie.
"""

from __future__ import annotations

import random

from .config import CrowdConfig
from .strategy.basic import BasicStrategy

TYPES: tuple[str, ...] = ("basic", "cautious", "dealer_mimic", "reckless")
BASIC, CAUTIOUS, MIMIC, RECKLESS = range(4)
# per-type round state
ACTIVE, STOOD, DOUBLING, DOUBLED, BUST, CHARLIE = range(6)


class Crowd:
    """Crowd type counts + per-round stop states."""

    __slots__ = ("cfg", "strategy", "charlie", "counts", "state", "totals", "enabled")

    def __init__(self, cfg: CrowdConfig, strategy: BasicStrategy, charlie_cards: int) -> None:
        self.cfg = cfg
        self.enabled = cfg.enabled and cfg.size > 0
        self.strategy = strategy
        self.charlie = charlie_cards
        self.counts = [0, 0, 0, 0]
        self.state = [STOOD] * 4
        self.totals = [0] * 4

    def resample(self, rng: random.Random) -> None:
        """Sample the type of each of ``size`` players (called at each shoe start)."""
        counts = [0, 0, 0, 0]
        if self.enabled:
            weights = [max(0.0, float(self.cfg.mix.get(t, 0.0))) for t in TYPES]
            tot = sum(weights) or 1.0
            cum, acc = [], 0.0
            for w in weights:
                acc += w / tot
                cum.append(acc)
            for _ in range(self.cfg.size):
                x = rng.random()
                k = 0
                while k < 3 and x >= cum[k]:
                    k += 1
                counts[k] += 1
        self.counts = counts

    def start_round(self) -> None:
        cnt = self.counts
        self.state = [ACTIVE if cnt[i] else STOOD for i in range(4)]
        self.totals = [0, 0, 0, 0]

    def _wants(self, typ: int, total: int, soft: bool, n: int, up: int) -> int:
        """0 = stand, 1 = hit, 2 = double."""
        if total >= 21:
            return 0
        if typ == BASIC:
            a = self.strategy.action(total, soft, n, up, n == 2)
            return 1 if a == "hit" else (2 if a == "double" else 0)
        if typ == CAUTIOUS:
            return 1 if total <= 11 else 0
        if typ == MIMIC:
            return 1 if total < 17 else 0
        return 1 if total < 19 else 0

    def before_card(self, total: int, soft: bool, n: int, up: int) -> int:
        """Active types decide on the current shared hand; returns the number of crowd players taking
        the next card (types that stand are frozen at ``total``)."""
        hitters = 0
        st, cnt = self.state, self.counts
        for i in range(4):
            if st[i] != ACTIVE:
                continue
            w = self._wants(i, total, soft, n, up)
            if w == 0:
                st[i] = STOOD
                self.totals[i] = total
            else:
                if w == 2:
                    st[i] = DOUBLING
                hitters += cnt[i]
        return hitters

    def after_card(self, total: int, n: int) -> None:
        """Update stop states after a shared card was dealt (bust / Charlie / doubled)."""
        st = self.state
        for i in range(4):
            s = st[i]
            if s != ACTIVE and s != DOUBLING:
                continue
            if total > 21:
                st[i] = BUST
            elif self.charlie and n >= self.charlie:
                st[i] = CHARLIE
            elif s == DOUBLING:
                st[i] = DOUBLED
            else:
                continue
            self.totals[i] = total

    def any_active(self) -> bool:
        return ACTIVE in self.state

    def needs_dealer(self) -> bool:
        """True if some crowd type ended with a live (stood/doubled, not bust/Charlie) hand."""
        cnt = self.counts
        return any(cnt[i] and self.state[i] in (ACTIVE, STOOD, DOUBLED) for i in range(4))

    @property
    def hitters(self) -> int:
        return sum(self.counts[i] for i in range(4) if self.state[i] in (ACTIVE, DOUBLING))

    @property
    def size(self) -> int:
        return sum(self.counts)

    def copy(self) -> "Crowd":
        c = Crowd.__new__(Crowd)
        c.cfg, c.strategy, c.charlie, c.enabled = self.cfg, self.strategy, self.charlie, self.enabled
        c.counts, c.state, c.totals = self.counts[:], self.state[:], self.totals[:]
        return c
