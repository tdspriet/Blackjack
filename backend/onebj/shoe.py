"""NapoleonShoe (SPEC §1.1): 8 decks shuffled together, split into two 4-deck halves.

* Each half gets ``burn_cards`` unseen burn cards and a cut card after ``penetration * half_size``
  cards (± uniform ``cut_jitter``). When the cut card comes out the current round is completed;
  :meth:`end_round` then switches to half 2, or (after half 2) performs a full reshuffle.
* Seen/unseen composition is tracked over the **whole 8-deck shoe** and is only reset by a reshuffle,
  never at the half switch: everything not seen (remaining cards of both halves, burn cards, the
  residue behind half 1's cut card, an unrevealed hole card) is equally unknown.
* ``split_halves=False`` gives a single classic shoe of ``num_decks`` decks.

The shoe does not decide what is "seen": callers draw with :meth:`draw` and mark exposed cards with
:meth:`mark_seen` (the engine does this for every face-up card and for the hole card when revealed).
"""

from __future__ import annotations

import random
from typing import Callable

from .cards import BUCKET_OF, NUM_CARDS, full_bucket_counts
from .config import ShoeConfig

#: cards always kept behind the cut card so a round can never run a half dry
_SAFETY = 40


class NapoleonShoe:
    """The physical shoe: card order, position, halves, cut card and seen-card bookkeeping."""

    def __init__(self, cfg: ShoeConfig, rng: random.Random) -> None:
        self.cfg = cfg
        self.rng = rng
        self.num_decks = cfg.num_decks
        self.total_cards = 52 * cfg.num_decks
        self.split = cfg.split_halves
        self.half_size = self.total_cards // 2 if self.split else self.total_cards
        self.cards: list[int] = [c for c in range(NUM_CARDS) for _ in range(cfg.num_decks)]
        self.shoe_number = 0
        self.half = 1
        self.half_start = 0
        self.half_end = self.half_size
        self.pos = 0
        self.cut_at = 0  # cut card position relative to the half start
        self.cut_index = 0  # absolute index of the first card behind the cut card
        self.cut_out = False
        self.burned = 0
        self.burned_cards: list[int] = []
        self.full_buckets = full_bucket_counts(cfg.num_decks)
        self.seen_card = [0] * NUM_CARDS
        self.seen_rank = [0] * 13
        self.seen_bucket = [0] * 10
        self.cards_seen = 0
        self.shuffle()

    # ------------------------------------------------------------------ shoe life cycle
    def shuffle(self) -> None:
        """Full reshuffle: new shoe, new split, start half 1 (burn + cut card). Resets seen counts."""
        self.rng.shuffle(self.cards)
        self.shoe_number += 1
        self.seen_card = [0] * NUM_CARDS
        self.seen_rank = [0] * 13
        self.seen_bucket = [0] * 10
        self.cards_seen = 0
        self._start_half(1)

    def _start_half(self, half: int) -> None:
        self.half = half
        self.half_start = 0 if half == 1 else self.half_size
        self.half_end = self.half_start + self.half_size
        self.pos = self.half_start
        cfg = self.cfg
        cut = round(cfg.penetration * self.half_size)
        if cfg.cut_jitter:
            cut += self.rng.randint(-cfg.cut_jitter, cfg.cut_jitter)
        self.cut_at = max(cfg.burn_cards + 1, min(cut, self.half_size - _SAFETY))
        self.cut_index = self.half_start + self.cut_at
        self.cut_out = False
        self.burned = 0
        self.burned_cards = []
        for _ in range(cfg.burn_cards):
            self.burned_cards.append(self.draw())
            self.burned += 1

    def end_round(self) -> str | None:
        """Call between rounds. Returns ``"half_switch"`` / ``"reshuffle"`` if one happened, else None."""
        if not self.cut_out:
            return None
        if self.split and self.half == 1:
            self._start_half(2)
            return "half_switch"
        self.shuffle()
        return "reshuffle"

    @property
    def pending(self) -> str | None:
        """What :meth:`end_round` will do: ``"half_switch"``, ``"reshuffle"`` or None."""
        if not self.cut_out:
            return None
        return "half_switch" if self.split and self.half == 1 else "reshuffle"

    # ------------------------------------------------------------------ dealing
    def draw(self) -> int:
        """Next card (not marked seen). Sets :attr:`cut_out` when the cut card appears."""
        pos = self.pos
        if pos >= self.cut_index:
            self.cut_out = True
        if pos >= self.half_end:  # cannot happen with _SAFETY, kept as a hard guard
            raise RuntimeError("shoe half exhausted")
        self.pos = pos + 1
        return self.cards[pos]

    def mark_seen(self, card: int) -> None:
        self.seen_card[card] += 1
        self.seen_rank[card >> 2] += 1
        self.seen_bucket[BUCKET_OF[card]] += 1
        self.cards_seen += 1

    def deal_seen(self) -> int:
        """Draw a face-up card and mark it seen."""
        c = self.draw()
        self.mark_seen(c)
        return c

    # ------------------------------------------------------------------ views
    @property
    def dealt_in_half(self) -> int:
        return self.pos - self.half_start

    @property
    def cards_unseen(self) -> int:
        return self.total_cards - self.cards_seen

    @property
    def cards_left_to_cut(self) -> int:
        return self.cut_index - self.pos

    def unseen_buckets(self) -> list[int]:
        f, s = self.full_buckets, self.seen_bucket
        return [f[i] - s[i] for i in range(10)]

    def unseen_cards(self) -> list[int]:
        """Unseen count per card type (52 entries)."""
        d = self.num_decks
        return [d - s for s in self.seen_card]

    def unseen_ranks(self) -> list[int]:
        d4 = 4 * self.num_decks
        return [d4 - s for s in self.seen_rank]

    def future_positions(self) -> range | list[int]:
        """Absolute indices of cards that can still be dealt before the next reshuffle."""
        if self.split and self.half == 1:
            return list(range(self.pos, self.half_end)) + list(range(self.half_size, self.total_cards))
        return range(self.pos, self.half_end)

    def copy(self, rng: random.Random) -> "NapoleonShoe":
        s = NapoleonShoe.__new__(NapoleonShoe)
        s.__dict__.update(self.__dict__)
        s.rng = rng
        s.cards = self.cards[:]
        s.burned_cards = self.burned_cards[:]
        s.seen_card = self.seen_card[:]
        s.seen_rank = self.seen_rank[:]
        s.seen_bucket = self.seen_bucket[:]
        return s

    def resample_unseen(self, rng: random.Random, n_hidden: int = 0,
                        hidden_ok: Callable[[int], bool] | None = None) -> list[int]:
        """Reshuffle every unseen card among all not-yet-seen slots (future cards + residue/burn).

        ``n_hidden`` dealt-but-unseen cards held by the caller (e.g. the hole card) are resampled too;
        their new values are returned so the caller can replace them. ``hidden_ok`` constrains them
        (e.g. a peeked hole card cannot complete a blackjack): a uniform draw among acceptable cards.
        """
        unseen = self.unseen_cards()  # includes burn cards, residue and the hole card
        pool = [c for c in range(NUM_CARDS) for _ in range(unseen[c])]
        rng.shuffle(pool)
        out: list[int] = []
        for k in range(n_hidden):
            if hidden_ok is not None:
                j = next(j for j in range(k, len(pool)) if hidden_ok(pool[j]))
                pool[k], pool[j] = pool[j], pool[k]
            out.append(pool[k])
        k = n_hidden
        future = self.future_positions()
        for i in future:
            self.cards[i] = pool[k]
            k += 1
        # leftover = burn cards + residue behind half 1's cut card: never dealt again
        leftover = pool[k:]
        self.burned_cards = leftover[:len(self.burned_cards)]
        # Overwrite every already-dealt slot too (seen cards + resampled hidden/leftover cards), so the
        # clone's card array holds no trace of the true hole/burn cards and still is a full shoe.
        fut = set(future)
        past = [i for i in range(self.total_cards) if i not in fut]
        rest = [c for c in range(NUM_CARDS) for _ in range(self.seen_card[c])] + out + leftover
        for i, c in zip(past, rest):
            self.cards[i] = c
        return out
