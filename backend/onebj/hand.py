"""Hand totals, soft hands, blackjack and Six Card Charlie."""

from __future__ import annotations

from typing import Iterable

from .cards import BUCKET_OF, VALUE_OF


def hand_value(cards: Iterable[int]) -> tuple[int, bool]:
    """(best total, is_soft) of card ints; an ace counts 11 when that doesn't bust."""
    hard = 0
    ace = False
    for c in cards:
        v = VALUE_OF[c]
        hard += v
        if v == 1:
            ace = True
    if ace and hard <= 11:
        return hard + 10, True
    return hard, False


def total_of(hard: int, aces: int) -> int:
    """Best total from the hard sum (aces counted 1) and ace count."""
    return hard + 10 if aces and hard <= 11 else hard


def is_blackjack(cards: list[int], from_split: bool = False) -> bool:
    """Natural: exactly two cards totalling 21, not on a split hand."""
    return (not from_split and len(cards) == 2
            and VALUE_OF[cards[0]] + VALUE_OF[cards[1]] == 11
            and (VALUE_OF[cards[0]] == 1 or VALUE_OF[cards[1]] == 1))


def is_charlie(cards: list[int], charlie_cards: int = 6) -> bool:
    """Six (``charlie_cards``) card non-busted hand; ``charlie_cards == 0`` disables the rule."""
    return bool(charlie_cards) and len(cards) >= charlie_cards and hand_value(cards)[0] <= 21


class Hand:
    """Mutable hand used by the engine (``__slots__`` for speed).

    ``status``: "active" | "stood" | "bust" | "blackjack" | "charlie" | "done";
    ``result``: None | "win" | "lose" | "push" | "blackjack" | "charlie".
    """

    __slots__ = ("cards", "hard", "aces", "bet", "doubled", "is_split", "split_aces", "status",
                 "result", "payout")

    def __init__(self, bet: float = 0.0, is_split: bool = False) -> None:
        self.cards: list[int] = []
        self.hard = 0
        self.aces = 0
        self.bet = bet
        self.doubled = False
        self.is_split = is_split
        self.split_aces = False
        self.status = "active"
        self.result: str | None = None
        self.payout = 0.0

    def add(self, card: int) -> None:
        self.cards.append(card)
        v = VALUE_OF[card]
        self.hard += v
        if v == 1:
            self.aces += 1

    @property
    def total(self) -> int:
        return self.hard + 10 if self.aces and self.hard <= 11 else self.hard

    @property
    def soft(self) -> bool:
        return bool(self.aces) and self.hard <= 11

    @property
    def bust(self) -> bool:
        return self.hard > 21

    @property
    def blackjack(self) -> bool:
        return not self.is_split and len(self.cards) == 2 and self.aces > 0 and self.hard == 11

    def pair_bucket(self) -> int:
        """Bucket of the pair if this is a two-card pair by value (10/J/Q/K pair), else -1."""
        if len(self.cards) == 2 and BUCKET_OF[self.cards[0]] == BUCKET_OF[self.cards[1]]:
            return BUCKET_OF[self.cards[0]]
        return -1

    def copy(self) -> "Hand":
        h = Hand.__new__(Hand)
        h.cards = self.cards[:]
        h.hard, h.aces, h.bet, h.doubled = self.hard, self.aces, self.bet, self.doubled
        h.is_split, h.split_aces, h.status = self.is_split, self.split_aces, self.status
        h.result, h.payout = self.result, self.payout
        return h

    def __repr__(self) -> str:
        from .cards import CODES
        return f"Hand({[CODES[c] for c in self.cards]}, total={self.total}, status={self.status})"
