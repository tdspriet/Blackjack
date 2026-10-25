"""Side bet settlement (SPEC §1.5). Odds are "X to 1"; a winning stake is returned on top.

Each ``settle_*`` returns ``(outcome, odds)``: ``outcome`` is None for a losing bet, ``"push"`` for a
returned stake (odds 0), otherwise a label such as ``"mixed_pair"``; :func:`payout` turns it into money.
"""

from __future__ import annotations

from typing import Mapping

from .cards import IS_RED, RANK_OF, SUIT_OF

SIDE_BETS: tuple[str, ...] = ("perfect_pairs", "twenty_one_plus_three", "crazy_7", "bust_bonus")
SEVEN = 6  # rank index of the 7

Outcome = tuple[str | None, float]


def payout(stake: float, outcome: str | None, odds: float) -> float:
    """Money returned for a settled side bet (stake + winnings, the stake on a push, else 0)."""
    if outcome is None:
        return 0.0
    if outcome == "push":
        return stake
    return stake * (odds + 1)


def settle_perfect_pairs(c1: int, c2: int, paytable: Mapping[str, float]) -> Outcome:
    """Player's first two cards: perfect (same suit), coloured (same colour), mixed (same rank)."""
    if RANK_OF[c1] != RANK_OF[c2]:
        return None, 0.0
    if SUIT_OF[c1] == SUIT_OF[c2]:
        return "perfect_pair", paytable["perfect"]
    if IS_RED[c1] == IS_RED[c2]:
        return "coloured_pair", paytable["coloured"]
    return "mixed_pair", paytable["mixed"]


def is_straight(r1: int, r2: int, r3: int) -> bool:
    """Three distinct consecutive ranks; ace plays low (A-2-3) or high (Q-K-A), no wrap (K-A-2)."""
    a, b, c = sorted((r1, r2, r3))
    if a == b or b == c:
        return False
    return (c - a == 2) or (a == 0 and b == 11 and c == 12)


def settle_21_3(c1: int, c2: int, up: int, paytable: Mapping[str, float]) -> Outcome:
    """Player's two cards + dealer up card as a 3-card poker hand (highest hand only)."""
    r1, r2, r3 = RANK_OF[c1], RANK_OF[c2], RANK_OF[up]
    flush = SUIT_OF[c1] == SUIT_OF[c2] == SUIT_OF[up]
    if r1 == r2 == r3:
        return ("suited_trips", paytable["suited_trips"]) if flush else \
            ("three_of_a_kind", paytable["three_of_a_kind"])
    straight = is_straight(r1, r2, r3)
    if straight and flush:
        return "straight_flush", paytable["straight_flush"]
    if straight:
        return "straight", paytable["straight"]
    if flush:
        return "flush", paytable["flush"]
    return None, 0.0


def settle_crazy_7(c1: int, c2: int, c3: int | None, paytable: Mapping[str, float]) -> Outcome:
    """Crazy 7 on the player's first three cards (official rules).

    ``c3`` is the player's third card: the 3rd card of the shared hand (own hit/double or a crowd hit),
    or after splitting 7s the next card dealt to split hand 1; None if no third card was drawn (e.g. the
    dealer peeked a blackjack, or everybody stood). First card 7 → ``one``; first two 7s →
    ``pair_(un)suited``; first three 7s → ``three_(un)suited`` ("suited" = all the same suit).
    Only the highest tier pays.
    """
    if RANK_OF[c1] != SEVEN:
        return None, 0.0
    if RANK_OF[c2] != SEVEN:
        return "one_seven", paytable["one"]
    if c3 is not None and RANK_OF[c3] == SEVEN:
        if SUIT_OF[c1] == SUIT_OF[c2] == SUIT_OF[c3]:
            return "three_sevens_suited", paytable["three_suited"]
        return "three_sevens", paytable["three_unsuited"]
    if SUIT_OF[c1] == SUIT_OF[c2]:
        return "two_sevens_suited", paytable["pair_suited"]
    return "two_sevens", paytable["pair_unsuited"]


def crazy_7_needs_third(c1: int, c2: int) -> bool:
    """True if the Crazy 7 outcome depends on a third card (first two cards are both 7s)."""
    return RANK_OF[c1] == SEVEN and RANK_OF[c2] == SEVEN


def settle_bust_bonus(num_cards: int, dealer_bust: bool, player_blackjack: bool,
                      paytable: Mapping[str, float], push_on_player_blackjack: bool = True) -> Outcome:
    """Dealer busts with ``num_cards`` cards → paytable[str(n)] (the highest key covers "n or more").
    Official rule: if the player has a blackjack with the initial hand the stake is returned."""
    if player_blackjack and push_on_player_blackjack:
        return "push", 0.0
    if not dealer_bust:
        return None, 0.0
    keys = sorted(int(k) for k in paytable)
    n = min(num_cards, keys[-1])
    if n < keys[0]:
        return None, 0.0
    return f"bust_{num_cards}_cards", paytable[str(n)]
