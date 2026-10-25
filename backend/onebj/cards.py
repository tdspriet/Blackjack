"""Cards as ints 0..51: ``card = rank * 4 + suit``.

* rank index 0..12 = A, 2, ..., 10, J, Q, K
* suit index 0..3 = hearts, diamonds, clubs, spades (hearts/diamonds are red)
* rank bucket 0..9 = A, 2, ..., 9, T (10/J/Q/K) — what matters for blackjack totals and counting
* value = bucket + 1 (ace counts 1; soft totals are handled in :mod:`onebj.hand`)
"""

from __future__ import annotations

from typing import Any

RANKS: tuple[str, ...] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS: tuple[str, ...] = ("hearts", "diamonds", "clubs", "spades")
BUCKETS: tuple[str, ...] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "T")
NUM_CARDS = 52
ACE = 0  # bucket / rank index of the ace
TEN = 9  # bucket index of 10-value cards

RANK_OF: tuple[int, ...] = tuple(c // 4 for c in range(NUM_CARDS))
SUIT_OF: tuple[int, ...] = tuple(c % 4 for c in range(NUM_CARDS))
BUCKET_OF: tuple[int, ...] = tuple(min(c // 4, 9) for c in range(NUM_CARDS))
VALUE_OF: tuple[int, ...] = tuple(min(c // 4, 9) + 1 for c in range(NUM_CARDS))
IS_RED: tuple[bool, ...] = tuple(c % 4 < 2 for c in range(NUM_CARDS))
RANK_TO_BUCKET: tuple[int, ...] = tuple(min(r, 9) for r in range(13))
#: number of ranks in each bucket (T = 10, J, Q, K)
RANKS_PER_BUCKET: tuple[int, ...] = (1,) * 9 + (4,)


def make_card(rank: int | str, suit: int | str) -> int:
    """Card int from rank index/label and suit index/name."""
    r = RANKS.index(rank) if isinstance(rank, str) else rank
    s = SUITS.index(suit) if isinstance(suit, str) else suit
    if not (0 <= r < 13 and 0 <= s < 4):
        raise ValueError(f"invalid card rank={rank!r} suit={suit!r}")
    return r * 4 + s


def card_code(card: int) -> str:
    """Image code matching ``cards/<code>.png``, e.g. ``10_of_hearts``, ``K_of_spades``."""
    return f"{RANKS[card // 4]}_of_{SUITS[card % 4]}"


CODES: tuple[str, ...] = tuple(card_code(c) for c in range(NUM_CARDS))
_SHORT_SUITS = {"h": 0, "d": 1, "c": 2, "s": 3}


def parse_card(text: str) -> int:
    """Parse ``"A_of_hearts"``, ``"10_of_clubs"`` or short forms like ``"AH"``, ``"10c"``, ``"Ts"``."""
    if "_of_" in text:
        rank, suit = text.split("_of_")
        return make_card(rank, suit)
    rank, suit = text[:-1].upper(), text[-1].lower()
    if rank == "T":
        rank = "10"
    return make_card(rank, _SHORT_SUITS[suit])


def cards_from(text: str) -> list[int]:
    """Parse a space separated list of short card codes, e.g. ``"AH Ks 7d"``."""
    return [parse_card(t) for t in text.split()]


def card_json(card: int, owner: str, face_up: bool = True) -> dict[str, Any]:
    """Card JSON per SPEC §6 (face-down cards only reveal ``code: "back"``)."""
    if not face_up:
        return {"face_up": False, "code": "back", "owner": owner}
    return {"rank": RANKS[card // 4], "suit": SUITS[card % 4], "code": CODES[card],
            "face_up": True, "owner": owner}


def full_bucket_counts(num_decks: int) -> list[int]:
    """Bucket counts (A, 2..9, T) of a full ``num_decks`` shoe."""
    return [4 * num_decks] * 9 + [16 * num_decks]
