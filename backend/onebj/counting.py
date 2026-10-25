"""Card counting systems: Hi-Lo, KO, Omega II, Wong Halves, Zen Count.

All tags depend only on the rank bucket (A, 2..9, T), so running counts are a dot product of the seen
bucket counts with the tag vector. True counts divide by the decks *unseen in the whole 8-deck shoe*
(``cards_unseen / 52``): the Napoleon two-halves procedure must not reset the count at the half switch.

KO is unbalanced (+4 per deck): its running count starts at the standard IRC ``4 - 4 * decks``; its
"true" count is reported in balanced form, ``(rc - IRC - 4 * cards_seen / 52) / decks_unseen``.

:class:`CountTracker` consumes engine events (face-up cards, the hole card on reveal, reshuffles) for
clients that only see the event stream; :func:`counts_from_seen` derives the same numbers directly from
seen bucket counts (what the engine does).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .cards import BUCKET_OF

#: tags per bucket (A, 2, 3, 4, 5, 6, 7, 8, 9, T)
SYSTEMS: dict[str, tuple[float, ...]] = {
    "hilo": (-1, 1, 1, 1, 1, 1, 0, 0, 0, -1),
    "ko": (-1, 1, 1, 1, 1, 1, 1, 0, 0, -1),
    "omega2": (0, 1, 1, 2, 2, 2, 1, 0, -1, -2),
    "wong_halves": (-1, 0.5, 1, 1, 1.5, 1, 0.5, 0, -0.5, -1),
    "zen": (-1, 1, 1, 2, 2, 2, 1, 0, 0, -2),
}
SYSTEM_NAMES: tuple[str, ...] = tuple(SYSTEMS)
HILO_TAGS = SYSTEMS["hilo"]


def ko_irc(num_decks: int) -> int:
    """Standard KO initial running count (key count +4 convention)."""
    return 4 - 4 * num_decks


def counts_from_seen(seen_buckets: Sequence[int], num_decks: int = 8) -> tuple[dict[str, float], dict[str, float]]:
    """(running, true) counts for all systems from the seen bucket counts of the current shoe."""
    seen = sum(seen_buckets)
    unseen = 52 * num_decks - seen
    decks = max(unseen, 1) / 52
    running: dict[str, float] = {}
    true: dict[str, float] = {}
    for name, tags in SYSTEMS.items():
        rc = sum(t * n for t, n in zip(tags, seen_buckets))
        if name == "ko":
            irc = ko_irc(num_decks)
            running[name] = rc + irc
            true[name] = (rc - 4 * seen / 52) / decks
        else:
            running[name] = rc
            true[name] = rc / decks
    return running, true


def hilo_true(seen_buckets: Sequence[int], num_decks: int = 8) -> float:
    """Fast Hi-Lo true count (hot path for agents)."""
    s = seen_buckets
    rc = s[1] + s[2] + s[3] + s[4] + s[5] - s[0] - s[9]
    unseen = 52 * num_decks - sum(s)
    return rc * 52 / max(unseen, 1)


class CountTracker:
    """Event-driven counter with a per-round history (reset on reshuffle, not on half switch)."""

    def __init__(self, num_decks: int = 8) -> None:
        self.num_decks = num_decks
        self.seen = [0] * 10
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.seen = [0] * 10
        self.history = []

    def see(self, card: int) -> None:
        self.seen[BUCKET_OF[card]] += 1

    def apply_event(self, ev: Mapping[str, Any]) -> None:
        kind = ev.get("type")
        if kind == "card" and ev.get("face_up", True):
            self.see(ev["card"])
        elif kind == "reveal":
            self.see(ev["card"])
        elif kind == "shuffle" and ev.get("kind") == "reshuffle":
            self.reset()

    def apply_events(self, events: Iterable[Mapping[str, Any]]) -> None:
        for ev in events:
            self.apply_event(ev)

    def running(self) -> dict[str, float]:
        return counts_from_seen(self.seen, self.num_decks)[0]

    def true(self) -> dict[str, float]:
        return counts_from_seen(self.seen, self.num_decks)[1]

    def end_round(self, round_id: int) -> None:
        rc, tc = counts_from_seen(self.seen, self.num_decks)
        self.history.append({"round": round_id, "rc": rc["hilo"], "tc": round(tc["hilo"], 3)})
