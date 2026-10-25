"""Card-by-card streaming of a table's state.

The engine runs a whole decision (deal, or player action → crowd → dealer → settlement) synchronously.
To animate it, :class:`StreamingTable` notifies a listener on every engine event *at the moment it is
recorded*, and :class:`FrameRecorder` captures a ``{"game", "shoe"}`` snapshot for every visible change
(round start, each card dealt, hole card revealed, split). Intermediate frames have no legal actions
and no cash-out offers (nothing can be decided while cards are moving) and report the transient phase
(``crowd`` / ``dealer``) from the engine's phase events. The final state is always a full snapshot.

The initial deal is special: the engine draws all four cards before recording their events, so those
frames are trimmed to the cards dealt so far.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

from ..cards import BUCKET_OF
from ..engine import Phase, Table
from ..hand import hand_value

Frame = dict[str, Any]  # {"game": ..., "shoe": ...}


class _EventList(list):
    """List whose ``append`` also notifies the owning table's listener."""

    __slots__ = ("_owner",)

    def __init__(self, items: Any, owner: StreamingTable) -> None:
        super().__init__(items)
        self._owner = owner

    def append(self, ev: Any) -> None:  # type: ignore[override]
        super().append(ev)
        cb = self._owner.listener
        if cb is not None:
            cb(ev)


class StreamingTable(Table):
    """A :class:`Table` that reports each event to ``listener`` as it happens (clones are plain tables)."""

    listener: Callable[[dict[str, Any]], None] | None = None

    @property  # type: ignore[override]
    def events(self) -> list[dict[str, Any]]:
        return self._events

    @events.setter
    def events(self, value: list[dict[str, Any]]) -> None:
        self._events = _EventList(value, self)


class FrameRecorder:
    """Context manager collecting intermediate frames while the engine runs.

    >>> with FrameRecorder(table) as rec:
    ...     table.act("stand")
    >>> rec.frames   # list of {"game", "shoe"} (without the final state)
    """

    def __init__(self, table: StreamingTable, analytics: bool = True) -> None:
        self.table = table
        self.analytics = analytics
        self.frames: list[Frame] = []
        self.phase = table.phase.value
        self._deal: list[tuple[str, int]] | None = None  # initial deal (owner, card) in event order

    def __enter__(self) -> Self:
        self.table.listener = self._on_event
        return self

    def __exit__(self, *exc: object) -> None:
        self.table.listener = None

    # ------------------------------------------------------------------ capture
    def _on_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("type")
        if kind == "round_start":
            self.phase = Phase.PLAYER.value
            self._deal = []
            self._capture()
        elif kind == "phase":
            self.phase = ev["phase"]
            self._deal = None
        elif kind == "card":
            if self._deal is not None and len(self._deal) < 4:
                self._deal.append((ev["owner"], ev["card"]))
                self._capture(trim=self._deal)
                if len(self._deal) == 4:
                    self._deal = None
            else:
                self._capture()
        elif kind in ("reveal", "split"):
            self._capture()

    def _capture(self, trim: list[tuple[str, int]] | None = None) -> None:
        try:
            snap = self.table.snapshot(analytics=self.analytics)
        except Exception:  # noqa: BLE001 - a snapshot mid-mutation must never break the round
            return
        g = snap["game"]
        g["phase"] = self.phase
        g["legal_actions"] = []
        g["round_net"] = None
        for h in g["hands"]:
            h["cashout_offer"] = None
        if trim is not None:
            _trim_initial_deal(g, trim)
        self.frames.append(snap)


def _trim_initial_deal(g: dict[str, Any], dealt: list[tuple[str, int]]) -> None:
    """Show only the initial-deal cards dealt so far (P1, dealer up, P2, hole)."""
    player = [c for o, c in dealt if o == "player"]
    n_dealer = sum(1 for o, _ in dealt if o == "dealer")
    if g["hands"]:
        h = g["hands"][0]
        h["cards"] = h["cards"][:len(player)]
        h["total"], h["soft"] = hand_value(player) if player else (0, False)
        h["status"] = "active"
    d = g["dealer"]
    d["cards"] = d["cards"][:n_dealer]
    if n_dealer == 0:
        d.update(total=0, soft=False)
    else:
        up = next(c for o, c in dealt if o == "dealer")
        ace = BUCKET_OF[up] == 0
        d.update(total=11 if ace else BUCKET_OF[up] + 1, soft=ace)
    d["blackjack"] = False
    d["bust"] = False
    g["side_bet_results"] = []


def visual_signature(game: dict[str, Any]) -> tuple:
    """What a frame shows on the felt (to drop frames identical to the final state)."""
    return (
        tuple(len(h["cards"]) for h in game["hands"]),
        tuple(c.get("code") for c in game["dealer"]["cards"]),
        len(game["crowd"]["cards"]),
    )
