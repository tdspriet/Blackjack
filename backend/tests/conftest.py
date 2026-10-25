"""Shared test helpers: rigging the shoe with a known card sequence."""

from __future__ import annotations

import pytest

from onebj.cards import cards_from
from onebj.config import GameConfig, load_config
from onebj.engine import Table


def stack(table: Table, spec: str) -> None:
    """Make the next cards drawn from the shoe be ``spec`` (e.g. ``"Ts 6h 9c 7d"``) by swapping cards
    in the shoe, so its card multiset is unchanged."""
    shoe = table.shoe
    want = cards_from(spec)
    for k, c in enumerate(want):
        i = shoe.pos + k
        j = next(j for j in range(i, shoe.total_cards) if shoe.cards[j] == c)
        shoe.cards[i], shoe.cards[j] = shoe.cards[j], shoe.cards[i]


def make_table(patch: dict | None = None, seed: int = 7, **kw) -> Table:
    base = {"crowd": {"enabled": False}}
    cfg = load_config(patch=_merge(base, patch or {}))
    return Table(cfg, seed=seed, **kw)


def _merge(a: dict, b: dict) -> dict:
    from onebj.config import deep_merge
    return deep_merge(a, b)


@pytest.fixture
def cfg() -> GameConfig:
    return load_config()
