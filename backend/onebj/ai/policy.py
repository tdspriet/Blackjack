"""Light-weight play policies operating directly on a :class:`~onebj.engine.Table` (fast paths).

Used by the data generator (behaviour + rollout continuation policies), by the evaluators and by
:class:`onebj.ai.agent.NNAgent`'s fast path. Only public state is read (``decision_state``, the cash-out
offer, seen-card counts) — never the hole card or the shoe order.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ..engine import Action, Phase, Table
from .features import KIND_PLAY, FeatureEncoder
from .model import PLAY_LABELS, NumpyNet

ACTIONS: tuple[Action, ...] = (Action.STAND, Action.HIT, Action.DOUBLE, Action.SPLIT, Action.CASHOUT)
_BASIC = {"stand": Action.STAND, "hit": Action.HIT, "double": Action.DOUBLE, "split": Action.SPLIT}


class Policy(Protocol):
    def action(self, table: Table) -> Action: ...


def legal_mask(state: tuple, cashout: bool) -> list[bool]:
    """Legality of [stand, hit, double, split, cashout] from ``table.decision_state()``."""
    return [True, True, bool(state[4]), bool(state[6]), cashout]


class BasicPolicy:
    """Basic strategy for the table's rules (composition independent, never cashes out)."""

    def action(self, table: Table) -> Action:
        total, soft, n, up, can_double, pair, can_split, _ = table.decision_state()
        return _BASIC[table.strategy.action(total, soft, n, up, can_double, pair, can_split)]


class NetPolicy:
    """Greedy w.r.t. the network's play head (masked argmax over legal actions, ties → first label)."""

    def __init__(self, net: NumpyNet, encoder: FeatureEncoder) -> None:
        self.net = net
        self.enc = encoder

    def q_values(self, table: Table, state: tuple | None = None) -> tuple[np.ndarray, list[bool]]:
        state = state or table.decision_state()
        x = self.enc.encode_play_state(table, state)
        q = self.net.q(x)
        return q, legal_mask(state, table.cashout_table is not None)

    def action(self, table: Table, state: tuple | None = None) -> Action:
        q, legal = self.q_values(table, state)
        best, bi = -1e9, 0
        for i in range(5):
            if legal[i] and q[i] > best:
                best, bi = q[i], i
        return ACTIONS[bi]


def play_out(table: Table, policy: Any) -> None:
    """Finish the pending player decisions of the round with ``policy`` (insurance declined)."""
    while True:
        ph = table.phase
        if ph is Phase.PLAYER:
            table.act(policy.action(table))
        elif ph is Phase.INSURANCE:
            table.act(Action.NO_INSURANCE)
        else:
            return


def argmax_label(q: np.ndarray, legal: list[bool]) -> str:
    best, bi = -1e9, 0
    for i in range(len(legal)):
        if legal[i] and q[i] > best:
            best, bi = q[i], i
    return PLAY_LABELS[bi]


__all__ = [
    "ACTIONS",
    "KIND_PLAY",
    "BasicPolicy",
    "NetPolicy",
    "Policy",
    "argmax_label",
    "legal_mask",
    "play_out",
]
