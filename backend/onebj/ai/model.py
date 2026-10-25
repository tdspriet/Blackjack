"""The network: one MLP trunk, two heads (SPEC §7).

::

    input (61 features) ─ trunk.0 (128, ReLU) ─ trunk.1 (64, ReLU) ─┬─ play: Q(stand, hit, double, split) + cashout
                                                                    └─ bet:  EV(main, PP, 21+3, Crazy 7, Bust Bonus, insurance)

* **play** — action values in EV per unit of the hand's original bet. The ``cashout`` output is not
  learned: it is wired straight to the ``cashout_value`` input (= offer / bet − 1, known exactly), so the
  network only has to learn when holding beats it. Illegal actions are masked by the agent.
* **bet** — EV per unit stake of each bet for the coming round (main under the learned play policy); the
  ``insurance`` output is read in the insurance phase (EV per unit insurance stake). Outputs are
  de-standardised with fixed ``bet_offset`` / ``bet_scale`` buffers (set from the training targets).

:class:`NumpyNet` mirrors the forward pass in NumPy for fast batch-1 inference (~10 µs) inside the
agent and the multiprocessing data generators; it also returns every layer's activations for traces.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

PLAY_LABELS: list[str] = ["stand", "hit", "double", "split", "cashout"]
BET_LABELS: list[str] = ["main", "perfect_pairs", "twenty_one_plus_three", "crazy_7", "bust_bonus", "insurance"]
N_LEARNED_PLAY = 4  # stand, hit, double, split (cashout is wired to the input)


class BlackjackNet(nn.Module):
    def __init__(self, n_in: int, cash_index: int, trunk: Sequence[int] = (128, 64)) -> None:
        super().__init__()
        self.n_in = n_in
        self.cash_index = cash_index
        self.trunk_sizes = tuple(int(h) for h in trunk)
        layers = []
        prev = n_in
        for h in self.trunk_sizes:
            layers.append(nn.Linear(prev, h))
            prev = h
        self.trunk = nn.ModuleList(layers)
        self.play = nn.Linear(prev, N_LEARNED_PLAY)
        self.bet = nn.Linear(prev, len(BET_LABELS))
        self.register_buffer("bet_offset", torch.zeros(len(BET_LABELS)))
        self.register_buffer("bet_scale", torch.ones(len(BET_LABELS)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for lin in self.trunk:
            h = torch.relu(lin(h))
        q = self.play(h)
        q = torch.cat([q, x[:, self.cash_index:self.cash_index + 1]], dim=1)
        b = self.bet(h) * self.bet_scale + self.bet_offset
        return q, b

    def arch(self) -> dict[str, Any]:
        return {"n_in": self.n_in, "cash_index": self.cash_index, "trunk": list(self.trunk_sizes)}


def build_model(arch: dict[str, Any]) -> BlackjackNet:
    return BlackjackNet(int(arch["n_in"]), int(arch["cash_index"]), arch.get("trunk", (128, 64)))


class NumpyNet:
    """NumPy mirror of :class:`BlackjackNet` (float32). ``weights`` = :meth:`export` dict."""

    def __init__(self, weights: dict[str, Any]) -> None:
        self.Ws = [np.asarray(w, dtype=np.float32) for w in weights["trunk_w"]]  # (in, out)
        self.bs = [np.asarray(b, dtype=np.float32) for b in weights["trunk_b"]]
        self.Wp = np.asarray(weights["play_w"], dtype=np.float32)
        self.bp = np.asarray(weights["play_b"], dtype=np.float32)
        self.Wb = np.asarray(weights["bet_w"], dtype=np.float32)
        self.bb = np.asarray(weights["bet_b"], dtype=np.float32)
        self.off = np.asarray(weights["bet_offset"], dtype=np.float32)
        self.scale = np.asarray(weights["bet_scale"], dtype=np.float32)
        self.cash_index = int(weights["cash_index"])

    @staticmethod
    def export(model: BlackjackNet) -> dict[str, Any]:
        """Plain-NumPy weights (picklable, for worker processes)."""
        sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
        n = len(model.trunk)
        return {"trunk_w": [sd[f"trunk.{i}.weight"].T.copy() for i in range(n)],
                "trunk_b": [sd[f"trunk.{i}.bias"] for i in range(n)],
                "play_w": sd["play.weight"].T.copy(), "play_b": sd["play.bias"],
                "bet_w": sd["bet.weight"].T.copy(), "bet_b": sd["bet.bias"],
                "bet_offset": sd["bet_offset"], "bet_scale": sd["bet_scale"], "cash_index": model.cash_index}

    def trunk(self, x: np.ndarray) -> list[np.ndarray]:
        acts = []
        h = x
        for W, b in zip(self.Ws, self.bs):
            h = h @ W + b
            np.maximum(h, 0.0, out=h)
            acts.append(h)
        return acts

    def q(self, x: np.ndarray) -> np.ndarray:
        """Play Q-values (5,) or (B, 5) incl. the wired cash-out value."""
        h = x
        for W, b in zip(self.Ws, self.bs):
            h = h @ W + b
            np.maximum(h, 0.0, out=h)
        q = h @ self.Wp + self.bp
        if x.ndim == 1:
            return np.concatenate([q, x[self.cash_index:self.cash_index + 1]])
        return np.concatenate([q, x[:, self.cash_index:self.cash_index + 1]], axis=1)

    def bet(self, x: np.ndarray) -> np.ndarray:
        """Bet-head EVs (6,) or (B, 6)."""
        h = x
        for W, b in zip(self.Ws, self.bs):
            h = h @ W + b
            np.maximum(h, 0.0, out=h)
        return (h @ self.Wb + self.bb) * self.scale + self.off

    def full(self, x: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """(trunk activations, play Q (5), bet EVs (6)) for one input vector — used for traces."""
        acts = self.trunk(x)
        h = acts[-1]
        q = np.concatenate([h @ self.Wp + self.bp, x[self.cash_index:self.cash_index + 1]])
        b = (h @ self.Wb + self.bb) * self.scale + self.off
        return acts, q, b
