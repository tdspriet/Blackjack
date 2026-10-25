"""NNAgent: the trained network playing ONE Blackjack (SPEC §5, §7) + checkpoint discovery.

>>> from onebj.ai.agent import NNAgent, list_checkpoints
>>> list_checkpoints()                      # [{name, path, created, metrics}, ...] newest first
>>> agent = NNAgent.load("nn_v1", cfg)      # name in checkpoints/ or a path to a .pt file
>>> bets, trace = agent.decide_bets(table.observation())
>>> agent.trace = False                     # headless simulation: skip trace building

Decisions
* **play** — greedy on the play head: argmax over the legal actions of Q(stand, hit, double, split,
  cashout) (EV per unit of the hand's original bet; the cash-out value is exact).
* **insurance** — the bet head's ``insurance`` output evaluated on the insurance-phase features (EV per
  unit insurance stake); insure when it exceeds ``BetConfig.insurance_threshold`` (0).
* **bets** — the bet head on pre-deal features predicts the EV of every bet. Main stake = fractional
  Kelly on the predicted edge, ``kelly_fraction × edge / main_variance × bankroll``, rounded to whole
  euros and clamped to ``[unit, unit × max_units]`` (and the table limits). A side bet is placed when its
  predicted EV exceeds ``side_threshold``; its stake is the side-bet Kelly stake clamped to
  ``[side_min, side_min × side_max_units]`` — i.e. the table minimum in practice (side bets have huge
  variance, Crazy 7 σ ≈ 11 per unit, so Kelly stakes are below €1 for any realistic bankroll).

All three run on one :class:`~onebj.ai.model.NumpyNet` forward pass (~10 µs). ``fast_bets`` /
``fast_insurance`` / ``fast_action`` read the table directly (used by :mod:`onebj.sim`), and give exactly
the same decisions as the Observation-based methods.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.base import Trace
from ..analytics import (
    bust_bonus_probs,
    crazy_7_probs,
    insurance_ev,
    main_edge_estimate,
    perfect_pairs_probs,
    side_bet_evs,
    twenty_one_plus_three_probs,
)
from ..cards import BUCKET_OF, full_bucket_counts
from ..config import GameConfig, load_config
from ..engine import Action, Bets, Observation, Table
from ..sidebets import SIDE_BETS
from ..strategy.basic import get_basic_strategy, hilo_action
from .features import KIND_BET, KIND_INSURANCE, KIND_PLAY, FeatureEncoder, FeatureSpec, cfg_half_size
from .model import BET_LABELS, PLAY_LABELS, NumpyNet
from .policy import legal_mask

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = "checkpoints"
MAX_EDGES = 400  # per consecutive layer pair in describe()
_ACT = {"stand": Action.STAND, "hit": Action.HIT, "double": Action.DOUBLE, "split": Action.SPLIT,
        "cashout": Action.CASHOUT}
_SIDE_IDX = {name: BET_LABELS.index(name) for name in SIDE_BETS}
I_MAIN = BET_LABELS.index("main")
I_INS = BET_LABELS.index("insurance")


@dataclass
class BetConfig:
    """Bet sizing knobs (also settable per agent: ``NNAgent.load(..., bet={"unit": 10})``)."""

    unit: float = 1.0  # minimum / base main stake in money
    max_units: float = 12.0  # spread cap: main stake <= unit * max_units
    kelly_fraction: float = 0.5  # fraction of the Kelly stake
    main_variance: float = 1.33  # variance of the main-bet return per unit (doubles/splits included)
    bankroll_mode: str = "fixed"  # "fixed": Kelly on `bankroll`; "live": on the current balance
    bankroll: float | None = None  # None -> config bankroll.start
    side_bets: bool = True
    side_threshold: float = 0.02  # place a side bet when predicted EV > threshold (margin for model error)
    side_kelly_fraction: float = 0.5
    side_max_units: float = 5.0  # side stake <= side_min * side_max_units
    insurance_threshold: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> BetConfig:
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown bet config keys {sorted(unknown)}")
        return cls(**d)


# ---------------------------------------------------------------------- checkpoint discovery
def resolve_dir(directory: str | Path = DEFAULT_DIR) -> Path:
    """Relative directories are tried against the CWD first, then the repository root."""
    p = Path(directory)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def resolve_checkpoint(path_or_name: str | Path, directory: str | Path = DEFAULT_DIR) -> Path:
    p = Path(path_or_name)
    if p.suffix == ".pt" and p.exists():
        return p
    if p.suffix == ".json" and p.with_suffix(".pt").exists():
        return p.with_suffix(".pt")
    cand = resolve_dir(directory) / f"{p.name.removesuffix('.pt')}.pt"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"checkpoint {path_or_name!r} not found (looked in {resolve_dir(directory)})")


def read_meta(pt_path: Path) -> dict[str, Any]:
    js = pt_path.with_suffix(".json")
    return json.loads(js.read_text()) if js.exists() else {}


def list_checkpoints(dir: str | Path = DEFAULT_DIR) -> list[dict[str, Any]]:
    """``[{name, path, created, metrics}]`` for every ``<name>.pt`` with a ``<name>.json`` sidecar, newest
    first. ``metrics`` is a flat dict of headline numbers (see docs/NN.md)."""
    d = resolve_dir(dir)
    if not d.exists():
        return []
    out = []
    for pt in d.glob("*.pt"):
        meta = read_meta(pt)
        if not meta:
            continue
        out.append({"name": meta.get("name", pt.stem), "path": str(pt.resolve()), "created": meta.get("created"),
                    "metrics": meta.get("metrics")})
    out.sort(key=lambda c: c.get("created") or "", reverse=True)
    return out


def load_training_log(path_or_name: str | Path, dir: str | Path = DEFAULT_DIR) -> list[dict[str, Any]]:
    """The training log ``[{step, loss, eval_ev, ...}]`` stored in ``<name>.json`` (key ``training_log``)."""
    return list(read_meta(resolve_checkpoint(path_or_name, dir)).get("training_log", []))


def _side_variances(cfg: GameConfig) -> dict[str, float]:
    """Variance of each side bet's return per unit stake at a fresh shoe (for Kelly sizing)."""
    nd = cfg.shoe.num_decks
    cards = [nd] * 52
    buckets = full_bucket_counts(nd)
    sb = cfg.side_bets
    out = {}
    for name, probs, table in (
            ("perfect_pairs", perfect_pairs_probs(cards), sb.perfect_pairs),
            ("twenty_one_plus_three", twenty_one_plus_three_probs(cards), sb.twenty_one_plus_three),
            ("crazy_7", crazy_7_probs(cards, cfg.rules.peek_on_ace), sb.crazy_7)):
        m = sum(p * table[k] for k, p in probs.items()) - (1 - sum(probs.values()))
        e2 = sum(p * table[k] ** 2 for k, p in probs.items()) + (1 - sum(probs.values()))
        out[name] = e2 - m * m
    bb = bust_bonus_probs(buckets, cfg.rules.dealer_hits_soft_17)["bust_by_cards"]
    keys = sorted(int(k) for k in sb.bust_bonus)
    pw = {n: p for n, p in bb.items() if min(n, keys[-1]) >= keys[0]}
    win = sum(pw.values())
    m = sum(p * sb.bust_bonus[str(min(n, keys[-1]))] for n, p in pw.items()) - (1 - win)
    e2 = sum(p * sb.bust_bonus[str(min(n, keys[-1]))] ** 2 for n, p in pw.items()) + (1 - win)
    out["bust_bonus"] = e2 - m * m
    return out


def _r(x: float, nd: int = 5) -> float:
    return round(float(x), nd)


# ---------------------------------------------------------------------- the agent
class NNAgent:
    """Neural-network agent implementing :class:`onebj.agents.base.Agent` (+ fast paths for onebj.sim)."""

    def __init__(self, net: NumpyNet, encoder: FeatureEncoder, cfg: GameConfig | None = None,
                 meta: dict[str, Any] | None = None, ckpt_name: str = "nn", path: str | None = None,
                 bet: BetConfig | dict | None = None, trace: bool = True) -> None:
        self.net = net
        self.enc = encoder
        self.cfg = cfg or load_config()
        self.meta = meta or {}
        self.ckpt_name = ckpt_name
        self.name = f"nn:{ckpt_name}"
        self.path = path
        self.bet_cfg = bet if isinstance(bet, BetConfig) else BetConfig.from_dict(bet)
        self.trace = trace
        self.half_size = cfg_half_size(self.cfg)
        self.bs = get_basic_strategy(self.cfg.rules, self.cfg.shoe.num_decks)
        self.side_var = _side_variances(self.cfg)
        self.allow_cashout = True  # False: never cash out (ablation); features still see the offer
        self.cashout_enabled = bool(self.cfg.rules.cashout)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path_or_name: str | Path, config: GameConfig | None = None, *, trace: bool = True,
             bet: BetConfig | dict | None = None, dir: str | Path = DEFAULT_DIR) -> NNAgent:
        """Load ``checkpoints/<name>.pt`` (or a path). ``config`` = the game config to play under."""
        import torch

        pt = resolve_checkpoint(path_or_name, dir)
        ck = torch.load(pt, map_location="cpu", weights_only=True)
        from .model import build_model
        model = build_model(ck["arch"])
        model.load_state_dict(ck["state_dict"])
        spec = FeatureSpec.from_dict(ck.get("feature_spec"))
        enc = FeatureEncoder(spec)
        names = list(ck.get("feature_names") or [])
        if names and names != enc.names:
            raise ValueError(f"checkpoint {pt.name} was trained with different features")
        meta = read_meta(pt)
        if bet is None and meta.get("bet_config"):
            bet = meta["bet_config"]
        agent = cls(NumpyNet(NumpyNet.export(model)), enc, config, meta, meta.get("name", pt.stem), str(pt.resolve()),
                    bet, trace)
        return agent

    def set_trace(self, on: bool) -> None:
        self.trace = bool(on)

    # ------------------------------------------------------------------ bet sizing
    def _bankroll(self, balance: float) -> float:
        b = self.bet_cfg
        if b.bankroll_mode == "live":
            return max(balance, 0.0)
        return float(b.bankroll if b.bankroll is not None else self.cfg.bankroll.start)

    def size_bets(self, pred: np.ndarray, balance: float) -> tuple[Bets, dict[str, Any]]:
        """Bets from the bet-head predictions (EV per unit stake, ``BET_LABELS`` order)."""
        b, lim = self.bet_cfg, self.cfg.limits
        bank = self._bankroll(balance)
        edge = float(pred[I_MAIN])
        kelly = b.kelly_fraction * edge / b.main_variance * bank if edge > 0 else 0.0
        hi = min(b.unit * b.max_units, lim.main_max)
        main = min(max(round(kelly), b.unit, lim.main_min), hi)
        bets = Bets(main=main)
        sides: dict[str, float] = {}
        if b.side_bets:
            for name, i in _SIDE_IDX.items():
                ev = float(pred[i])
                if ev > b.side_threshold:
                    k = b.side_kelly_fraction * ev / self.side_var[name] * bank
                    stake = min(max(math.floor(k), lim.side_min), lim.side_min * b.side_max_units, lim.side_max)
                    setattr(bets, name, stake)
                    sides[name] = round(k, 3)
        return bets, {"edge": _r(edge), "kelly_stake": round(kelly, 2), "bankroll": bank, "units": main / b.unit,
                      "side_kelly_stakes": sides}

    # ------------------------------------------------------------------ fast paths (onebj.sim)
    def fast_bets(self, table: Table) -> Bets:
        x = self.enc.encode_table(table, KIND_BET)
        return self.size_bets(self.net.bet(x), table.balance)[0]

    def fast_insurance(self, table: Table) -> bool:
        x = self.enc.encode_table(table, KIND_INSURANCE)
        return float(self.net.bet(x)[I_INS]) > self.bet_cfg.insurance_threshold

    def fast_action(self, table: Table, state: tuple) -> str:
        x = self.enc.encode_play_state(table, state)
        q = self.net.q(x)
        legal = legal_mask(state, self.cashout_enabled and self.allow_cashout)
        best, bi = -1e9, 0
        for i in range(5):
            if legal[i] and q[i] > best:
                best, bi = q[i], i
        return PLAY_LABELS[bi]

    # ------------------------------------------------------------------ Agent protocol
    def decide_bets(self, obs: Observation) -> tuple[Bets, Trace | None]:
        if not self.trace:
            x = self.enc.encode_obs(obs, KIND_BET, self.half_size)
            return self.size_bets(self.net.bet(x), obs.bankroll)[0], None
        x, raw = self.enc.encode_obs(obs, KIND_BET, self.half_size, raw=True)
        acts, _q, pred = self.net.full(x)
        bets, info = self.size_bets(pred, obs.bankroll)
        evs = side_bet_evs(obs.unseen_card_counts, self.cfg)
        analytic = {"main": _r(main_edge_estimate(obs.unseen_counts, self.cfg.rules, obs.num_decks)),
                    **{k: _r(v) for k, v in evs.items()},
                    "insurance": _r(insurance_ev(obs.unseen_counts, self.cfg.rules.insurance_payout))}
        tc = obs.true_counts.get("hilo", 0.0)
        hilo_units = _hilo_units(tc)
        trace: Trace = {
            "kind": "bet", "features": self._features(x, raw),
            "layers": self._layers(x, acts, "bet", pred),
            "outputs": {"labels": list(BET_LABELS), "values": [_r(v) for v in pred],
                        "legal": [True] * len(BET_LABELS), "unit": "EV per unit stake (predicted)"},
            "chosen": f"{bets.main:g}",
            "reference": {"basic_strategy": None, "hilo_deviation": None, "hilo_bet_units": str(hilo_units)},
            "extra": {"predicted_ev": {k: _r(v) for k, v in zip(BET_LABELS, pred)},
                      "bet_fraction": _r(bets.total / max(obs.bankroll, 1e-9)), "bets": bets.to_dict(),
                      "kelly": info, "analytic_ev": analytic, "true_count": round(tc, 2)},
        }
        return bets, trace

    def decide_insurance(self, obs: Observation) -> tuple[bool, Trace | None]:
        thr = self.bet_cfg.insurance_threshold
        if not self.trace:
            x = self.enc.encode_obs(obs, KIND_INSURANCE, self.half_size)
            return float(self.net.bet(x)[I_INS]) > thr, None
        x, raw = self.enc.encode_obs(obs, KIND_INSURANCE, self.half_size, raw=True)
        acts, _q, pred = self.net.full(x)
        ev = float(pred[I_INS])
        take = ev > thr
        tc = obs.true_counts.get("hilo", 0.0)
        trace: Trace = {
            "kind": "insurance", "features": self._features(x, raw),
            "layers": self._layers(x, acts, "bet", pred),
            "outputs": {"labels": ["insure", "no_insurance"], "values": [_r(ev, 4), 0.0], "legal": [True, True],
                        "unit": "EV per unit insurance stake (predicted)"},
            "chosen": "insure" if take else "no_insurance",
            "reference": {"basic_strategy": "no_insurance", "hilo_deviation": "insurance" if tc >= 3 else None},
            "extra": {"predicted_ev": {"insurance": _r(ev)},
                      "analytic_ev": {"insurance": _r(insurance_ev(obs.unseen_counts, self.cfg.rules.insurance_payout))},
                      "true_count": round(tc, 2)},
        }
        return take, trace

    def decide_action(self, obs: Observation) -> tuple[Action, Trace | None]:
        h = obs.hand
        legal_set = {a.value for a in obs.legal_actions}
        legal = [lab in legal_set for lab in PLAY_LABELS]
        if not self.allow_cashout:
            legal[4] = False
        if not self.trace:
            x = self.enc.encode_obs(obs, KIND_PLAY, self.half_size)
            q = self.net.q(x)
            return _ACT[_argmax(q, legal)], None
        x, raw = self.enc.encode_obs(obs, KIND_PLAY, self.half_size, raw=True)
        acts, q, _b = self.net.full(x)
        chosen = _argmax(q, legal)
        up = BUCKET_OF[obs.dealer_up]
        tc = obs.true_counts.get("hilo", 0.0)
        basic = self.bs.action(h.total, h.soft, h.num_cards, up, h.can_double, h.pair, h.can_split)
        _a, dev = hilo_action(self.bs, h.total, h.soft, h.num_cards, up, h.can_double, h.pair, h.can_split, tc)
        vals = [(_r(q[i], 4) if legal[i] else None) for i in range(5)]
        legal_q = sorted((float(q[i]) for i in range(5) if legal[i]), reverse=True)
        trace: Trace = {
            "kind": "play", "features": self._features(x, raw),
            "layers": self._layers(x, acts, "play", np.where(legal, q, 0.0)),
            "outputs": {"labels": list(PLAY_LABELS), "values": vals, "legal": legal,
                        "unit": "EV per unit of the hand's original bet (predicted; cashout exact)"},
            "chosen": chosen,
            "reference": {"basic_strategy": basic, "hilo_deviation": dev.name if dev else None},
            "extra": {"cashout_offer": h.cashout_offer, "q_margin": _r(legal_q[0] - legal_q[1], 4) if len(legal_q) > 1 else None,
                      "true_count": round(tc, 2)},
        }
        return _ACT[chosen], trace

    # ------------------------------------------------------------------ traces / topology
    def _features(self, x: np.ndarray, raw: list[Any]) -> list[dict[str, Any]]:
        return [{"name": n, "value": _r(v, 4), "raw": r} for n, v, r in zip(self.enc.names, x.tolist(), raw)]

    def _layers(self, x: np.ndarray, acts: list[np.ndarray], head: str, head_vals: Any) -> list[dict[str, Any]]:
        layers = [{"name": "input", "size": int(x.size), "activations": [_r(v, 4) for v in x.tolist()]}]
        for i, a in enumerate(acts):
            layers.append({"name": f"trunk.{i}", "size": int(a.size), "activations": [_r(v, 4) for v in a.tolist()]})
        hv = [_r(v, 4) for v in np.asarray(head_vals).tolist()]
        layers.append({"name": head, "size": len(hv), "activations": hv})
        return layers

    def describe(self) -> dict[str, Any]:
        """Static topology for the NN panel (SPEC §5): layers, heads and top-K weights per layer pair."""
        net = self.net
        layers: list[dict[str, Any]] = [{"name": "input", "size": self.enc.size, "labels": list(self.enc.names)}]
        names = ["input"]
        for i, W in enumerate(net.Ws):
            layers.append({"name": f"trunk.{i}", "size": int(W.shape[1])})
            names.append(f"trunk.{i}")
        last = names[-1]
        heads = [{"name": "play", "size": len(PLAY_LABELS), "labels": list(PLAY_LABELS), "from": last},
                 {"name": "bet", "size": len(BET_LABELS), "labels": list(BET_LABELS), "from": last}]
        edges: dict[str, list[list[float]]] = {}
        for i, W in enumerate(net.Ws):
            edges[f"{names[i]}->{names[i + 1]}"] = _top_edges(W, MAX_EDGES)
        edges[f"{last}->play"] = _top_edges(net.Wp, MAX_EDGES)  # cashout (index 4) is wired, no weights
        edges[f"{last}->bet"] = _top_edges(net.Wb * net.scale[None, :], MAX_EDGES)
        m = self.meta
        training = {k: m.get(k) for k in ("created", "config_hash", "metrics", "data", "timing", "train_config",
                                           "gen_config", "description") if k in m}
        training["cashout_output"] = "wired to input 'cashout_value' (exact EV of cashing out)"
        training["bet_config"] = asdict(self.bet_cfg)
        return {"name": self.ckpt_name, "checkpoint": self.path, "layers": layers, "heads": heads, "edges": edges,
                "training": training}

    # ------------------------------------------------------------------ helpers for analysis code
    def q_values_obs(self, obs: Observation) -> np.ndarray:
        return self.net.q(self.enc.encode_obs(obs, KIND_PLAY, self.half_size))

    def bet_predictions_table(self, table: Table) -> np.ndarray:
        return self.net.bet(self.enc.encode_table(table, KIND_BET))


def _hilo_units(tc: float) -> int:
    from ..agents.basic import DEFAULT_SPREAD
    f = math.floor(tc)
    for min_tc, units in sorted(DEFAULT_SPREAD, reverse=True):
        if f >= min_tc:
            return units
    return 1


def _argmax(q: np.ndarray, legal: list[bool]) -> str:
    best, bi = -1e9, 0
    for i in range(len(legal)):
        if legal[i] and q[i] > best:
            best, bi = q[i], i
    return PLAY_LABELS[bi]


def _top_edges(W: np.ndarray, k: int) -> list[list[float]]:
    """Top-``k`` |weights| of an (in, out) matrix as ``[[i, j, w], ...]``."""
    flat = np.abs(W).ravel()
    k = min(k, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    n_out = W.shape[1]
    return [[int(t // n_out), int(t % n_out), round(float(W.flat[t]), 4)] for t in idx]


def now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


__all__ = [
    "BET_LABELS",
    "PLAY_LABELS",
    "BetConfig",
    "NNAgent",
    "list_checkpoints",
    "load_training_log",
    "resolve_checkpoint",
]
