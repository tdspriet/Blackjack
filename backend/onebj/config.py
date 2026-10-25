"""Game configuration (SPEC §3): dataclasses, JSON loading, deep-merge patches.

``load_config()`` reads ``config/default.json``; ``apply_patch(cfg, patch)`` returns a new config with a
(partial, nested) patch deep-merged in, as used by the server's ``set_config`` command.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.json"


@dataclass
class ShoeConfig:
    num_decks: int = 8
    split_halves: bool = True
    penetration: float = 0.75
    cut_jitter: int = 8
    burn_cards: int = 1


@dataclass
class RulesConfig:
    blackjack_payout: float = 1.5
    dealer_hits_soft_17: bool = False
    peek_on_ace: bool = True
    peek_on_ten: bool = False
    late_dealer_bj_takes: str = "all"  # "all" | "original"
    double_any_two: bool = True  # False: double on hard 9-11 only
    double_after_split: bool = True
    max_splits: int = 1
    split_by_value: bool = True
    split_aces_one_card: bool = True
    charlie_cards: int = 6  # 0 disables Six Card Charlie
    charlie_beats_dealer_bj: bool = True
    insurance: bool = True
    insurance_payout: float = 2.0
    surrender: bool = False  # not supported (ONE Blackjack has no surrender)
    cashout: bool = True  # "Uitbetalen": cash a main hand out for cashout_rtp x its fresh-shoe fair value
    cashout_rtp: float = 0.98


@dataclass
class LimitsConfig:
    main_min: float = 1
    main_max: float = 5000
    side_min: float = 1
    side_max: float = 500
    chips: list[float] = field(default_factory=lambda: [1, 2, 5, 10, 25, 100, 500, 1000])


@dataclass
class SideBetsConfig:
    perfect_pairs: dict[str, float] = field(
        default_factory=lambda: {"mixed": 6, "coloured": 12, "perfect": 25})
    twenty_one_plus_three: dict[str, float] = field(
        default_factory=lambda: {"flush": 5, "straight": 10, "three_of_a_kind": 30,
                                 "straight_flush": 40, "suited_trips": 100})
    crazy_7: dict[str, float] = field(
        default_factory=lambda: {"one": 5, "pair_unsuited": 25, "pair_suited": 150,
                                 "three_unsuited": 500, "three_suited": 2000})
    bust_bonus: dict[str, float] = field(
        default_factory=lambda: {"3": 1, "4": 2, "5": 9, "6": 50, "7": 100, "8": 250})
    bust_bonus_push_on_player_blackjack: bool = True  # official: stake returned on a player blackjack


@dataclass
class CrowdConfig:
    enabled: bool = True
    size: int = 250
    min_hitters: int = 1
    mix: dict[str, float] = field(
        default_factory=lambda: {"basic": 0.75, "cautious": 0.12, "dealer_mimic": 0.10, "reckless": 0.03})


@dataclass
class BankrollConfig:
    start: float = 1000


@dataclass
class GameConfig:
    shoe: ShoeConfig = field(default_factory=ShoeConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    side_bets: SideBetsConfig = field(default_factory=SideBetsConfig)
    crowd: CrowdConfig = field(default_factory=CrowdConfig)
    bankroll: BankrollConfig = field(default_factory=BankrollConfig)


def _build(cls: type, data: Mapping[str, Any], path: str) -> Any:
    """Recursively build dataclass ``cls`` from ``data``; unknown keys raise ValueError."""
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key '{path}{key}'")
        default = getattr(cls(), key)
        if is_dataclass(default):
            if not isinstance(value, Mapping):
                raise ValueError(f"config key '{path}{key}' must be an object")
            kwargs[key] = _build(type(default), value, f"{path}{key}.")
        else:
            kwargs[key] = copy.deepcopy(value)
    return cls(**kwargs)


def from_dict(data: Mapping[str, Any]) -> GameConfig:
    """Build and validate a :class:`GameConfig` from a (possibly partial) dict."""
    cfg = _build(GameConfig, data, "")
    validate(cfg)
    return cfg


def to_dict(cfg: GameConfig) -> dict[str, Any]:
    """JSON-able dict of the config (same shape as ``config/default.json``)."""
    return asdict(cfg)


def deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``patch`` merged in recursively (dicts merged, everything else replaced)."""
    out = copy.deepcopy(dict(base))
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_patch(cfg: GameConfig, patch: Mapping[str, Any]) -> GameConfig:
    """New config = ``cfg`` deep-merged with ``patch`` (e.g. ``{"crowd": {"enabled": False}}``)."""
    return from_dict(deep_merge(to_dict(cfg), patch))


def load_config(path: str | Path | None = None, patch: Mapping[str, Any] | None = None) -> GameConfig:
    """Load the config from ``path`` (default ``config/default.json``), optionally applying ``patch``."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = json.loads(p.read_text()) if p.exists() else {}
    if patch:
        data = deep_merge(data, patch)
    return from_dict(data)


def validate(cfg: GameConfig) -> None:
    """Raise ValueError on inconsistent / unsupported settings."""
    s, r, c = cfg.shoe, cfg.rules, cfg.crowd
    if s.num_decks < 1:
        raise ValueError("shoe.num_decks must be >= 1")
    if s.split_halves and s.num_decks % 2:
        raise ValueError("shoe.split_halves requires an even number of decks")
    if not 0.1 <= s.penetration <= 0.95:
        raise ValueError("shoe.penetration must be within [0.1, 0.95]")
    if s.cut_jitter < 0 or s.burn_cards < 0:
        raise ValueError("shoe.cut_jitter / burn_cards must be >= 0")
    if r.late_dealer_bj_takes not in ("all", "original"):
        raise ValueError("rules.late_dealer_bj_takes must be 'all' or 'original'")
    if r.surrender:
        raise ValueError("rules.surrender is not supported (ONE Blackjack has no surrender)")
    if r.max_splits < 0:
        raise ValueError("rules.max_splits must be >= 0")
    if r.charlie_cards and r.charlie_cards < 3:
        raise ValueError("rules.charlie_cards must be 0 (off) or >= 3")
    if not 0 < r.cashout_rtp <= 1:
        raise ValueError("rules.cashout_rtp must be within (0, 1]")
    if any(not str(k).isdigit() for k in cfg.side_bets.bust_bonus):
        raise ValueError("side_bets.bust_bonus keys must be card counts ('3'..'8')")
    unknown = set(c.mix) - {"basic", "cautious", "dealer_mimic", "reckless"}
    if unknown:
        raise ValueError(f"unknown crowd types {sorted(unknown)}")
    if c.size < 0 or c.min_hitters < 1:
        raise ValueError("crowd.size must be >= 0 and crowd.min_hitters >= 1")
