"""Training-data generation: all-action Monte-Carlo targets from resampled table clones.

A worker plays rounds on a real :class:`~onebj.engine.Table` with a *behaviour* policy (current policy +
ε-random actions) and labels the states it visits:

* **play states** (every player decision), per ``GenConfig.targets``:

  - ``"solver"`` (default): EV of each legal action from the cash-aware composition solver
    (:mod:`onebj.ai.oracle`) at the current unseen composition — noise-free, ~0.1 ms;
  - ``"mc"``: the all-action Monte-Carlo returns described next;
  - ``"blend"``: ``solver + mc_weight × (mc − solver)``.

  Monte Carlo: for ``k_play`` independent ``table.clone(resample_unseen=True)``
  copies — every unseen card, incl. the hole card, redistributed uniformly, so nothing leaks from the true
  shoe order — *every* legal action is applied to a plain copy of the same resampled clone (common random
  numbers: the actions face the same future cards) and the round is finished with the *continuation*
  policy. Target = mean return of the hand (and its split children) per unit of its original bet.
  Cash-out needs no rollout: its value is exact and wired into the network.
* **bet states** (before a deal): ``k_bet`` resampled clones play a full round with the continuation
  policy → realised main-bet return (per unit; insurance declined) and realised Bust Bonus return. The
  Perfect Pairs, 21+3 and Crazy 7 returns — decided by the first five cards — come from a *deal Monte
  Carlo*: random permutations of the unseen pool are cut into 5-card deals (P1, up, P2, hole, next card)
  and settled with the engine's own ``settle_*`` functions (tabulated once). Each deal is a realised
  outcome; averaging ~``deal_groups × unseen/5`` of them per state cuts the target variance by two to
  three orders of magnitude (Crazy 7 has σ ≈ 11 per unit!) without changing its expectation.
* **insurance states** (dealer ace up): ``k_insurance`` resampled hole cards → realised insurance return.

State coverage: at the start of each shoe, with probability ``p_tilt``, the undealt cards are reordered
by a *tilted shuffle* (sort key = uniform + α·direction[rank]: the Hi-Lo direction, a random direction or
the sevens) so that rich/poor shoes — and 7-rich shoes for Crazy 7 — are common. Because every target is
computed on resampled clones, the tilt changes only *which* states are visited, never their targets.
ε-random actions reach rare hand states (5-card hands, odd soft totals, splits).
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..cards import BUCKET_OF, NUM_CARDS, RANK_OF
from ..config import GameConfig, from_dict
from ..engine import Action, Bets, Phase, Table
from ..sidebets import payout, settle_21_3, settle_crazy_7, settle_perfect_pairs
from .features import HILO_TAGS, KIND_BET, KIND_INSURANCE, KIND_PLAY, FeatureEncoder, FeatureSpec
from .model import NumpyNet
from .policy import BasicPolicy, NetPolicy

PLAY_IDX = {Action.STAND: 0, Action.HIT: 1, Action.DOUBLE: 2, Action.SPLIT: 3}
SEVEN_RANK = 6


@dataclass
class GenConfig:
    """Knobs of the data generator (see module docstring)."""

    targets: str = "mc"  # play-head targets: "mc" | "solver" | "blend" (see module docstring)
    mc_weight: float = 0.25  # "blend": target = solver + mc_weight * (mc - solver)
    k_play: int = 2  # resampled continuations per play state (all legal actions on each)
    k_bet: int = 4  # resampled full rounds per bet state
    deal_groups: int = 2  # unseen-pool permutations per bet state for the side-bet deal Monte Carlo
    k_insurance: int = 8  # resampled hole cards per insurance state
    epsilon: float = 0.10  # behaviour policy: probability of a random legal (non cash-out) action
    p_tilt: float = 0.5  # probability that a shoe gets a tilted shuffle
    tilt_max: float = 0.08  # max |α| of the tilt (0.08 on all shoes: TC 1st/99th pct ≈ ±9 vs ±5.7 natural)
    p_insurance: float = 1.0  # probability of labelling an insurance state

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------- side-bet payout tables
@lru_cache(maxsize=4)
def side_tables(pp_key: tuple, t213_key: tuple, c7_key: tuple) -> dict[str, np.ndarray]:
    """Net return per unit stake for every card combination, tabulated with the engine's settle_*."""
    pp, t213, c7 = dict(pp_key), dict(t213_key), dict(c7_key)
    PP = np.zeros((52, 52), dtype=np.float32)
    C72 = np.zeros((52, 52), dtype=np.float32)
    T = np.zeros((52, 52, 52), dtype=np.float32)
    C73 = np.zeros((52, 52, 52), dtype=np.float32)
    for a in range(52):
        for b in range(52):
            PP[a, b] = payout(1.0, *settle_perfect_pairs(a, b, pp)) - 1.0
            C72[a, b] = payout(1.0, *settle_crazy_7(a, b, None, c7)) - 1.0
            for c in range(52):
                T[a, b, c] = payout(1.0, *settle_21_3(a, b, c, t213)) - 1.0
                if RANK_OF[a] == SEVEN_RANK and RANK_OF[b] == SEVEN_RANK:
                    C73[a, b, c] = payout(1.0, *settle_crazy_7(a, b, c, c7)) - 1.0
                else:
                    C73[a, b, c] = C72[a, b]
    return {"pp": PP, "t213": T, "c72": C72, "c73": C73}


def tables_for(cfg: GameConfig) -> dict[str, np.ndarray]:
    sb = cfg.side_bets
    return side_tables(tuple(sorted(sb.perfect_pairs.items())), tuple(sorted(sb.twenty_one_plus_three.items())),
                       tuple(sorted(sb.crazy_7.items())))


_BUCKET = np.array(BUCKET_OF, dtype=np.int8)


def deal_mc(unseen_cards: list[int], tabs: dict[str, np.ndarray], groups: int, rng: np.random.Generator,
            peek_on_ace: bool = True, insurance_payout: float = 2.0) -> tuple[float, float, float, float, int]:
    """Mean realised returns (PP, 21+3, Crazy 7, insurance-if-offered) over 5-card deals cut from
    ``groups`` random permutations of the unseen pool, and the number of deals used."""
    pool = np.repeat(np.arange(NUM_CARDS, dtype=np.int16), unseen_cards)
    m = len(pool) // 5
    ds = []
    for _ in range(groups):
        perm = rng.permutation(pool)
        ds.append(perm[:5 * m].reshape(m, 5))
    d = np.concatenate(ds) if len(ds) > 1 else ds[0]
    p1, up, p2, hole, nxt = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4]
    pp = tabs["pp"][p1, p2].mean()
    t213 = tabs["t213"][p1, p2, up].mean()
    hole_ten = _BUCKET[hole] == 9
    if peek_on_ace:
        peek_bj = (_BUCKET[up] == 0) & hole_ten
        c7 = np.where(peek_bj, tabs["c72"][p1, p2], tabs["c73"][p1, p2, nxt]).mean()
    else:
        c7 = tabs["c73"][p1, p2, nxt].mean()
    ins = np.where(hole_ten, insurance_payout, -1.0).mean()
    return float(pp), float(t213), float(c7), float(ins), len(d)


# ---------------------------------------------------------------------- tilted shuffles
def tilt_shoe(table: Table, rng: random.Random, tilt_max: float) -> tuple[str, float]:
    """Reorder the undealt cards of a fresh shoe with a biased sort (see module docstring)."""
    u = rng.random()
    if u < 0.5:
        kind, d = "hilo", [float(t) for t in HILO_TAGS]
    elif u < 0.75:
        kind, d = "random", [rng.gauss(0.0, 0.7) for _ in range(10)]
    else:
        kind, d = "sevens", [0.0] * 6 + [2.5] + [0.0] * 3
    a = rng.uniform(-tilt_max, tilt_max)
    shoe = table.shoe
    seg = shoe.cards[shoe.pos:]
    rnd = rng.random
    keyed = sorted(seg, key=lambda c: rnd() + a * d[BUCKET_OF[c]])
    shoe.cards[shoe.pos:] = keyed
    return kind, a


# ---------------------------------------------------------------------- the worker
class Generator:
    def __init__(self, cfg: GameConfig, weights: dict[str, Any] | None, gen: GenConfig, seed: int,
                 spec: FeatureSpec | None = None) -> None:
        self.cfg = cfg
        self.gen = gen
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)
        self.enc = FeatureEncoder(spec)
        self.policy = NetPolicy(NumpyNet(weights), self.enc) if weights is not None else BasicPolicy()
        self.table = Table(cfg, seed=seed, record_events=False)
        self.tabs = tables_for(cfg)
        self.last_shoe = -1
        self.cash = cfg.rules.cashout
        if gen.targets not in ("mc", "solver", "blend"):
            raise ValueError(f"unknown targets {gen.targets!r}")
        from ..strategy.solver import SolverRules
        self.rules = SolverRules.from_config(cfg.rules)
        self.rtp = cfg.rules.cashout_rtp if cfg.rules.cashout else None

    # -- labelling ---------------------------------------------------------
    def solver_targets(self, table: Table, st: tuple) -> list[float]:
        from .oracle import composition_solver, state_evs
        e = state_evs(composition_solver(self.rules, tuple(table.shoe.unseen_buckets()), self.rtp), st)
        return [e.get("stand", 0.0), e.get("hit", 0.0), e.get("double", 0.0), e.get("split", 0.0)]

    def label_play(self, table: Table) -> tuple[np.ndarray, list[float], list[bool]]:
        x = self.enc.encode_table(table, KIND_PLAY)
        st = table.decision_state()
        mask = [True, True, bool(st[4]), bool(st[6])]
        mode = self.gen.targets
        if mode == "solver":
            return x, self.solver_targets(table, st), mask
        y = self.mc_targets(table, st)
        if mode == "blend":
            ys = self.solver_targets(table, st)
            w = self.gen.mc_weight
            y = [a + w * (b - a) for a, b in zip(ys, y)]
        return x, y, mask

    def mc_targets(self, table: Table, st: tuple) -> list[float]:
        legal = [Action.STAND, Action.HIT]
        if st[4]:
            legal.append(Action.DOUBLE)
        if st[6]:
            legal.append(Action.SPLIT)
        i0 = table.active
        nh0 = len(table.hands)
        bet0 = table.hands[i0].bet
        sums = [0.0, 0.0, 0.0, 0.0]
        pol = self.policy
        rng = self.rng
        K = self.gen.k_play
        for _ in range(K):
            base = table.clone(resample_unseen=True, seed=rng.getrandbits(62))
            for a in legal:
                c = base.clone(resample_unseen=False, seed=0)
                c.act(a)
                while c.phase is Phase.PLAYER:
                    c.act(pol.action(c))
                rr = c.round_result
                nd = len(rr.hands) - nh0 + 1
                ret = 0.0
                for h in rr.hands[i0:i0 + nd]:
                    ret += h.payout - h.bet
                sums[PLAY_IDX[a]] += ret / bet0
        return [s / K for s in sums]

    def label_bet(self, table: Table) -> tuple[np.ndarray, list[float]]:
        x = self.enc.encode_table(table, KIND_BET)
        pol = self.policy
        rng = self.rng
        K = self.gen.k_bet
        main = bb = 0.0
        bets = Bets(main=1, bust_bonus=1)
        for _ in range(K):
            c = table.clone(resample_unseen=True, seed=rng.getrandbits(62))
            c.place_bets(bets)
            while True:
                ph = c.phase
                if ph is Phase.PLAYER:
                    c.act(pol.action(c))
                elif ph is Phase.INSURANCE:
                    c.act(Action.NO_INSURANCE)
                else:
                    break
            rr = c.round_result
            main += rr.main_returned - rr.main_wagered
            bb += rr.side_net("bust_bonus")
        r = self.cfg.rules
        pp, t213, c7, ins, _ = deal_mc(table.shoe.unseen_cards(), self.tabs, self.gen.deal_groups, self.nprng,
                                       r.peek_on_ace, r.insurance_payout)
        return x, [main / K, pp, t213, c7, bb / K, ins]

    def label_insurance(self, table: Table) -> tuple[np.ndarray, float]:
        x = self.enc.encode_table(table, KIND_INSURANCE)
        rng = self.rng
        K = self.gen.k_insurance
        pay = self.cfg.rules.insurance_payout
        s = 0.0
        for _ in range(K):
            c = table.clone(resample_unseen=True, seed=rng.getrandbits(62))
            s += pay if c.dealer.blackjack else -1.0
        return x, s / K

    # -- behaviour ---------------------------------------------------------
    def behaviour(self, table: Table) -> Action:
        if self.rng.random() < self.gen.epsilon:
            st = table.decision_state()
            opts = [Action.STAND, Action.HIT]
            if st[4]:
                opts.append(Action.DOUBLE)
            if st[6]:
                opts.append(Action.SPLIT)
            return opts[self.rng.randrange(len(opts))]
        return self.policy.action(table)

    # -- main loop ---------------------------------------------------------
    def run(self, n_play: int, n_bet: int) -> dict[str, Any]:
        F = self.enc.size
        Xp = np.zeros((n_play, F), dtype=np.float32)
        Yp = np.zeros((n_play, 4), dtype=np.float32)
        Mp = np.zeros((n_play, 4), dtype=bool)
        cap_b = n_bet + n_bet // 4 + 64
        Xb = np.zeros((cap_b, F), dtype=np.float32)
        Yb = np.zeros((cap_b, 6), dtype=np.float32)
        Mb = np.zeros((cap_b, 6), dtype=bool)
        Xi: list[np.ndarray] = []  # insurance rows (no quota: ~1 per 13 rounds)
        Yi: list[float] = []
        ip = ib = 0
        rounds = 0
        tilts = 0
        t0 = time.perf_counter()
        table = self.table
        rng = self.rng
        g = self.gen
        play_per_round = 1.3
        bet_full = [True] * 5 + [True]
        ins_mask = [False] * 5 + [True]
        while ip < n_play:
            if table.shoe.shoe_number != self.last_shoe:
                self.last_shoe = table.shoe.shoe_number
                if rng.random() < g.p_tilt:
                    tilt_shoe(table, rng, g.tilt_max)
                    tilts += 1
            # bet states: spread over the run so that ~n_bet are labelled when the play quota is met
            rounds_left = max(1.0, (n_play - ip) / play_per_round)
            p_bet = min(1.0, max(0.0, (n_bet - ib) / rounds_left))
            if ib < cap_b - 8 and rng.random() < p_bet:
                x, y = self.label_bet(table)
                Xb[ib], Yb[ib], Mb[ib] = x, y, bet_full
                ib += 1
            table.place_bets(Bets(main=1))
            rounds += 1
            while True:
                ph = table.phase
                if ph is Phase.PLAYER:
                    if ip < n_play:
                        x, y, m = self.label_play(table)
                        Xp[ip], Yp[ip], Mp[ip] = x, y, m
                        ip += 1
                    table.act(self.behaviour(table))
                elif ph is Phase.INSURANCE:
                    if rng.random() < g.p_insurance:
                        x, yi = self.label_insurance(table)
                        Xi.append(x)
                        Yi.append(yi)
                    table.act(Action.NO_INSURANCE)
                else:
                    break
            play_per_round = 0.98 * play_per_round + 0.02 * (ip / rounds) if rounds > 50 else play_per_round
        ni = len(Xi)
        xi = np.array(Xi, dtype=np.float32).reshape(ni, F)
        yi = np.zeros((ni, 6), np.float32)
        yi[:, 5] = Yi
        mi = np.tile(np.array(ins_mask), (ni, 1))
        return {"Xp": Xp[:ip], "Yp": Yp[:ip], "Mp": Mp[:ip], "Xb": np.concatenate([Xb[:ib], xi]),
                "Yb": np.concatenate([Yb[:ib], yi]), "Mb": np.concatenate([Mb[:ib], mi]),
                "rounds": rounds, "tilted_shoes": tilts, "shoes": table.shoe.shoe_number,
                "seconds": time.perf_counter() - t0}


def _worker(args: tuple) -> dict[str, Any]:
    cfg_dict, weights, gen_dict, spec_dict, n_play, n_bet, seed = args
    cfg = from_dict(cfg_dict)
    g = Generator(cfg, weights, GenConfig(**gen_dict), seed, FeatureSpec.from_dict(spec_dict))
    return g.run(n_play, n_bet)


def generate(cfg: GameConfig, weights: dict[str, Any] | None, gen: GenConfig, n_play: int, n_bet: int,
             seed: int, workers: int = 1, spec: FeatureSpec | None = None, chunks_per_worker: int = 4,
             progress: Any = None) -> dict[str, Any]:
    """Generate ``n_play`` play states and ≈``n_bet`` bet (+ insurance) states over ``workers`` processes.
    Deterministic for a given ``seed`` and ``workers``/``chunks_per_worker`` split."""
    import multiprocessing as mp

    from ..config import to_dict

    n_chunks = max(1, workers * chunks_per_worker) if workers > 1 else 1
    per_p = [n_play // n_chunks + (1 if i < n_play % n_chunks else 0) for i in range(n_chunks)]
    per_b = [n_bet // n_chunks + (1 if i < n_bet % n_chunks else 0) for i in range(n_chunks)]
    spec_d = (spec or FeatureSpec()).to_dict()
    jobs = [(to_dict(cfg), weights, gen.to_dict(), spec_d, per_p[i], per_b[i], seed * 7919 + i * 104729 + 1)
            for i in range(n_chunks)]
    t0 = time.perf_counter()
    parts: list[dict[str, Any]] = []
    if workers <= 1:
        for j in jobs:
            parts.append(_worker(j))
            if progress:
                progress(len(parts), len(jobs))
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for p in pool.imap(_worker, jobs):
                parts.append(p)
                if progress:
                    progress(len(parts), len(jobs))
    out: dict[str, Any] = {k: np.concatenate([p[k] for p in parts]) for k in ("Xp", "Yp", "Mp", "Xb", "Yb", "Mb")}
    out["rounds"] = sum(p["rounds"] for p in parts)
    out["tilted_shoes"] = sum(p["tilted_shoes"] for p in parts)
    out["shoes"] = sum(p["shoes"] for p in parts)
    out["cpu_seconds"] = sum(p["seconds"] for p in parts)
    out["wall_seconds"] = time.perf_counter() - t0
    return out


__all__ = ["GenConfig", "Generator", "deal_mc", "generate", "tables_for", "tilt_shoe"]
