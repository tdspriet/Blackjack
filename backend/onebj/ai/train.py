"""Train the ONE Blackjack network (generalised policy iteration with Monte-Carlo action values).

CLI::

    uv run python -m onebj.ai.train --name nn_v1                 # default run (~14 min on an M4, 10 cores)
    uv run python -m onebj.ai.train --name smoke --quick         # ~10 s sanity run
    uv run python -m onebj.ai.train --name nn_v2 --config my.json --scale 2 --steps 120000
    uv run python -m onebj.ai.train --name nocash --game-patch '{"rules": {"cashout": false}}' --out checkpoints/ablations

Algorithm (details, measurements and justification in docs/NN.md §2)
    for generation g = 0 .. G-1:
        1. data: play rounds with the current policy (basic strategy for g = 0, then the network's greedy
           policy) + ε-exploration on tilted shoes and label the visited states (:mod:`onebj.ai.datagen`):
           play states with per-action targets — by default the EVs of the cash-aware composition solver
           (:mod:`onebj.ai.oracle`), optionally blended with / replaced by all-action Monte-Carlo returns on
           resampled clones (``targets_per_generation``); bet states with realised returns (main under the
           current policy, Bust Bonus) + deal Monte Carlo (PP, 21+3, Crazy 7, insurance); insurance states with
           resampled hole cards
        2. fit: masked MSE on the play head + standardised masked MSE on the bet head over *all* data
           generated so far (warm start, cosine learning-rate cycle per generation)
        3. evaluate: paired ("duplicate") flat-bet comparison against basic strategy on identical cards, and
           solver regret on a fixed benchmark of natural decision states → training log

Outputs ``checkpoints/<name>.pt`` (weights + architecture + feature spec) and ``checkpoints/<name>.json``
(metadata: config hash, feature names, hyper-parameters, data sizes, timing, metrics and the training log
``[{step, loss, eval_ev, ...}]`` plotted by the frontend's Training tab). Everything is seeded: the same
``--seed`` and config reproduce the same data and (on the same machine/thread count) the same weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import GameConfig, load_config, to_dict
from .agent import BetConfig, now_iso, resolve_dir
from .datagen import GenConfig, generate
from .features import FeatureEncoder, FeatureSpec
from .model import BET_LABELS, PLAY_LABELS, BlackjackNet, NumpyNet, build_model


@dataclass
class TrainConfig:
    """All training hyper-parameters (``--config`` JSON files may override any subset)."""

    seed: int = 1
    trunk: list[int] = field(default_factory=lambda: [128, 64])
    # data per generation (play states, bet states); generation 0 uses basic strategy as policy
    generations: list[dict[str, int]] = field(default_factory=lambda: [
        {"play": 4_000_000, "bet": 1_500_000},
        {"play": 3_000_000, "bet": 1_500_000},
        {"play": 3_000_000, "bet": 1_500_000},
    ])
    steps_per_generation: list[int] = field(default_factory=lambda: [20000, 15000, 25000])
    # play-head target mode per generation ("solver" | "blend" | "mc"); None → gen["targets"] everywhere.
    # Default: distil the cash-aware composition solver in every generation (later generations add data from the
    # network's own state distribution and on-policy bet-head targets). ["solver", "blend", "blend"] (engine
    # rollout corrections) was measured 0.020 ± 0.012 % per round worse — see docs/NN.md §2.1.
    targets_per_generation: list[str] | None = field(default_factory=lambda: ["solver", "solver", "solver"])
    lr_decay_per_generation: float = 0.5  # peak LR of generation g = lr * decay**g (warm-started fits)
    batch_size: int = 4096
    lr: float = 2e-3
    lr_min: float = 2e-5
    weight_decay: float = 1e-6
    bet_loss_weight: float = 0.5
    val_frac: float = 0.02
    eval_rounds: int = 1_000_000  # paired evaluation vs basic after each generation
    bench_states: int = 40_000  # natural decision states for the solver-regret diagnostic
    save_generations: bool = True  # also keep a snapshot per generation in checkpoints/<name>_generations/
    log_every: int = 100
    chunks: int = 40  # data-generation chunks per generation (fixed → reproducible across worker counts)
    torch_threads: int = 4
    feature_spec: dict[str, Any] = field(default_factory=lambda: FeatureSpec().to_dict())
    gen: dict[str, Any] = field(default_factory=lambda: GenConfig().to_dict())
    bet: dict[str, Any] = field(default_factory=lambda: asdict(BetConfig()))
    description: str = ("ONE Blackjack MLP 61-128-64: play Q head distilled from the cash-aware composition solver, "
                        "bet EV head from realised engine returns + deal Monte Carlo; 3 generations of policy iteration")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> TrainConfig:
        base = asdict(cls())
        for k, v in (d or {}).items():
            if k not in base:
                raise ValueError(f"unknown training config key {k!r}")
            if isinstance(base[k], dict) and isinstance(v, dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
        return cls(**base)


def quick_config() -> TrainConfig:
    """Tiny preset for smoke tests (~1 minute)."""
    return TrainConfig(generations=[{"play": 20_000, "bet": 8_000}, {"play": 10_000, "bet": 4_000}],
                       steps_per_generation=[300, 300], batch_size=1024, eval_rounds=20_000, log_every=50, chunks=8,
                       bench_states=4_000)


def config_hash(tc: TrainConfig, cfg: GameConfig) -> str:
    blob = json.dumps({"train": asdict(tc), "game": to_dict(cfg)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------- dataset
class Dataset:
    """Play rows (Q targets) and bet rows (bet-head targets) in one table with masks.

    Features are stored as float16 (halves the memory of multi-million-row datasets; the rounding is far
    below the target noise) and cast to float32 per batch. Every row gets a fixed train/validation flag
    when it is added, so the validation set is stable across generations.
    """

    KEYS = ("X", "Yq", "Mq", "Yb", "Mb")

    def __init__(self, n_features: int, val_frac: float, seed: int) -> None:
        self.F = n_features
        self.val_frac = val_frac
        self.rng = np.random.default_rng(seed)
        self.t: dict[str, torch.Tensor] = {}
        self.is_val = torch.zeros(0, dtype=torch.bool)

    def add(self, d: dict[str, Any]) -> None:
        npl, nb = len(d["Xp"]), len(d["Xb"])
        n = npl + nb
        x = np.empty((n, self.F), np.float16)
        x[:npl], x[npl:] = d["Xp"], d["Xb"]
        yq = np.zeros((n, 4), np.float32)
        mq = np.zeros((n, 4), bool)
        yb = np.zeros((n, 6), np.float32)
        mb = np.zeros((n, 6), bool)
        yq[:npl], mq[:npl] = d["Yp"], d["Mp"]
        yb[npl:], mb[npl:] = d["Yb"], d["Mb"]
        new = {"X": x, "Yq": yq, "Mq": mq, "Yb": yb, "Mb": mb}
        for k in self.KEYS:
            v = torch.from_numpy(new[k])
            self.t[k] = torch.cat([self.t[k], v]) if k in self.t else v
        self.is_val = torch.cat([self.is_val, torch.from_numpy(self.rng.random(n) < self.val_frac)])

    def __len__(self) -> int:
        return len(self.is_val)

    def batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        b = {k: v[idx] for k, v in self.t.items()}
        b["X"] = b["X"].float()
        return b


def losses(model: BlackjackNet, b: dict[str, torch.Tensor], bet_std: torch.Tensor) -> tuple[torch.Tensor, ...]:
    q, bet = model(b["X"])
    mq = b["Mq"].float()
    lp = (((q[:, :4] - b["Yq"]) ** 2) * mq).sum() / mq.sum().clamp(min=1.0)
    mb = b["Mb"].float()
    lb = ((((bet - b["Yb"]) / bet_std) ** 2) * mb).sum() / mb.sum().clamp(min=1.0)
    return lp, lb


@torch.no_grad()
def eval_losses(model: BlackjackNet, data: Dataset, idx: torch.Tensor, bet_std: torch.Tensor,
                bs: int = 65536) -> tuple[float, float]:
    n = len(idx)
    sp = sb = 0.0
    wp = wb = 0.0
    for i in range(0, n, bs):
        b = data.batch(idx[i:i + bs])
        lp, lb = losses(model, b, bet_std)
        cp, cb = float(b["Mq"].sum()), float(b["Mb"].sum())
        sp += float(lp) * cp
        sb += float(lb) * cb
        wp += cp
        wb += cb
    return sp / max(wp, 1), sb / max(wb, 1)


# ---------------------------------------------------------------------- checkpoint I/O
def save_checkpoint(out_dir: Path, name: str, model: BlackjackNet, enc: FeatureEncoder, meta: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pt = out_dir / f"{name}.pt"
    torch.save({"format": 1, "arch": model.arch(), "state_dict": model.state_dict(),
                "feature_spec": enc.spec.to_dict(), "feature_names": enc.names}, pt)
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))
    return pt


def load_model(pt: Path) -> BlackjackNet:
    ck = torch.load(pt, map_location="cpu", weights_only=True)
    m = build_model(ck["arch"])
    m.load_state_dict(ck["state_dict"])
    return m


# ---------------------------------------------------------------------- training
def train(name: str, tc: TrainConfig, cfg: GameConfig | None = None, out_dir: str | Path = "checkpoints",
          workers: int | None = None, init: str | None = None, verbose: bool = True) -> dict[str, Any]:
    """Run the full training; returns the metadata written to ``<out_dir>/<name>.json``."""
    t_start = time.perf_counter()
    cfg = cfg or load_config()
    out = resolve_dir(out_dir)
    workers = workers or mp.cpu_count()
    torch.manual_seed(tc.seed)
    torch.set_num_threads(tc.torch_threads)
    spec = FeatureSpec.from_dict(tc.feature_spec)
    enc = FeatureEncoder(spec)
    gen_cfg = GenConfig(**tc.gen)
    if init:
        from .agent import resolve_checkpoint
        model = load_model(resolve_checkpoint(init, out_dir))
        weights: dict[str, Any] | None = NumpyNet.export(model)
    else:
        model = BlackjackNet(enc.size, enc.i_cash, tc.trunk)
        weights = None
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    data = Dataset(enc.size, tc.val_frac, tc.seed + 17)
    log: list[dict[str, Any]] = []
    gen_meta: list[dict[str, Any]] = []
    step = 0
    bet_std = torch.ones(6)
    rng = torch.Generator().manual_seed(tc.seed)
    from .analysis import benchmark_states, regret_report
    from .evaluate import duplicate_eval

    def say(msg: str) -> None:
        if verbose:
            print(f"[{time.perf_counter() - t_start:7.0f}s] {msg}", flush=True)

    say(f"training {name}: {len(tc.generations)} generations, {workers} workers, features={enc.size}, "
        f"trunk={tc.trunk}")
    last_eval: dict[str, Any] = {}
    bench = benchmark_states(cfg, tc.bench_states, seed=tc.seed + 4242, workers=workers) if tc.bench_states else None
    last_reg: dict[str, Any] = {}
    for g, sizes in enumerate(tc.generations):
        if tc.targets_per_generation:
            tg = tc.targets_per_generation[min(g, len(tc.targets_per_generation) - 1)]
            gen_cfg = GenConfig(**{**tc.gen, "targets": tg})
        # 1. data ---------------------------------------------------------------------------
        t0 = time.perf_counter()
        pol = "basic strategy" if weights is None else "network"
        say(f"gen {g}: generating {sizes['play']:,} play + ~{sizes['bet']:,} bet states (policy: {pol}, "
            f"play targets: {gen_cfg.targets})")
        d = _generate(cfg, weights, gen_cfg, sizes, tc, g, workers, spec)
        t_gen = time.perf_counter() - t0
        n_ins = int((~d["Mb"][:, 0]).sum()) if len(d["Mb"]) else 0
        say(f"gen {g}: {len(d['Xp']):,} play / {len(d['Xb']) - n_ins:,} bet / {n_ins:,} insurance states from "
            f"{d['rounds']:,} rounds in {t_gen:.0f}s ({d['cpu_seconds'] / max(t_gen, 1e-9):.1f} cores busy)")
        ginfo = {"generation": g, "policy": pol, "targets": gen_cfg.targets,
                 "play_states": len(d["Xp"]), "bet_states": len(d["Xb"]) - n_ins, "insurance_states": n_ins,
                 "rounds": d["rounds"], "tilted_shoes": d["tilted_shoes"], "shoes": d["shoes"], "gen_seconds": t_gen}
        data.add(d)
        if g == 0 and not init:
            yb, mb = d["Yb"], d["Mb"]
            off = np.array([yb[mb[:, k], k].mean() if mb[:, k].any() else 0.0 for k in range(6)], np.float32)
            sd = np.array([yb[mb[:, k], k].std() if mb[:, k].any() else 1.0 for k in range(6)], np.float32)
            sd = np.clip(sd, 0.02, 10.0)
            model.bet_offset.copy_(torch.from_numpy(off))
            model.bet_scale.copy_(torch.from_numpy(sd))
        bet_std = model.bet_scale.detach().clone()
        del d
        n = len(data)
        val_idx = torch.nonzero(data.is_val).squeeze(1)
        tr_idx = torch.nonzero(~data.is_val).squeeze(1)
        n_tr = len(tr_idx)
        # 2. fit ----------------------------------------------------------------------------
        steps = tc.steps_per_generation[min(g, len(tc.steps_per_generation) - 1)]
        t1 = time.perf_counter()
        acc_l = acc_p = acc_b = 0.0
        acc_n = 0
        model.train()
        for s in range(steps):
            peak = tc.lr * tc.lr_decay_per_generation ** g
            lr = tc.lr_min + 0.5 * (peak - tc.lr_min) * (1 + math.cos(math.pi * s / max(steps - 1, 1)))
            for pg in opt.param_groups:
                pg["lr"] = lr
            idx = tr_idx[torch.randint(0, n_tr, (tc.batch_size,), generator=rng)]
            b = data.batch(idx)
            lp, lb = losses(model, b, bet_std)
            loss = lp + tc.bet_loss_weight * lb
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            acc_l += loss.item()
            acc_p += lp.item()
            acc_b += lb.item()
            acc_n += 1
            if step % tc.log_every == 0:
                log.append({"step": step, "loss": acc_l / acc_n, "loss_play": acc_p / acc_n, "loss_bet": acc_b / acc_n,
                            "lr": lr, "generation": g})
                acc_l = acc_p = acc_b = 0.0
                acc_n = 0
        t_fit = time.perf_counter() - t1
        model.eval()
        vp, vb = eval_losses(model, data, val_idx, bet_std)
        weights = NumpyNet.export(model)
        # 3. evaluate -----------------------------------------------------------------------
        t2 = time.perf_counter()
        ev = duplicate_eval(("weights", weights, spec.to_dict(), None), "basic", cfg, tc.eval_rounds,
                            seed=tc.seed * 31 + 7, workers=workers)
        reg = regret_report(NumpyNet(weights), bench) if bench is not None else {}
        last_reg = reg
        t_eval = time.perf_counter() - t2
        row = {"step": step, "loss": vp + tc.bet_loss_weight * vb, "val_loss_play": vp, "val_loss_bet": vb,
               "eval_ev": ev["ev_a"], "eval_ev_basic": ev["ev_b"], "gain_vs_basic": ev["diff"],
               "gain_se": ev["diff_se"], "agreement_basic": ev["agreement_basic"], "generation": g,
               "data_rows": n}
        if reg:
            row.update({"regret": reg["net"]["regret"], "regret_basic": reg["basic"]["regret"],
                        "cashout_rate": reg["net"]["cashout_rate"], "agree_solver": reg["net"]["agree_solver"],
                        "q_rmse_vs_solver": reg["q_rmse_vs_solver"]})
        log.append(row)
        last_eval = ev
        gen_meta.append({**ginfo, "fit_seconds": t_fit, "eval_seconds": t_eval, "train_rows": n_tr, "steps": steps})
        if tc.save_generations:
            snap = dict(row, name=f"{name}/gen{g}", created=now_iso(), feature_names=enc.names,
                        feature_spec=spec.to_dict(), bet_config=tc.bet,
                        metrics={k: row.get(k) for k in ("eval_ev", "gain_vs_basic", "gain_se", "regret")})
            save_checkpoint(out / f"{name}_generations", f"gen{g}", model, enc, snap)
        say(f"gen {g}: fit {steps} steps in {t_fit:.0f}s, val loss play {vp:.4f} bet {vb:.4f}; paired eval "
            f"{tc.eval_rounds:,} rounds: NN {ev['ev_a']:+.5f} vs basic {ev['ev_b']:+.5f} → gain {ev['diff']:+.5f} "
            f"± {ev['diff_se']:.5f}, agreement {100 * ev['agreement_basic']:.2f}%"
            + (f"; solver regret {reg['net']['regret']:.5f} (basic {reg['basic']['regret']:.5f}), cash-out rate "
               f"{100 * reg['net']['cashout_rate']:.1f}%, Q rmse {reg['q_rmse_vs_solver']:.4f}" if reg else ""))
    total = time.perf_counter() - t_start
    meta = {
        "name": name, "created": now_iso(), "config_hash": config_hash(tc, cfg), "description": tc.description,
        "feature_names": enc.names, "feature_spec": spec.to_dict(), "arch": model.arch(),
        "play_labels": PLAY_LABELS, "bet_labels": BET_LABELS, "bet_config": tc.bet,
        "train_config": asdict(tc), "gen_config": gen_cfg.to_dict(), "game_config": to_dict(cfg),
        "data": gen_meta, "timing": {"total_minutes": total / 60, "workers": workers},
        "metrics": {"eval_ev": last_eval.get("ev_a"), "eval_ev_basic": last_eval.get("ev_b"),
                    "gain_vs_basic": last_eval.get("diff"), "gain_se": last_eval.get("diff_se"),
                    "agreement_basic": last_eval.get("agreement_basic"), "eval_rounds": tc.eval_rounds,
                    "train_minutes": round(total / 60, 1), "steps": step,
                    **({"solver_regret": last_reg["net"]["regret"], "solver_regret_basic": last_reg["basic"]["regret"],
                        "cashout_rate": last_reg["net"]["cashout_rate"]} if last_reg else {})},
        "training_log": log,
    }
    pt = save_checkpoint(out, name, model, enc, meta)
    say(f"saved {pt} and {pt.with_suffix('.json')} — total {total / 60:.1f} min")
    return meta


def _generate(cfg: GameConfig, weights: dict | None, gen_cfg: GenConfig, sizes: dict[str, int], tc: TrainConfig,
              g: int, workers: int, spec: FeatureSpec) -> dict[str, Any]:
    """Data for one generation in ``tc.chunks`` deterministic chunks (independent of the worker count)."""
    chunks = max(1, tc.chunks)
    per_worker = max(1, math.ceil(chunks / max(workers, 1)))
    eff_workers = max(1, math.ceil(chunks / per_worker))
    return generate(cfg, weights, gen_cfg, sizes["play"], sizes["bet"], seed=tc.seed * 1000 + g,
                    workers=eff_workers if workers > 1 else 1, spec=spec, chunks_per_worker=per_worker)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train the ONE Blackjack network (see docs/NN.md)")
    ap.add_argument("--name", default="nn_v1")
    ap.add_argument("--config", default=None, help="training config JSON (any subset of TrainConfig)")
    ap.add_argument("--game-config", default=None, help="game config JSON (default config/default.json)")
    ap.add_argument("--game-patch", default=None, help='JSON patch of the game config, e.g. \'{"rules": {"cashout": false}}\'')
    ap.add_argument("--steps", type=int, default=None, help="total optimiser steps (split like the default schedule)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the data per generation")
    ap.add_argument("--generations", type=int, default=None, help="number of generations (repeats the last size)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--init", default=None, help="warm start from a checkpoint (its policy drives generation 0)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test preset")
    a = ap.parse_args(argv)
    tc = quick_config() if a.quick else TrainConfig()
    if a.config:
        tc = TrainConfig.from_dict({**asdict(tc), **json.loads(Path(a.config).read_text())})
    if a.seed is not None:
        tc.seed = a.seed
    if a.generations:
        last_g, last_s = tc.generations[-1], tc.steps_per_generation[-1]
        tc.generations = (tc.generations + [copy.deepcopy(last_g)] * a.generations)[:a.generations]
        tc.steps_per_generation = (tc.steps_per_generation + [last_s] * a.generations)[:a.generations]
    if a.scale != 1.0:
        tc.generations = [{k: int(v * a.scale) for k, v in gsz.items()} for gsz in tc.generations]
    if a.steps:
        tot = sum(tc.steps_per_generation)
        tc.steps_per_generation = [max(1, round(s * a.steps / tot)) for s in tc.steps_per_generation]
    cfg = load_config(a.game_config, patch=json.loads(a.game_patch) if a.game_patch else None)
    train(a.name, tc, cfg, a.out, a.workers or None, a.init)


if __name__ == "__main__":
    main()
