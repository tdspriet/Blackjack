"""Headless simulations for the ``simulate`` command, off the event loop.

A simulation runs in ``workers`` spawned processes (each plays its share of the rounds on its own
fresh table, unlimited virtual bankroll) and streams :class:`LabMetrics` snapshots back through a
queue about every ``REPORT_SECONDS``; :class:`SimulationRun` merges them and calls ``on_progress``
(throttled to ~2/s). ``stop()`` sets a shared event the workers poll every few rounds.

If processes cannot be started (restricted environments) the same worker function runs in threads.
The worker entry point lives in this importable module, so the ``spawn`` start method works from any
``__main__`` (uvicorn, pytest).
"""

from __future__ import annotations

import asyncio
import logging
import math
import multiprocessing as mp
import os
import queue as queue_mod
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import from_dict
from ..engine import Action, Bets, Observation, Table
from ..sim import run_rounds
from .agents import make_agent
from .metrics import LabMetrics, merge_parts

log = logging.getLogger("onebj.server")

REPORT_SECONDS = 0.25
EMIT_SECONDS = 0.5
MAX_ROUNDS = 5_000_000


@dataclass(frozen=True)
class SimJob:
    config: dict[str, Any]
    agent: str
    rounds: int
    seed: int
    start: float
    checkpoints: str | None = None


class _AgreementAgent:
    """Wraps an agent to count agreement of its play decisions with basic strategy."""

    def __init__(self, inner: Any, metrics: LabMetrics, basic: Any) -> None:
        self.inner = inner
        self.m = metrics
        self._bs = basic
        self.name = getattr(inner, "name", "agent")

    def decide_bets(self, obs: Observation) -> tuple[Bets, Any]:
        return self.inner.decide_bets(obs)

    def decide_insurance(self, obs: Observation) -> tuple[bool, Any]:
        return self.inner.decide_insurance(obs)

    def decide_action(self, obs: Observation) -> tuple[Action, Any]:
        a, tr = self.inner.decide_action(obs)
        ref = (tr or {}).get("reference", {}).get("basic_strategy") if isinstance(tr, dict) else None
        if ref is None:
            h = obs.hand
            ref = self._bs.action(h.total, h.soft, h.num_cards, min(obs.dealer_up // 4, 9), h.can_double, h.pair,
                                  h.can_split)
        self.m.decision(Action(a).value == ref)
        return a, tr


class _FastAgreementAgent(_AgreementAgent):
    def fast_bets(self, table: Table) -> Bets:
        return self.inner.fast_bets(table)

    def fast_insurance(self, table: Table) -> bool:
        return self.inner.fast_insurance(table)

    def fast_action(self, table: Table, state: tuple) -> str:
        a = self.inner.fast_action(table, state)
        total, soft, n, up, can_double, pair, can_split, _ = state
        self.m.decision(a == table.strategy.action(total, soft, n, up, can_double, pair, can_split))
        return a


def worker_main(job: SimJob, idx: int, rounds: int, seed: int, out: Any, stop: Any) -> None:
    """Simulation worker (process or thread): reports ("progress"|"done"|"error", idx, payload)."""
    try:
        cfg = from_dict(job.config)
        m = LabMetrics()
        inner = make_agent(job.agent, cfg, Path(job.checkpoints) if job.checkpoints else None, trace=False)
        fast = hasattr(inner, "fast_action")
        agent = (_FastAgreementAgent if fast else _AgreementAgent)(inner, m, None if fast else _basic(cfg))
        table = Table(cfg, seed=seed, record_events=False)
        last = time.monotonic()

        def progress(_: Any) -> None:
            nonlocal last
            now = time.monotonic()
            if now - last >= REPORT_SECONDS:
                last = now
                out.put(("progress", idx, m))

        run_rounds(table, agent, rounds, metrics=m, progress=progress, progress_every=25,
                   should_stop=stop.is_set)
        out.put(("done", idx, m))
    except BaseException as e:  # noqa: BLE001 - report everything to the parent
        out.put(("error", idx, f"{type(e).__name__}: {e}"))


def _basic(cfg: Any) -> Any:
    from ..strategy.basic import get_basic_strategy
    return get_basic_strategy(cfg.rules, cfg.shoe.num_decks)


def default_workers(rounds: int) -> int:
    cores = os.cpu_count() or 2
    return max(1, min(8, cores - 1, math.ceil(rounds / 2500)))


ProgressFn = Callable[[int, int, dict[str, Any], bool], Awaitable[None]]


class SimulationRun:
    """One running simulation. ``await run.run()`` until finished; ``run.stop()`` to cancel."""

    def __init__(self, job: SimJob, on_progress: ProgressFn, workers: int | None = None,
                 use_processes: bool | None = None) -> None:
        self.job = job
        self.on_progress = on_progress
        self.workers = max(1, min(workers or default_workers(job.rounds), job.rounds))
        if use_processes is None:
            use_processes = os.environ.get("ONEBJ_SIM_PROCESSES", "1") != "0"
        self.use_processes = use_processes
        self.error: str | None = None
        self.stopped = False
        self._procs: list[Any] = []
        self._stop: Any = None
        self._queue: Any = None

    def stop(self) -> None:
        self.stopped = True
        if self._stop is not None:
            self._stop.set()

    def _start(self) -> None:
        n, total = self.workers, self.job.rounds
        per = [total // n + (1 if i < total % n else 0) for i in range(n)]
        seeds = [self.job.seed * 1_000_003 + i for i in range(n)]
        if self.use_processes:
            try:
                ctx = mp.get_context("spawn")
                self._queue, self._stop = ctx.Queue(), ctx.Event()
                for i in range(n):
                    p = ctx.Process(target=worker_main, args=(self.job, i, per[i], seeds[i], self._queue, self._stop),
                                    daemon=True, name=f"onebj-sim-{i}")
                    p.start()
                    self._procs.append(p)
                return
            except Exception as e:  # noqa: BLE001  # pragma: no cover - environment specific
                log.warning("cannot start simulation processes (%s); using threads", e)
                self.terminate()
                self._procs = []
        self._queue, self._stop = queue_mod.Queue(), threading.Event()
        for i in range(n):
            t = threading.Thread(target=worker_main, args=(self.job, i, per[i], seeds[i], self._queue, self._stop),
                                 daemon=True, name=f"onebj-sim-{i}")
            t.start()
            self._procs.append(t)

    def terminate(self) -> None:
        for p in self._procs:
            if isinstance(p, mp.process.BaseProcess) and p.is_alive():
                p.terminate()

    async def run(self) -> dict[str, Any]:
        """Run to completion (or stop); returns the final merged metrics dict."""
        self._start()
        if self.stopped:
            self._stop.set()
        t0 = time.perf_counter()
        parts: dict[int, LabMetrics] = {}
        finished: set[int] = set()
        last_emit = 0.0
        q = self._queue

        def merged() -> dict[str, Any]:
            m = merge_parts([parts[i] for i in sorted(parts)], self.job.start)
            d = m.to_dict()
            wall = time.perf_counter() - t0
            d["rounds_per_sec"] = round(m.rounds / wall) if wall > 0 and m.rounds else None
            return d

        try:
            while len(finished) < self.workers:
                try:
                    kind, idx, payload = await asyncio.to_thread(q.get, True, 0.2)
                except queue_mod.Empty:
                    if self._all_dead() and len(finished) < self.workers:
                        # drain anything left, then give up on silent workers
                        await asyncio.sleep(0.2)
                        if q.empty():
                            self.error = self.error or "simulation worker exited unexpectedly"
                            break
                    continue
                if kind == "error":
                    self.error = payload
                    finished.add(idx)
                    self.stop()
                    continue
                parts[idx] = payload
                if kind == "done":
                    finished.add(idx)
                now = time.perf_counter()
                if len(finished) < self.workers and now - last_emit >= EMIT_SECONDS:
                    last_emit = now
                    d = merged()
                    await self.on_progress(d["rounds"], self.job.rounds, d, False)
        finally:
            await asyncio.to_thread(self._join)
        d = merged()
        await self.on_progress(d["rounds"], self.job.rounds, d, True)
        return d

    def _all_dead(self) -> bool:
        return all(not p.is_alive() for p in self._procs)

    def _join(self) -> None:
        deadline = time.monotonic() + 5
        for p in self._procs:
            p.join(max(0.0, deadline - time.monotonic()))
        self.terminate()
