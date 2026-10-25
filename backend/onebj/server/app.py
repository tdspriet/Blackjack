"""FastAPI app (SPEC §6): REST + WebSocket ``/ws``, card images, and the built frontend.

Run (dev):  ``uv run uvicorn onebj.server.app:app --reload --port 8000``
Production: ``npm run build`` in ``frontend/`` then the same command without ``--reload``: the app serves
``frontend/dist`` at ``/`` (SPA fallback to ``index.html``).

Environment: ``ONEBJ_CHECKPOINTS`` (checkpoint directory, default ``<repo>/checkpoints``),
``ONEBJ_DEBUG=1`` (enables the ``debug_rig`` command used by tests / screenshots),
``ONEBJ_SIM_PROCESSES=0`` (run simulations in threads instead of worker processes).
WebSocket query parameter ``?seed=N`` makes a session reproducible.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import load_config, to_dict
from . import agents as registry
from .session import Session

log = logging.getLogger("onebj.server")

REPO_ROOT = registry.REPO_ROOT
CARDS_DIR = REPO_ROOT / "cards"
DIST_DIR = REPO_ROOT / "frontend" / "dist"
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_LOG_KEYS = ("training_log", "log", "history", "training")


def load_training_log(name: str, directory: Path | None = None) -> list[dict[str, Any]]:
    """Training log rows ``[{step, loss, eval_ev, ...}]`` of checkpoint ``name``.

    Reads ``<checkpoints>/<name>.json``: either the list itself, or a metadata object holding it under
    one of ``training_log`` / ``log`` / ``history`` / ``training`` (possibly nested one level)."""
    if not _NAME_RE.match(name) or name.startswith("."):
        raise HTTPException(400, "invalid checkpoint name")
    d = directory or registry.checkpoints_dir()
    path = d / f"{name}.json"
    if not path.is_file():
        raise HTTPException(404, f"no training log for {name!r}")
    try:
        data = json.loads(path.read_text())
    except ValueError as e:
        raise HTTPException(500, f"unreadable training log: {e}") from None
    return _find_log(data)


def _find_log(data: Any, depth: int = 0) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict) and depth < 2:
        for k in _LOG_KEYS:
            if k in data:
                rows = _find_log(data[k], depth + 1)
                if rows:
                    return rows
    return []


def create_app(debug: bool | None = None, checkpoints: Path | None = None,
               sim_processes: bool | None = None) -> FastAPI:
    debug = os.environ.get("ONEBJ_DEBUG", "0") not in ("", "0", "false") if debug is None else debug
    app = FastAPI(title="ONE Blackjack Lab", version=__version__)

    def ck_dir() -> Path:
        return checkpoints or registry.checkpoints_dir()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        cks = registry.list_checkpoints(ck_dir())
        return {"status": "ok", "version": __version__, "checkpoints": len(cks),
                "agents": registry.available_agents(ck_dir()), "debug": debug}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return to_dict(load_config())

    @app.get("/api/checkpoints")
    def checkpoints_list() -> list[dict[str, Any]]:
        return [{"name": c["name"], "created": c.get("created"), "metrics": c.get("metrics")}
                for c in registry.list_checkpoints(ck_dir())]

    @app.get("/api/training/{name}")
    def training(name: str) -> list[dict[str, Any]]:
        return load_training_log(name, ck_dir())

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        seed_q = ws.query_params.get("seed")
        seed = int(seed_q) if seed_q and seed_q.lstrip("-").isdigit() else None
        session = Session(ws.send_text, seed=seed, debug=debug, checkpoints=ck_dir(), sim_processes=sim_processes)
        try:
            await session.start()
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                text = msg.get("text")
                if text is None and msg.get("bytes") is not None:
                    text = msg["bytes"].decode("utf-8", "replace")
                if text is not None:
                    await session.handle_text(text)
        except (WebSocketDisconnect, RuntimeError):
            pass  # client went away
        finally:
            await session.close()

    if CARDS_DIR.is_dir():
        app.mount("/cards", StaticFiles(directory=CARDS_DIR), name="cards")

    if DIST_DIR.is_dir() and (DIST_DIR / "index.html").is_file():
        index = DIST_DIR / "index.html"
        if (DIST_DIR / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith(("api/", "ws", "cards/")):
                raise HTTPException(404)
            f = (DIST_DIR / path).resolve()
            if path and f.is_file() and DIST_DIR.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(index)

    return app


app = create_app()
