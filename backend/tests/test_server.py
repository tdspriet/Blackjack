"""Server tests: REST endpoints and the WebSocket protocol (SPEC §6) through FastAPI's TestClient."""

from __future__ import annotations

import json
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from onebj.server import agents as registry
from onebj.server import session as session_mod
from onebj.server.app import create_app

Msg = dict[str, Any]

METRIC_KEYS = {"rounds", "net", "wagered", "ev_per_round", "edge", "wins", "losses", "pushes", "blackjacks",
               "charlies", "bankroll", "side_bets", "agreement_basic", "max_drawdown", "rounds_per_sec"}
GAME_KEYS = {"round_id", "phase", "balance", "total_bet", "bets", "dealer", "hands", "active_hand", "crowd",
             "legal_actions", "side_bet_results", "round_net", "messages"}
SHOE_KEYS = {"shoe_number", "half", "half_size", "dealt_in_half", "cut_at", "burned", "cards_unseen",
             "decks_unseen", "seen_counts", "unseen_counts", "running", "true", "analytics", "history"}


@pytest.fixture(autouse=True)
def fast_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_mod, "INTERACTIVE_FRAME_DELAY", 0.0)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(debug=True, checkpoints=tmp_path))


def recv_until(ws: Any, pred: Callable[[Msg], bool], limit: int = 2000) -> list[Msg]:
    """Receive messages until ``pred(msg)``; returns all of them (the matching one last)."""
    out = []
    for _ in range(limit):
        m = ws.receive_json()
        out.append(m)
        if pred(m):
            return out
    raise AssertionError(f"condition not met within {limit} messages; last: {str(out[-1])[:300]}")


def is_decision(m: Msg) -> bool:
    return m["type"] == "state" and bool(m["game"]["legal_actions"])


def settle(ws: Any) -> list[Msg]:
    return recv_until(ws, lambda m: is_decision(m) and m["game"]["legal_actions"] == ["deal"])


def connect(client: TestClient, seed: int = 1) -> tuple[Any, Msg, Msg]:
    ctx = client.websocket_connect(f"/ws?seed={seed}")
    ws = ctx.__enter__()
    hello = ws.receive_json()
    state = ws.receive_json()
    ws._ctx = ctx  # keep the context manager alive
    return ws, hello, state


def crowd_off(ws: Any) -> None:
    ws.send_json({"cmd": "set_config", "patch": {"crowd": {"enabled": False}}})
    hello = ws.receive_json()
    assert hello["type"] == "hello" and hello["config"]["crowd"]["enabled"] is False
    st = ws.receive_json()
    assert st["game"]["crowd"]["enabled"] is False


def deal(ws: Any, cards: list[str] | None = None, **bets: float) -> list[Msg]:
    if cards:
        ws.send_json({"cmd": "debug_rig", "cards": cards})
    ws.send_json({"cmd": "place_bets", "bets": {"main": 10, **bets}})
    return recv_until(ws, is_decision)


def codes(cards: list[Msg]) -> list[str]:
    return [c["code"] for c in cards]


# ---------------------------------------------------------------------- REST
def test_rest_endpoints(client: TestClient, tmp_path: Path) -> None:
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["agents"] == ["hilo", "basic"]
    cfg = client.get("/api/config").json()
    assert cfg["shoe"]["num_decks"] == 8 and cfg["crowd"]["enabled"] is True
    assert client.get("/api/checkpoints").json() == []
    assert client.get("/api/training/nope").status_code == 404
    assert client.get("/api/training/..%2Fx").status_code in (400, 404)
    (tmp_path / "v1.json").write_text(json.dumps({"name": "v1", "training_log": [{"step": 1, "loss": 0.5}]}))
    (tmp_path / "v2.json").write_text(json.dumps([{"step": 2, "loss": 0.4, "eval_ev": -0.01}]))
    assert client.get("/api/training/v1").json() == [{"step": 1, "loss": 0.5}]
    assert client.get("/api/training/v2").json()[0]["eval_ev"] == -0.01
    r = client.get("/cards/A_of_hearts.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"


# ---------------------------------------------------------------------- hello / state shape
def test_hello_and_initial_state(client: TestClient) -> None:
    ws, hello, st = connect(client)
    assert hello["type"] == "hello"
    assert hello["network"] is None  # no checkpoint yet
    assert hello["agents"] == ["hilo", "basic"]
    assert hello["config"]["limits"]["chips"] == [1, 2, 5, 10, 25, 100, 500, 1000]
    assert st["type"] == "state"
    assert GAME_KEYS <= set(st["game"]) and SHOE_KEYS <= set(st["shoe"]) and METRIC_KEYS <= set(st["metrics"])
    assert st["game"]["phase"] == "betting" and st["game"]["legal_actions"] == ["deal"]
    assert st["game"]["balance"] == 1000
    assert st["ai"] == {"agent": "hilo", "advice": None, "last_trace": None, "autoplay": False}
    assert set(st["shoe"]["analytics"]["p_dealer_bust_by_upcard"]) == set("A23456789T")
    assert st["metrics"]["bankroll"] == [[0, 1000.0]]
    ws.close()


# ---------------------------------------------------------------------- manual play
def test_manual_round_streams_each_card(client: TestClient) -> None:
    ws, _, _ = connect(client, seed=3)
    msgs = deal(ws, ["10_of_hearts", "6_of_spades", "5_of_diamonds", "10_of_clubs"], perfect_pairs=1, bust_bonus=1)
    # round start, P1, up, P2, then the decision state (the hole-card frame is identical to it)
    assert len(msgs) == 5
    assert [len(m["game"]["hands"][0]["cards"]) for m in msgs] == [0, 1, 1, 2, 2]
    assert [len(m["game"]["dealer"]["cards"]) for m in msgs] == [0, 0, 1, 1, 2]
    assert all(m["game"]["legal_actions"] == [] for m in msgs[:-1])
    dec = msgs[-1]["game"]
    assert dec["phase"] == "player" and set(dec["legal_actions"]) >= {"hit", "stand", "double", "cashout"}
    assert codes(dec["hands"][0]["cards"]) == ["10_of_hearts", "5_of_diamonds"]
    assert dec["dealer"]["cards"][1] == {"face_up": False, "code": "back", "owner": "dealer"}
    assert dec["dealer"]["total"] == 6 and dec["hands"][0]["cashout_offer"] > 0
    assert dec["balance"] == 1000 - 12 and dec["total_bet"] == 12
    assert dec["side_bet_results"] == [{"bet": "perfect_pairs", "outcome": "lose", "odds": 0, "stake": 1, "payout": 0}]

    ws.send_json({"cmd": "action", "action": "stand"})
    msgs = settle(ws)
    frames, final = msgs[:-1], msgs[-1]
    phases = [m["game"]["phase"] for m in frames]
    assert "dealer" in phases and all(m["game"]["legal_actions"] == [] for m in frames)
    n_cards = [len(m["game"]["dealer"]["cards"]) + len(m["game"]["crowd"]["cards"]) for m in msgs]
    assert n_cards == sorted(n_cards)  # cards only ever get added
    g = final["game"]
    assert g["phase"] == "settled" and g["round_net"] is not None
    assert g["dealer"]["cards"][1]["code"] == "10_of_clubs"
    assert all(c["owner"] == "crowd" for c in g["crowd"]["cards"])
    assert g["balance"] == pytest.approx(1000 + g["round_net"])
    m = final["metrics"]
    assert m["rounds"] == 1 and m["net"] == g["round_net"] and m["agreement_basic"] == 1.0
    assert m["bankroll"][-1] == [1, g["balance"]]
    assert final["shoe"]["history"][-1]["round"] == 1 and "half" in final["shoe"]["history"][-1]
    ws.close()


def test_insurance_flow(client: TestClient) -> None:
    ws, _, _ = connect(client)
    msgs = deal(ws, ["10_of_hearts", "A_of_spades", "9_of_clubs", "K_of_diamonds"])
    g = msgs[-1]["game"]
    assert g["phase"] == "insurance" and g["legal_actions"] == ["insure", "no_insurance"]
    ws.send_json({"cmd": "action", "action": "hit"})
    assert ws.receive_json()["type"] == "error"
    ws.receive_json()  # re-sync state
    ws.send_json({"cmd": "action", "action": "insure"})
    g = settle(ws)[-1]["game"]
    assert g["bets"]["insurance"] == 5 and g["dealer"]["blackjack"] is True
    assert g["round_net"] == 0  # main -10, insurance +10
    assert g["hands"][0]["result"] == "lose"
    assert {"bet": "insurance", "outcome": "dealer_blackjack", "odds": 2.0, "stake": 5.0, "payout": 15.0} \
        in g["side_bet_results"]
    ws.close()


def test_split_double_flow(client: TestClient) -> None:
    ws, _, _ = connect(client)
    crowd_off(ws)
    deal(ws, ["8_of_hearts", "6_of_spades", "8_of_diamonds", "10_of_clubs", "3_of_clubs", "9_of_spades",
              "K_of_hearts", "7_of_clubs"])
    ws.send_json({"cmd": "action", "action": "split"})
    msgs = recv_until(ws, is_decision)
    assert any(len(m["game"]["hands"]) == 2 and len(m["game"]["hands"][0]["cards"]) == 1 for m in msgs)
    g = msgs[-1]["game"]
    assert len(g["hands"]) == 2 and g["active_hand"] == 0
    assert codes(g["hands"][0]["cards"]) == ["8_of_hearts", "3_of_clubs"]
    assert "double" in g["legal_actions"] and "split" not in g["legal_actions"]
    ws.send_json({"cmd": "action", "action": "double"})
    g = recv_until(ws, is_decision)[-1]["game"]
    assert g["active_hand"] == 1 and g["hands"][0]["doubled"] and g["hands"][0]["total"] == 20
    assert codes(g["hands"][1]["cards"]) == ["8_of_diamonds", "K_of_hearts"]
    ws.send_json({"cmd": "action", "action": "stand"})
    g = settle(ws)[-1]["game"]
    assert g["dealer"]["bust"] and len(g["dealer"]["cards"]) == 3
    assert [h["result"] for h in g["hands"]] == ["win", "win"]
    assert g["round_net"] == 30 and g["crowd"]["cards"] == []
    ws.close()


def test_cashout(client: TestClient) -> None:
    ws, _, _ = connect(client)
    g = deal(ws, ["10_of_hearts", "10_of_spades", "6_of_diamonds", "9_of_clubs"])[-1]["game"]
    offer = g["hands"][0]["cashout_offer"]
    assert 0 < offer < 10
    ws.send_json({"cmd": "action", "action": "cashout"})
    final = settle(ws)[-1]
    g = final["game"]
    m = final["metrics"]
    assert m["cashouts"] == 1 and m["losses"] == 1 and m["wins"] == 0  # offer < stake counts as a loss
    assert g["hands"][0]["result"] == "cashout" and g["hands"][0]["payout"] == offer
    assert g["round_net"] == pytest.approx(offer - 10)
    ws.close()


def test_busy_while_streaming(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_mod, "INTERACTIVE_FRAME_DELAY", 0.15)
    ws, _, _ = connect(client)
    ws.send_json({"cmd": "place_bets", "bets": {"main": 10}})
    ws.receive_json()  # first frame; the rest is still queued
    ws.send_json({"cmd": "action", "action": "stand"})
    msgs = recv_until(ws, lambda m: m["type"] == "error")
    assert "wait" in msgs[-1]["message"].lower()
    recv_until(ws, is_decision)
    ws.close()


# ---------------------------------------------------------------------- AI
def test_ai_step_and_advice(client: TestClient) -> None:
    ws, _, _ = connect(client, seed=5)
    ws.send_json({"cmd": "advice", "on": True})
    st = ws.receive_json()
    adv = st["ai"]["advice"]
    assert adv["kind"] == "bet" and adv["extra"]["bets"]["main"] >= 1
    assert adv["outputs"]["labels"][0] == "main" and len(adv["outputs"]["values"]) == len(adv["outputs"]["labels"])
    ws.send_json({"cmd": "ai_step"})
    last = recv_until(ws, is_decision)[-1]
    trace = last["ai"]["last_trace"]
    assert trace["kind"] == "bet" and trace["extra"]["bets"]["main"] == last["game"]["bets"]["main"]
    for _ in range(12):
        if last["game"]["legal_actions"] == ["deal"]:
            break
        ws.send_json({"cmd": "ai_step"})
        last = recv_until(ws, is_decision)[-1]
        t = last["ai"]["last_trace"]
        assert t["kind"] in ("play", "insurance") and t["reference"]["basic_strategy"]
        if t["kind"] == "play":
            assert t["outputs"]["labels"] == ["stand", "hit", "double", "split", "cashout"]
            assert t["outputs"]["values"][1] is not None  # hit EV
            assert t["chosen"] in t["outputs"]["labels"]
        if last["game"]["phase"] == "player":
            assert last["ai"]["advice"]["kind"] == "play"
    assert last["game"]["legal_actions"] == ["deal"] and last["metrics"]["rounds"] == 1
    ws.close()


def test_autoplay_a_few_rounds(client: TestClient) -> None:
    ws, _, _ = connect(client)
    ws.send_json({"cmd": "set_agent", "agent": "basic"})
    assert ws.receive_json()["ai"]["agent"] == "basic"
    ws.send_json({"cmd": "autoplay", "on": True, "delay_ms": 0})
    st = ws.receive_json()
    assert st["ai"]["autoplay"] is True
    recv_until(ws, lambda m: m["type"] == "state" and m["metrics"]["rounds"] >= 3)
    ws.send_json({"cmd": "autoplay", "on": False})
    last = recv_until(ws, lambda m: m["type"] == "state" and m["ai"]["autoplay"] is False)[-1]
    assert last["ai"]["last_trace"] is not None
    m = last["metrics"]
    assert m["wins"] + m["losses"] + m["pushes"] >= 3 and len(m["bankroll"]) >= 4
    ws.close()


def test_simulate_progress_and_stop(client: TestClient) -> None:
    ws, _, _ = connect(client)
    ws.send_json({"cmd": "simulate", "rounds": 4000, "agent": "basic"})
    msgs = recv_until(ws, lambda m: m["type"] == "sim_progress" and m["finished"])
    prog = [m for m in msgs if m["type"] == "sim_progress"]
    assert prog[0]["done"] == 0 and prog[0]["total"] == 4000
    final = prog[-1]
    assert final["done"] == 4000 and final["metrics"]["rounds"] == 4000
    m = final["metrics"]
    assert METRIC_KEYS <= set(m) and -0.2 < m["edge"] < 0.2 and m["agreement_basic"] == 1.0
    assert m["bankroll"][0] == [0, 1000.0] and m["bankroll"][-1][0] == 4000
    assert m["rounds_per_sec"] > 0
    st = recv_until(ws, lambda x: x["type"] == "state")[-1]
    assert st["metrics"]["rounds"] == 0  # simulations don't touch the session

    ws.send_json({"cmd": "simulate", "rounds": 5_000_000, "agent": "hilo"})
    recv_until(ws, lambda m: m["type"] == "sim_progress" and m["done"] > 0)
    ws.send_json({"cmd": "stop"})
    final = recv_until(ws, lambda m: m["type"] == "sim_progress" and m["finished"])[-1]
    assert 0 < final["done"] < 5_000_000 and final["metrics"]["rounds"] == final["done"]
    ws.close()


def test_simulate_rejects_bad_requests(client: TestClient) -> None:
    ws, _, _ = connect(client)
    for bad in ({"rounds": 0}, {"rounds": "many"}, {"rounds": 10.5}, {"rounds": 10, "agent": "nn:nope"}):
        ws.send_json({"cmd": "simulate", **bad})
        assert ws.receive_json()["type"] == "error"
        assert ws.receive_json()["type"] == "state"
    ws.close()


# ---------------------------------------------------------------------- session commands
def test_invalid_commands_never_close_the_socket(client: TestClient) -> None:
    ws, _, _ = connect(client)
    ws.send_text("not json")
    assert ws.receive_json()["type"] == "error"
    for bad in ({"nocmd": 1}, {"cmd": 3}):
        ws.send_json(bad)
        assert ws.receive_json()["type"] == "error"
    bad_cmds = [
        {"cmd": "fly"},
        {"cmd": "action", "action": "stand"},  # nothing to decide
        {"cmd": "action", "action": "surrender"},
        {"cmd": "place_bets", "bets": {"main": 0}},
        {"cmd": "place_bets", "bets": {"main": 10, "lucky": 5}},
        {"cmd": "place_bets", "bets": {"main": 99999}},
        {"cmd": "place_bets", "bets": {"main": 10, "perfect_pairs": 9999}},
        {"cmd": "place_bets", "bets": {"main": -5}},
        {"cmd": "place_bets", "bets": "ten"},
        {"cmd": "set_agent", "agent": "skynet"},
        {"cmd": "set_config", "patch": {"crowd": {"colour": "red"}}},
        {"cmd": "set_config", "patch": {"shoe": {"penetration": 2}}},
        {"cmd": "autoplay", "on": "yes"},
        {"cmd": "reset", "bankroll": -1},
    ]
    for cmd in bad_cmds:
        ws.send_json(cmd)
        err = ws.receive_json()
        assert err["type"] == "error" and err["message"], cmd
        if cmd["cmd"] != "fly":
            assert ws.receive_json()["type"] == "state"
    # still fully usable
    g = deal(ws)[-1]["game"]
    assert g["round_id"] == 1
    ws.send_json({"cmd": "place_bets", "bets": {"main": 10}})  # round in progress
    assert ws.receive_json()["type"] == "error"
    ws.close()


def test_set_config_crowd_off(client: TestClient) -> None:
    ws, _, _ = connect(client)
    crowd_off(ws)
    deal(ws, ["10_of_hearts", "7_of_spades", "2_of_diamonds", "10_of_clubs"])
    ws.send_json({"cmd": "action", "action": "stand"})
    g = settle(ws)[-1]["game"]
    assert g["crowd"]["enabled"] is False and g["crowd"]["cards"] == [] and g["crowd"]["hitters"] == 0
    # rules change rebuilds the table but keeps the bankroll
    bal = g["balance"]
    ws.send_json({"cmd": "set_config", "patch": {"rules": {"dealer_hits_soft_17": True}}})
    assert ws.receive_json()["config"]["rules"]["dealer_hits_soft_17"] is True
    st = ws.receive_json()
    assert st["game"]["balance"] == bal and st["shoe"]["shoe_number"] == 1
    ws.close()


def test_new_shoe_reset_and_set_agent(client: TestClient) -> None:
    ws, _, st = connect(client)
    assert st["shoe"]["shoe_number"] == 1
    ws.send_json({"cmd": "new_shoe"})
    st = ws.receive_json()
    assert st["shoe"]["shoe_number"] == 2 and st["shoe"]["cards_unseen"] == 416
    deal(ws)
    ws.send_json({"cmd": "new_shoe"})  # mid-round
    assert ws.receive_json()["type"] == "error"
    ws.receive_json()
    ws.send_json({"cmd": "reset", "bankroll": 500})
    st = recv_until(ws, lambda m: m["type"] == "state" and m["game"]["phase"] == "betting")[-1]
    assert st["game"]["balance"] == 500 and st["metrics"]["rounds"] == 0 and st["metrics"]["bankroll"] == [[0, 500.0]]
    ws.send_json({"cmd": "set_agent", "agent": "basic"})
    assert ws.receive_json()["ai"]["agent"] == "basic"
    ws.close()


# ---------------------------------------------------------------------- NN integration (fake module)
class _FakeNN:
    name = "nn"

    def __init__(self) -> None:
        self.trace = True

    def describe(self) -> dict:
        return {"name": "fake", "checkpoint": "fake", "layers": [{"name": "input", "size": 3}, {"name": "h", "size": 2}],
                "heads": [{"name": "play", "size": 5, "from": "h"}], "edges": {"input->h": [[0, 1, 0.5]]}}

    def _trace(self, kind: str, chosen: str) -> dict | None:
        if not self.trace:
            return None
        return {"kind": kind, "features": [{"name": "x", "value": 0.1}], "chosen": chosen,
                "layers": [{"name": "input", "size": 3, "activations": [0.1, 0.2, 0.3]}],
                "outputs": {"labels": ["stand", "hit"], "values": [0.1, -0.1], "legal": [True, True]},
                "reference": {"basic_strategy": chosen}}

    def decide_bets(self, obs: Any) -> tuple:
        from onebj.engine import Bets
        return Bets(main=2), self._trace("bet", "2")

    def decide_insurance(self, obs: Any) -> tuple:
        return False, self._trace("insurance", "no_insurance")

    def decide_action(self, obs: Any) -> tuple:
        from onebj.engine import Action
        return Action.STAND, self._trace("play", "stand")


def test_nn_agent_integration(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.SimpleNamespace(
        list_checkpoints=lambda d: [{"name": "fake", "path": f"{d}/fake.pt", "created": "2026-10-25T00:00:00Z",
                                     "metrics": {"edge": 0.001}}],
        NNAgent=types.SimpleNamespace(load=lambda path, cfg: _FakeNN()))
    monkeypatch.setattr(registry, "_ai_module", lambda: fake)
    assert client.get("/api/checkpoints").json() == [{"name": "fake", "created": "2026-10-25T00:00:00Z",
                                                      "metrics": {"edge": 0.001}}]
    ws, hello, st = connect(client)
    assert hello["agents"] == ["nn:fake", "hilo", "basic"] and hello["network"]["name"] == "fake"
    assert st["ai"]["agent"] == "nn:fake"
    ws.send_json({"cmd": "ai_step"})
    last = recv_until(ws, is_decision)[-1]
    assert last["game"]["bets"]["main"] == 2 and last["ai"]["last_trace"]["extra"]["bets"]["main"] == 2
    ws.send_json({"cmd": "set_agent", "agent": "hilo"})
    st = ws.receive_json()  # network unchanged → no new hello
    assert st["type"] == "state" and st["ai"]["agent"] == "hilo" and st["ai"]["last_trace"] is None
    ws.close()
