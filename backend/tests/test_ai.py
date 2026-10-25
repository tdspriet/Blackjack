"""Tests for onebj.ai: features, model, agent interface (SPEC §5 / frontend protocol), checkpoints,
data generation and a tiny end-to-end training run."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
from onebj.agents.base import Agent
from onebj.ai.agent import BetConfig, NNAgent, list_checkpoints, load_training_log
from onebj.ai.datagen import GenConfig, Generator, deal_mc, tables_for
from onebj.ai.features import KIND_BET, KIND_INSURANCE, KIND_PLAY, FeatureEncoder, feature_names
from onebj.ai.model import BET_LABELS, PLAY_LABELS, BlackjackNet, NumpyNet
from onebj.ai.oracle import CashSolver, composition_solver
from onebj.ai.train import TrainConfig, train
from onebj.analytics import crazy_7_ev, perfect_pairs_ev, twenty_one_plus_three_ev
from onebj.config import load_config
from onebj.engine import Action, Bets, Phase, Table
from onebj.sim import play_round
from onebj.strategy.solver import Solver, SolverRules


# ---------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def ckpt_dir(tmp_path_factory, cfg):
    """A tiny model trained end to end (doubles as the training smoke test, a few seconds)."""
    d = tmp_path_factory.mktemp("ckpt")
    tc = TrainConfig(generations=[{"play": 2500, "bet": 800}, {"play": 1200, "bet": 400}],
                     targets_per_generation=["solver", "blend"], steps_per_generation=[60, 30],
                     batch_size=256, eval_rounds=2000, bench_states=600, log_every=10, chunks=1,
                     gen={"k_play": 1, "k_bet": 1})
    meta = train("tiny", tc, cfg, d, workers=1, verbose=False)
    assert (d / "tiny.pt").exists() and (d / "tiny.json").exists()
    assert meta["metrics"]["steps"] == 90
    return d


@pytest.fixture()
def agent(ckpt_dir, cfg):
    return NNAgent.load("tiny", cfg, dir=ckpt_dir)


def _random_states(cfg, n_rounds=150, seed=3):
    """Yield (table, phase) at every decision of rounds played with random legal actions."""
    t = Table(cfg, seed=seed)
    rng = random.Random(seed)
    for _ in range(n_rounds):
        yield t, Phase.BETTING
        t.place_bets(Bets(main=rng.choice([1, 5, 10])))
        while t.phase in (Phase.PLAYER, Phase.INSURANCE):
            yield t, t.phase
            t.act(rng.choice(t.legal_actions()))


# ---------------------------------------------------------------------- features
def test_feature_names_and_size():
    enc = FeatureEncoder()
    names = feature_names()
    assert names == enc.names and len(names) == enc.size == 61
    assert len(set(names)) == len(names)
    assert names[enc.i_cash] == "cashout_value"


def test_obs_and_table_paths_agree(cfg):
    enc = FeatureEncoder()
    hs = 208
    for t, ph in _random_states(cfg):
        kind = {Phase.BETTING: KIND_BET, Phase.INSURANCE: KIND_INSURANCE, Phase.PLAYER: KIND_PLAY}[ph]
        a = enc.encode_obs(t.observation(), kind, hs)
        b = enc.encode_table(t, kind)
        assert a.dtype == np.float32 and a.shape == (61,)
        np.testing.assert_array_equal(a, b)
        assert np.all(np.isfinite(a)) and np.all(np.abs(a) <= 2.0 + 1e-6)
        if kind == KIND_PLAY:
            np.testing.assert_array_equal(b, enc.encode_play_state(t, t.decision_state()))
        if kind == KIND_BET:  # nothing about a hand or dealer before the deal
            assert a[enc.i_total] == 0 and a[enc.i_up0:enc.i_up0 + 10].sum() == 0 and a[enc.i_pbet] == 1


def test_features_never_see_the_hole_card(cfg):
    """Changing the hole card must not change any feature (only public information is encoded)."""
    enc = FeatureEncoder()
    for t, ph in _random_states(cfg, 60, seed=11):
        if ph is not Phase.PLAYER or t.hole_revealed:
            continue
        x = enc.encode_table(t, KIND_PLAY)
        u = t.clone(resample_unseen=True, seed=5)
        assert enc.encode_table(u, KIND_PLAY).tolist() == x.tolist()


# ---------------------------------------------------------------------- model
def test_numpy_mirror_matches_torch():
    torch.manual_seed(0)
    m = BlackjackNet(61, 34)
    m.bet_offset.fill_(0.1)
    m.bet_scale.fill_(0.5)
    x = torch.randn(32, 61)
    q, b = m(x)
    net = NumpyNet(NumpyNet.export(m))
    np.testing.assert_allclose(net.q(x.numpy()), q.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(net.bet(x.numpy()), b.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(q[:, 4].detach().numpy(), x[:, 34].numpy())  # cashout wired to its input
    acts, q1, b1 = net.full(x[0].numpy())
    np.testing.assert_allclose(q1, q[0].detach().numpy(), atol=1e-5)
    assert [a.shape for a in acts] == [(128,), (64,)]


# ---------------------------------------------------------------------- agent: decisions
def test_agent_protocol_and_legality(agent, cfg):
    assert isinstance(agent, Agent)
    agent.trace = False
    for t, ph in _random_states(cfg, 200, seed=21):
        obs = t.observation()
        if ph is Phase.PLAYER:
            a, tr = agent.decide_action(obs)
            assert tr is None and a in obs.legal_actions
            assert agent.fast_action(t, t.decision_state()) == a.value
        elif ph is Phase.INSURANCE:
            take, _ = agent.decide_insurance(obs)
            assert agent.fast_insurance(t) == take
        else:
            bets, _ = agent.decide_bets(obs)
            assert agent.fast_bets(t) == bets
            lim = cfg.limits
            assert lim.main_min <= bets.main <= lim.main_max
            for name in ("perfect_pairs", "twenty_one_plus_three", "crazy_7", "bust_bonus"):
                v = getattr(bets, name)
                assert v == 0 or lim.side_min <= v <= lim.side_max


def test_deterministic_load_and_play(ckpt_dir, cfg):
    a1 = NNAgent.load("tiny", cfg, dir=ckpt_dir, trace=False)
    a2 = NNAgent.load(str(ckpt_dir / "tiny.pt"), cfg, trace=False)
    t1, t2 = Table(cfg, seed=9, record_events=False), Table(cfg, seed=9, record_events=False)
    for _ in range(300):
        assert play_round(t1, a1).net == play_round(t2, a2).net
    assert t1.balance == t2.balance


def test_bet_sizing(agent):
    agent.bet_cfg = BetConfig(unit=10, max_units=12, kelly_fraction=0.5, main_variance=1.33, bankroll=1000)
    pred = np.zeros(6, np.float32)
    pred[0] = -0.01
    pred[1:5] = -0.05
    bets, info = agent.size_bets(pred, 1000)
    assert bets.main == 10 and bets.total == 10
    pred[0] = 0.02  # Kelly 0.5 * 0.02 / 1.33 * 1000 = 7.5 < unit → unit
    assert agent.size_bets(pred, 1000)[0].main == 10
    pred[0] = 0.5  # huge edge → capped at 12 units
    assert agent.size_bets(pred, 1000)[0].main == 120
    pred[3] = 0.2  # Crazy 7 predicted +EV → table minimum
    b, info = agent.size_bets(pred, 1000)
    assert b.crazy_7 == 1 and b.perfect_pairs == 0 and "crazy_7" in info["side_kelly_stakes"]


# ---------------------------------------------------------------------- traces & describe (frontend protocol)
def _check_trace(tr, desc, kind):
    assert set(tr) >= {"kind", "features", "layers", "outputs", "chosen", "reference", "extra"}
    assert tr["kind"] == kind
    json.dumps(tr)  # JSON-able
    names = desc["layers"][0]["labels"]
    assert [f["name"] for f in tr["features"]] == names
    assert all(set(f) == {"name", "value", "raw"} and isinstance(f["value"], float) for f in tr["features"])
    known = {l["name"]: l["size"] for l in desc["layers"]} | {h["name"]: h["size"] for h in desc["heads"]}
    for layer in tr["layers"]:
        assert layer["name"] in known and layer["size"] == known[layer["name"]] == len(layer["activations"])
    assert [l["name"] for l in tr["layers"][:3]] == ["input", "trunk.0", "trunk.1"]
    assert tr["layers"][0]["activations"] == [f["value"] for f in tr["features"]]
    out = tr["outputs"]
    assert len(out["labels"]) == len(out["values"]) == len(out["legal"]) and isinstance(out["unit"], str)
    for k in ("basic_strategy", "hilo_deviation"):
        assert k in tr["reference"]
    for v in tr["reference"].values():
        assert v is None or isinstance(v, str)


def test_traces_match_protocol(agent, cfg):
    desc = agent.describe()
    seen = set()
    for t, ph in _random_states(cfg, 250, seed=5):
        obs = t.observation()
        if ph is Phase.PLAYER:
            a, tr = agent.decide_action(obs)
            _check_trace(tr, desc, "play")
            out = tr["outputs"]
            assert out["labels"] == PLAY_LABELS
            assert out["legal"] == [lab in {x.value for x in obs.legal_actions} for lab in PLAY_LABELS]
            assert all((v is None) == (not ok) for v, ok in zip(out["values"], out["legal"]))
            assert tr["chosen"] == a.value and out["legal"][PLAY_LABELS.index(a.value)]
            assert tr["layers"][-1]["name"] == "play"
            legal_vals = [v for v in out["values"] if v is not None]
            assert out["values"][PLAY_LABELS.index(a.value)] == max(legal_vals)
            if obs.hand.cashout_offer is not None:
                assert abs(out["values"][4] - (obs.hand.cashout_offer / obs.hand.bet - 1)) < 1e-3
        elif ph is Phase.INSURANCE:
            take, tr = agent.decide_insurance(obs)
            _check_trace(tr, desc, "insurance")
            assert tr["outputs"]["labels"] == ["insure", "no_insurance"]
            assert tr["chosen"] == ("insure" if take else "no_insurance")
            assert tr["layers"][-1]["name"] == "bet"
        else:
            bets, tr = agent.decide_bets(obs)
            _check_trace(tr, desc, "bet")
            assert tr["outputs"]["labels"] == BET_LABELS
            ex = tr["extra"]
            assert set(ex["predicted_ev"]) == set(BET_LABELS)
            assert ex["bets"] == bets.to_dict() and isinstance(ex["bet_fraction"], float)
            assert tr["layers"][-1]["name"] == "bet"
        seen.add(ph)
    assert seen == {Phase.BETTING, Phase.PLAYER, Phase.INSURANCE}


def test_describe_schema(agent):
    d = agent.describe()
    json.dumps(d)
    assert d["name"] == "tiny" and d["checkpoint"].endswith("tiny.pt")
    sizes = {l["name"]: l["size"] for l in d["layers"]}
    assert list(sizes) == ["input", "trunk.0", "trunk.1"] and sizes["input"] == 61
    assert len(d["layers"][0]["labels"]) == 61
    heads = {h["name"]: h for h in d["heads"]}
    assert heads["play"]["labels"] == PLAY_LABELS and heads["bet"]["labels"] == BET_LABELS
    sizes.update({h["name"]: h["size"] for h in d["heads"]})
    assert set(d["edges"]) == {"input->trunk.0", "trunk.0->trunk.1", "trunk.1->play", "trunk.1->bet"}
    for key, edges in d["edges"].items():
        a, b = key.split("->")
        assert 0 < len(edges) <= 400
        for i, j, w in edges:
            assert 0 <= i < sizes[a] and 0 <= j < sizes[b] and isinstance(w, float)
        assert all(j != 4 for _, j, _ in d["edges"]["trunk.1->play"])  # cashout output has no weights
    assert "metrics" in d["training"]


def test_list_checkpoints_and_training_log(ckpt_dir):
    cks = list_checkpoints(str(ckpt_dir))
    assert [c["name"] for c in cks] == ["tiny"]
    c = cks[0]
    assert set(c) == {"name", "path", "created", "metrics"} and c["path"].endswith("tiny.pt")
    assert isinstance(c["metrics"]["eval_ev"], float)
    log = load_training_log("tiny", ckpt_dir)
    assert log and all("step" in r for r in log)
    assert any("loss" in r for r in log) and any(isinstance(r.get("eval_ev"), float) for r in log)
    assert list_checkpoints(str(ckpt_dir / "nope")) == []


def test_set_trace_toggle(agent, cfg):
    t = Table(cfg, seed=1)
    agent.set_trace(False)
    assert agent.decide_bets(t.observation())[1] is None
    agent.set_trace(True)
    assert agent.decide_bets(t.observation())[1] is not None


# ---------------------------------------------------------------------- data generation & oracle
def test_deal_mc_matches_exact_side_bet_evs(cfg):
    tabs = tables_for(cfg)
    unseen = [8] * 52
    unseen[24:28] = [12, 12, 12, 12]  # 7-rich composition
    rng = np.random.default_rng(0)
    acc = np.zeros(3)
    reps = 400
    for _ in range(reps):
        pp, t213, c7, _ins, n = deal_mc(unseen, tabs, 10, rng)
        acc += (pp, t213, c7)
    acc /= reps  # ~350k deals: SE ≈ 0.004 (PP, 21+3), 0.02 (Crazy 7)
    sb = cfg.side_bets
    assert abs(acc[0] - perfect_pairs_ev(unseen, sb.perfect_pairs)) < 0.015
    assert abs(acc[1] - twenty_one_plus_three_ev(unseen, sb.twenty_one_plus_three)) < 0.015
    assert abs(acc[2] - crazy_7_ev(unseen, sb.crazy_7)) < 0.07


def test_generator_shapes_and_masks(cfg):
    for targets in ("mc", "solver", "blend"):
        g = Generator(cfg, None, GenConfig(targets=targets, k_play=1, k_bet=1), seed=4)
        d = g.run(200, 60)
        assert d["Xp"].shape == (200, 61) and d["Yp"].shape == (200, 4)
        assert d["Mp"][:, :2].all()  # stand and hit always legal
        assert (d["Yp"][~d["Mp"]] == 0).all()
        bet_rows = d["Mb"][:, 0]
        assert bet_rows.sum() >= 50 and d["Mb"][bet_rows].all()
        assert (d["Mb"][~bet_rows] == [False] * 5 + [True]).all()  # insurance rows
        assert np.isfinite(d["Yp"]).all() and np.isfinite(d["Yb"]).all()


def test_cash_solver(cfg):
    rules = SolverRules.from_config(cfg.rules)
    p = tuple([1 / 13] * 9 + [4 / 13])
    plain, cash = Solver(rules, p), CashSolver(rules, p, None)
    assert plain.evs(16, False, 2, 9) == cash.evs(16, False, 2, 9)
    full = tuple([32] * 9 + [128])
    s = composition_solver(rules, full, cfg.rules.cashout_rtp)
    e_cash, e_plain = s.evs(12, False, 2, 9), plain.evs(12, False, 2, 9)
    assert e_cash["H"] >= e_plain["H"] - 1e-12 and e_cash["S"] == pytest.approx(e_plain["S"])
    assert s.split_ev(7, 9) >= plain.split_ev(7, 9) - 1e-12


def test_quick_training_cli(tmp_path):
    """`python -m onebj.ai.train --quick` end to end (multiprocess-free, a few seconds)."""
    from onebj.ai.train import main
    main(["--name", "cli", "--quick", "--out", str(tmp_path), "--workers", "1", "--generations", "1",
          "--scale", "0.2", "--steps", "20"])
    meta = json.loads((tmp_path / "cli.json").read_text())
    assert meta["name"] == "cli" and len(meta["feature_names"]) == 61 and meta["config_hash"]
    a = NNAgent.load("cli", dir=tmp_path)
    t = Table(load_config(), seed=2, record_events=False)
    for _ in range(50):
        play_round(t, a)
    assert isinstance(a.describe()["edges"], dict)
    assert Action.CASHOUT.value in PLAY_LABELS
