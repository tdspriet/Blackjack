from onebj.agents import Agent, BasicStrategyAgent, HiLoAgent
from onebj.config import load_config
from onebj.engine import Table
from onebj.sim import Metrics, make_agent, play_round, run_rounds, simulate


def test_agents_satisfy_protocol_and_generic_path():
    cfg = load_config()
    for a in (BasicStrategyAgent(cfg), HiLoAgent(cfg)):
        assert isinstance(a, Agent)

    class Slow:  # no fast path: exercises the Observation-based protocol
        def __init__(self, inner):
            self.inner = inner
            self.name = "slow"

        def decide_bets(self, obs):
            return self.inner.decide_bets(obs)

        def decide_insurance(self, obs):
            return self.inner.decide_insurance(obs)

        def decide_action(self, obs):
            return self.inner.decide_action(obs)

    for inner in (BasicStrategyAgent(cfg), HiLoAgent(cfg)):
        a, b = Table(cfg, seed=4), Table(cfg, seed=4)
        slow = Slow(inner)
        for _ in range(300):
            ra, rb = play_round(a, slow), play_round(b, inner)
            assert ra.net == rb.net


def test_hilo_traces_json():
    import json
    cfg = load_config()
    t = Table(cfg, seed=1)
    ag = HiLoAgent(cfg)
    bets, tr = ag.decide_bets(t.observation())
    t.place_bets(bets)
    json.dumps(tr)
    while t.phase.value in ("insurance", "player"):
        if t.phase.value == "insurance":
            take, tr = ag.decide_insurance(t.observation())
            t.act("insure" if take else "no_insurance")
        else:
            a, tr = ag.decide_action(t.observation())
            json.dumps(tr)
            t.act(a)


def test_simulate_multiprocess_and_deterministic():
    cfg = load_config(patch={"crowd": {"enabled": False}})
    m1 = simulate(cfg, "basic", 4000, seed=3, workers=2)
    m2 = simulate(cfg, "basic", 4000, seed=3, workers=2)
    assert m1["rounds"] == 4000 and m1["net"] == m2["net"]
    assert m1["wins"] + m1["losses"] + m1["pushes"] >= 4000
    assert -0.2 < m1["edge"] < 0.2


def test_run_rounds_progress():
    cfg = load_config()
    calls = []
    m = run_rounds(Table(cfg, seed=1, record_events=False), make_agent("hilo", cfg), 2000,
                   progress=lambda mm: calls.append(mm.rounds), progress_every=500, bankroll_every=100)
    d = m.to_dict()
    assert calls == [500, 1000, 1500, 2000] and len(d["bankroll"]) == 20
    assert set(d["side_bets"]) == {"perfect_pairs", "twenty_one_plus_three", "crazy_7", "bust_bonus"}
    assert isinstance(Metrics(), Metrics)
