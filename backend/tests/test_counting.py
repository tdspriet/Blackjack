from onebj.config import load_config
from onebj.counting import SYSTEMS, CountTracker, counts_from_seen, hilo_true, ko_irc
from onebj.engine import Bets, Phase, Table


def test_systems_balanced():
    for name, tags in SYSTEMS.items():
        per_deck = sum(t * (16 if i == 9 else 4) for i, t in enumerate(tags))
        assert per_deck == (4 if name == "ko" else 0), name


def test_counts_from_seen_basic():
    seen = [0] * 10
    seen[4] = 3  # three 5s
    seen[9] = 1  # one ten
    rc, tc = counts_from_seen(seen)
    assert rc["hilo"] == 2 and rc["omega2"] == 4 and rc["zen"] == 4 and rc["wong_halves"] == 3.5
    assert rc["ko"] == 2 + ko_irc(8)
    assert abs(tc["hilo"] - 2 / (412 / 52)) < 1e-12
    assert abs(hilo_true(seen) - tc["hilo"]) < 1e-12


def test_tracker_from_events_matches_table_and_never_counts_hidden_hole():
    t = Table(load_config(), seed=8)
    tr = CountTracker()
    for _ in range(400):
        t.place_bets(Bets(main=1))
        # mid-round: the hole card must not be counted yet
        tr_mid = CountTracker()
        tr_mid.seen = tr.seen[:]
        tr_mid.apply_events(t.events)
        if t.phase is Phase.PLAYER:
            assert tr_mid.seen == t.shoe.seen_bucket
            assert sum(t.shoe.seen_bucket) + t.shoe.cards_unseen == 416
        while t.phase in (Phase.INSURANCE, Phase.PLAYER):
            t.act("no_insurance" if t.phase is Phase.INSURANCE else "stand")
        tr.apply_events(t.events)
        assert tr.seen == t.shoe.seen_bucket
        assert tr.running() == counts_from_seen(t.shoe.seen_bucket)[0]
    assert t.shoe.shoe_number > 2


def test_history_resets_only_on_reshuffle():
    t = Table(load_config(patch={"crowd": {"enabled": False}}), seed=1)
    lens = []
    for _ in range(150):
        t.place_bets(Bets(main=1))
        while t.phase in (Phase.INSURANCE, Phase.PLAYER):
            t.act("no_insurance" if t.phase is Phase.INSURANCE else "stand")
        kinds = [e.get("kind") for e in t.events if e["type"] == "shuffle"]
        if "half_switch" in kinds:
            assert len(t.history) > 20  # history continues across the half switch
        if "reshuffle" in kinds:
            assert t.history == []
        lens.append(len(t.history))
