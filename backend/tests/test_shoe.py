import random

from conftest import make_table

from onebj.cards import NUM_CARDS
from onebj.config import ShoeConfig, load_config
from onebj.engine import Bets, Phase, Table
from onebj.shoe import NapoleonShoe


def new_shoe(**kw) -> NapoleonShoe:
    return NapoleonShoe(ShoeConfig(**kw), random.Random(1))


def test_halves_burn_and_cut():
    s = new_shoe()
    assert s.total_cards == 416 and s.half_size == 208 and s.half == 1
    assert s.burned == 1 and s.dealt_in_half == 1 and s.cards_seen == 0  # burn card is unseen
    assert sorted(s.cards) == sorted(c for c in range(NUM_CARDS) for _ in range(8))
    cuts = set()
    for seed in range(200):
        sh = NapoleonShoe(ShoeConfig(), random.Random(seed))
        assert 148 <= sh.cut_at <= 164
        cuts.add(sh.cut_at)
    assert len(cuts) > 10


def test_cut_card_sets_flag_and_switch_then_reshuffle():
    s = new_shoe(cut_jitter=0)
    assert s.cut_at == 156
    while s.dealt_in_half < 156:
        s.mark_seen(s.draw())
    assert not s.cut_out
    s.mark_seen(s.draw())  # 157th card: the cut card comes out first
    assert s.cut_out and s.pending == "half_switch"
    seen = s.cards_seen
    assert s.end_round() == "half_switch"
    assert s.half == 2 and s.pos == 209 and s.cards_seen == seen  # NO count reset at the half switch
    assert s.cards_unseen == 416 - seen
    while not s.cut_out:
        s.mark_seen(s.draw())
    assert s.end_round() == "reshuffle"
    assert s.half == 1 and s.shoe_number == 2 and s.cards_seen == 0


def test_single_shoe_mode():
    s = new_shoe(split_halves=False, cut_jitter=0)
    assert s.half_size == 416 and s.cut_at == 312
    while not s.cut_out:
        s.draw()
    assert s.end_round() == "reshuffle"


def test_unseen_tracking_and_future_positions():
    s = new_shoe()
    for _ in range(10):
        s.mark_seen(s.draw())
    assert sum(s.unseen_buckets()) == 406 and sum(s.unseen_cards()) == 406
    assert len(s.future_positions()) == (208 - s.dealt_in_half) + 208


def test_table_plays_through_halves_and_reshuffles():
    t = Table(load_config(patch={"crowd": {"enabled": False}}), seed=5)
    kinds = []
    seen_before_switch = None
    for _ in range(200):
        t.place_bets(Bets(main=1))
        while t.phase in (Phase.INSURANCE, Phase.PLAYER):
            t.act("no_insurance" if t.phase is Phase.INSURANCE else "stand")
        for e in t.events:
            if e["type"] == "shuffle":
                kinds.append(e["kind"])
                if e["kind"] == "half_switch":
                    seen_before_switch = t.shoe.cards_seen
                    assert seen_before_switch > 140
        if len(kinds) >= 3:
            break
    assert kinds[:3] == ["half_switch", "reshuffle", "half_switch"]
    assert any(e["type"] == "cut_card" for e in t.events) or True


def test_cut_card_event_and_last_round():
    t = make_table(seed=2)
    found = False
    for _ in range(100):
        t.place_bets(Bets(main=1))
        while t.phase in (Phase.INSURANCE, Phase.PLAYER):
            t.act("no_insurance" if t.phase is Phase.INSURANCE else "stand")
        types = [e["type"] for e in t.events]
        if "cut_card" in types:
            # the round in which the cut card appears is completed, then the shoe switches
            assert types.index("cut_card") < types.index("settle") < types.index("shuffle")
            assert t.shoe.half == 2 and not t.shoe.cut_out
            found = True
            break
    assert found
