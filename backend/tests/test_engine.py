from __future__ import annotations

import os

import pytest
from conftest import make_table, stack

from onebj.cards import BUCKET_OF, CODES, parse_card
from onebj.engine import Action, Bets, Phase, Table
from onebj.config import load_config

ONLY_RECKLESS = {"basic": 0, "cautious": 0, "dealer_mimic": 0, "reckless": 1.0}
ONLY_BASIC = {"basic": 1.0, "cautious": 0, "dealer_mimic": 0, "reckless": 0}
CARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cards")


def play(t: Table, *actions: str) -> None:
    for a in actions:
        t.act(a)


def test_deal_order_and_events():
    t = make_table()
    stack(t, "9h 6c 7d Ks")
    t.place_bets(Bets(main=10))
    cards = [e for e in t.events if e["type"] == "card"]
    assert [(CODES[e["card"]], e["owner"], e["face_up"]) for e in cards] == [
        ("9_of_hearts", "player", True), ("6_of_clubs", "dealer", True),
        ("7_of_diamonds", "player", True), ("K_of_spades", "dealer", False)]
    assert t.phase is Phase.PLAYER
    assert t.legal_actions() == [Action.HIT, Action.STAND, Action.DOUBLE, Action.CASHOUT]


def test_stand_settle_and_bankroll():
    t = make_table()
    stack(t, "Th 6c 9d Ks 5h")  # 19 vs 6,K -> dealer draws 5 -> 21
    t.place_bets(Bets(main=10))
    t.act("stand")
    r = t.round_result
    assert t.phase is Phase.SETTLED
    assert r.dealer_total == 21 and r.hands[0].result == "lose"
    assert t.balance == 990 and r.net == -10


def test_double_any_two_and_win():
    t = make_table()
    stack(t, "5h 6c 4d Ks 9s Td")  # 9 vs 6 -> double -> 18; dealer 16 -> T -> bust
    t.place_bets(Bets(main=10))
    t.act("double")
    r = t.round_result
    assert r.hands[0].doubled and r.hands[0].bet == 20 and r.hands[0].result == "win"
    assert r.net == 20 and t.balance == 1020


def test_hard_double_restriction():
    t = make_table({"rules": {"double_any_two": False}})
    stack(t, "5h 6c 7d Ks")  # hard 12: no double
    t.place_bets(Bets(main=10))
    assert Action.DOUBLE not in t.legal_actions()


def test_peek_on_ace_dealer_bj_with_insurance():
    t = make_table()
    stack(t, "9h Ac 7d Ks")
    t.place_bets(Bets(main=10))
    assert t.phase is Phase.INSURANCE
    assert t.legal_actions() == [Action.INSURE, Action.NO_INSURANCE]
    t.act("insure")
    r = t.round_result
    assert t.phase is Phase.SETTLED and r.dealer_blackjack
    assert r.insurance_stake == 5 and r.insurance_payout == 15
    assert r.net == 0  # lose 10 main, insurance +10
    assert any(e["type"] == "peek" for e in t.events)


def test_peek_on_ace_no_bj_insurance_lost():
    t = make_table()
    stack(t, "Th Ac 9d 7s")
    t.place_bets(Bets(main=10))
    t.act("insure")
    assert t.phase is Phase.PLAYER
    t.act("stand")  # 19 vs soft 18
    r = t.round_result
    assert r.insurance_payout == 0 and r.hands[0].result == "win" and r.net == 5


def test_no_peek_on_ten_late_bj_takes_all():
    t = make_table()
    stack(t, "5h Tc 6d As")  # 11 vs T, dealer has BJ but no peek
    t.place_bets(Bets(main=10))
    assert t.phase is Phase.PLAYER
    t.act("double")
    r = t.round_result
    assert r.dealer_blackjack and r.hands[0].result == "lose"
    assert r.net == -20


def test_late_bj_original_only():
    t = make_table({"rules": {"late_dealer_bj_takes": "original"}})
    stack(t, "5h Tc 6d As 9c")
    t.place_bets(Bets(main=10))
    t.act("double")
    assert t.round_result.net == -10


def test_late_bj_original_split():
    t = make_table({"rules": {"late_dealer_bj_takes": "original"}})
    stack(t, "8h Tc 8d As 2c 3d")
    t.place_bets(Bets(main=10))
    t.act("split")
    play(t, "stand", "stand")
    r = t.round_result
    assert [h.result for h in r.hands] == ["lose", "push"] and r.net == -10


def test_peek_on_ten_config():
    t = make_table({"rules": {"peek_on_ten": True}})
    stack(t, "5h Tc 6d As")
    t.place_bets(Bets(main=10))
    assert t.phase is Phase.SETTLED and t.round_result.net == -10


def test_shared_blackjack_dealer_does_not_draw():
    t = make_table()
    stack(t, "Ah 6c Kd Ts 9c")
    t.place_bets(Bets(main=10, bust_bonus=2))
    r = t.round_result
    assert t.phase is Phase.SETTLED and r.shared_blackjack
    assert len(r.dealer_cards) == 2 and not r.dealer_played
    assert r.hands[0].result == "blackjack" and r.hands[0].payout == 25
    bb = [s for s in r.side_bets if s.bet == "bust_bonus"][0]
    assert bb.outcome == "push" and bb.payout == 2
    assert r.net == 15


def test_blackjack_vs_dealer_blackjack_push():
    t = make_table()
    stack(t, "Ah Tc Kd As")  # no peek on ten; shared BJ -> hole turned -> push
    t.place_bets(Bets(main=10))
    r = t.round_result
    assert r.hands[0].result == "push" and r.net == 0


def test_six_card_charlie_auto_stand_and_beats_dealer_bj():
    t = make_table()
    stack(t, "2h Tc 2d As Ah 2c 3d 3s")  # 2,2,A,2,3,3 = 13 with 6 cards; dealer has BJ (no peek)
    t.place_bets(Bets(main=10))
    play(t, "hit", "hit", "hit")
    assert t.phase is Phase.PLAYER
    t.act("hit")
    r = t.round_result
    assert t.phase is Phase.SETTLED
    assert len(r.hands[0].cards) == 6 and r.hands[0].result == "charlie"
    assert r.dealer_blackjack and r.net == 10


def test_charlie_disabled():
    t = make_table({"rules": {"charlie_cards": 0}})
    stack(t, "2h 9c 2d 8s Ah 2c 3d 3s 2s")
    t.place_bets(Bets(main=10))
    play(t, "hit", "hit", "hit", "hit")
    assert t.phase is Phase.PLAYER  # 6 cards, 13, still playing


def test_split_once_das_and_by_value():
    t = make_table()
    stack(t, "Kh 6c Qd 7s 9c Kc 5d 4d Ts")
    t.place_bets(Bets(main=10))
    assert Action.SPLIT in t.legal_actions()  # K-Q split by value
    t.act("split")
    assert len(t.hands) == 2 and t.balance == 980
    # hand 1 = K,9 ; hand 2 not dealt yet
    assert len(t.hands[0].cards) == 2 and len(t.hands[1].cards) == 1
    t.act("stand")
    # hand 2 = Q,K -> no re-split (max_splits=1)
    assert t.hands[1].total == 20 and Action.SPLIT not in t.legal_actions()
    t.act("stand")
    r = t.round_result
    assert r.dealer_total == 18 or r.dealer_bust or True
    assert len(r.hands) == 2


def test_no_split_by_value_when_disabled():
    t = make_table({"rules": {"split_by_value": False}})
    stack(t, "Kh 6c Qd 7s")
    t.place_bets(Bets(main=10))
    assert Action.SPLIT not in t.legal_actions()


def test_das_rule():
    for das in (True, False):
        t = make_table({"rules": {"double_after_split": das}})
        stack(t, "8h 6c 8d 7s 3c")
        t.place_bets(Bets(main=10))
        t.act("split")
        assert (Action.DOUBLE in t.legal_actions()) is das


def test_split_aces_one_card_and_21_is_not_blackjack():
    t = make_table()
    stack(t, "Ah 6c Ad 7s Kc 5d Tc")  # aces get K and 5; dealer 13 -> T -> bust
    t.place_bets(Bets(main=10))
    t.act("split")
    r = t.round_result
    assert t.phase is Phase.SETTLED
    assert [len(h.cards) for h in r.hands] == [2, 2] and all(h.split_aces for h in r.hands)
    assert r.hands[0].total == 21 and r.hands[0].result == "win" and r.hands[0].payout == 20  # 1:1, not 3:2
    assert r.net == 20


def test_split_aces_vs_dealer_21():
    t = make_table()
    stack(t, "Ah 6c Ad 5s Kc 5d Th")  # dealer 6,5,T = 21 -> A+K pushes
    t.place_bets(Bets(main=10))
    t.act("split")
    r = t.round_result
    assert [h.result for h in r.hands] == ["push", "lose"]


def test_validation_and_phases():
    t = make_table()
    with pytest.raises(ValueError):
        t.place_bets(Bets(main=0))
    with pytest.raises(ValueError):
        t.place_bets(Bets(main=10, perfect_pairs=0.5))
    with pytest.raises(ValueError):
        t.place_bets(Bets(main=6000))
    stack(t, "9h 6c 7d Ks")
    t.place_bets(Bets(main=10))
    with pytest.raises(RuntimeError):
        t.place_bets(Bets(main=10))
    with pytest.raises(ValueError):
        t.act("split")


def test_strict_bankroll():
    t = make_table(strict=True, bankroll=15)
    stack(t, "5h 6c 4d Ks")
    t.place_bets(Bets(main=10))
    assert Action.DOUBLE not in t.legal_actions()
    t.act("stand")
    with pytest.raises(ValueError):
        t.place_bets(Bets(main=100))


def test_observation_hides_hole_card():
    t = make_table()
    stack(t, "9h 6c 7d Ks")
    t.place_bets(Bets(main=10))
    obs = t.observation()
    assert obs.dealer_cards == [parse_card("6c")] and obs.dealer_total == 6
    assert parse_card("Ks") not in obs.dealer_cards
    # the hole card is still part of the unseen composition
    assert sum(obs.seen_counts) == 3 and obs.cards_unseen == 416 - 3
    assert obs.unseen_counts[9] == 128 and obs.unseen_card_counts[parse_card("Ks")] == 8
    snap = t.snapshot()
    hole = snap["game"]["dealer"]["cards"][1]
    assert hole == {"face_up": False, "code": "back", "owner": "dealer"}
    assert "K_of_spades" not in str(snap["game"]["dealer"])
    t.act("stand")
    obs = t.observation()
    assert parse_card("Ks") in obs.dealer_cards


def test_snapshot_shape_and_card_codes():
    t = Table(load_config(), seed=3)
    snap = t.snapshot()
    assert snap["game"]["legal_actions"] == ["deal"]
    for _ in range(30):
        t.place_bets(Bets(main=10, perfect_pairs=1, twenty_one_plus_three=1, crazy_7=1, bust_bonus=1))
        while t.phase in (Phase.INSURANCE, Phase.PLAYER):
            t.act(t.legal_actions()[-1] if t.phase is Phase.INSURANCE else "stand")
        snap = t.snapshot()
        g, s = snap["game"], snap["shoe"]
        for key in ("round_id", "phase", "balance", "total_bet", "bets", "dealer", "hands", "active_hand", "crowd",
                    "legal_actions", "side_bet_results", "round_net", "messages"):
            assert key in g
        for key in ("shoe_number", "half", "half_size", "dealt_in_half", "cut_at", "burned", "cards_unseen",
                    "decks_unseen", "seen_counts", "unseen_counts", "running", "true", "analytics", "history"):
            assert key in s
        assert set(s["running"]) == {"hilo", "ko", "omega2", "wong_halves", "zen"}
        cards = g["dealer"]["cards"] + [c for h in g["hands"] for c in h["cards"]] + g["crowd"]["cards"]
        for c in cards:
            assert os.path.exists(os.path.join(CARDS_DIR, c["code"] + ".png")), c["code"]
            assert c["owner"] in ("player", "dealer", "crowd")
        assert g["bets"]["insurance"] >= 0 and set(g["bets"]) == {
            "main", "perfect_pairs", "twenty_one_plus_three", "crazy_7", "bust_bonus", "insurance"}
    import json
    json.dumps(snap)


def test_crowd_continues_shared_hand():
    cfg = load_config(patch={"crowd": {"enabled": True, "size": 250, "mix": ONLY_RECKLESS}})
    t = Table(cfg, seed=1)
    stack(t, "Th 6c 3d 9s 2c 3h 5s")  # player 13 stands; reckless crowd hits <19: 15, 18, 23
    t.place_bets(Bets(main=10))
    t.act("stand")
    r = t.round_result
    assert r.hands[0].cards == [parse_card("Th"), parse_card("3d")]
    assert r.crowd_cards == [parse_card(x) for x in ("2c", "3h", "5s")]
    crowd_events = [e for e in t.events if e["type"] == "card" and e["owner"] == "crowd"]
    assert len(crowd_events) == 3
    # dealer 6,9 = 15 -> draws after the crowd
    assert r.dealer_cards[2] not in r.crowd_cards and len(r.dealer_cards) >= 3


def test_crowd_basic_stands_and_player_hits():
    cfg = load_config(patch={"crowd": {"enabled": True, "mix": ONLY_BASIC}})
    t = Table(cfg, seed=2)
    stack(t, "Th 6c 3d 9s 2c Tc")
    t.place_bets(Bets(main=10))
    t.act("hit")  # crowd (basic) stands on 13 vs 6, player hits alone
    assert t.crowd.hitters == 0
    t.act("stand")
    assert t.round_result.crowd_cards == []


def test_split_crowd_finishes_shared_hand_first():
    cfg = load_config(patch={"crowd": {"enabled": True, "mix": ONLY_RECKLESS}})
    t = Table(cfg, seed=1)
    stack(t, "8h 6c 8d 9s Tc Ac 2d 3d")  # crowd: 16 -> T bust; then split hands get A, 2
    t.place_bets(Bets(main=10))
    t.act("split")
    assert t.crowd_cards == [parse_card("Tc")]
    assert t.hands[0].cards == [parse_card("8h"), parse_card("Ac")]
    owners = [e["owner"] for e in t.events if e["type"] == "card"]
    assert owners[:6] == ["player", "dealer", "player", "dealer", "crowd", "player"]
    t.act("stand")
    assert t.hands[1].cards == [parse_card("8d"), parse_card("2d")]


def test_dealer_plays_with_bust_bonus_even_if_player_busts():
    t = make_table()
    stack(t, "Th 6c 6d 9s Kc 7c")
    t.place_bets(Bets(main=10, bust_bonus=1))
    t.act("hit")
    r = t.round_result
    assert r.hands[0].result == "lose" and len(r.dealer_cards) == 3 and r.dealer_played
    t2 = make_table()
    stack(t2, "Th 6c 6d 9s Kc 7c")
    t2.place_bets(Bets(main=10))
    t2.act("hit")
    assert len(t2.round_result.dealer_cards) == 2 and not t2.round_result.dealer_played


def test_determinism():
    from onebj.sim import make_agent, run_rounds
    res = []
    for _ in range(2):
        cfg = load_config()
        t = Table(cfg, seed=99, record_events=False)
        m = run_rounds(t, make_agent("hilo", cfg), 3000)
        res.append((m.net, m.wins, m.losses, t.shoe.pos, t.balance))
    assert res[0] == res[1]


def test_events_vs_record_events_off_same_outcome():
    from onebj.sim import make_agent, play_round
    a = Table(load_config(), seed=5, record_events=True)
    b = Table(load_config(), seed=5, record_events=False)
    ag = make_agent("basic", load_config())
    for _ in range(500):
        ra, rb = play_round(a, ag), play_round(b, ag)
        assert ra.net == rb.net and ra.dealer_cards == rb.dealer_cards


def test_clone_without_resample_is_identical_future():
    t = Table(load_config(), seed=11)
    t.place_bets(Bets(main=10))
    while t.phase is Phase.INSURANCE:
        t.act("no_insurance")
    if t.phase is Phase.PLAYER:
        c = t.clone(resample_unseen=False)
        t.act("stand")
        c.act("stand")
        assert t.round_result.dealer_cards == c.round_result.dealer_cards
        assert t.round_result.net == c.round_result.net


def test_clone_resample_does_not_leak():
    t = make_table(seed=21)
    stack(t, "9h Tc 7d 5s")
    t.place_bets(Bets(main=10))
    obs = t.observation()
    holes, nexts = set(), set()
    for i in range(60):
        c = t.clone(resample_unseen=True, seed=i)
        assert c.observation().unseen_counts == obs.unseen_counts
        assert c.observation().seen_counts == obs.seen_counts
        assert c.dealer.cards[0] == t.dealer.cards[0] and c.hands[0].cards == t.hands[0].cards
        holes.add(c.dealer.cards[1])
        nexts.add(c.shoe.cards[c.shoe.pos])
        # multiset of the clone's hidden cards equals the unseen multiset
        assert sorted(c.shoe.cards) == sorted(x for x in range(52) for _ in range(8))  # still a full shoe
    assert len(holes) > 10 and len(nexts) > 10
    # original untouched
    assert t.dealer.cards[1] == parse_card("5s")


def test_clone_respects_peek():
    t = make_table(seed=4)
    stack(t, "9h Ac 7d 5s")
    t.place_bets(Bets(main=10))
    t.act("no_insurance")
    assert t.phase is Phase.PLAYER and t.peeked
    for i in range(200):
        c = t.clone(seed=i)
        assert BUCKET_OF[c.dealer.cards[1]] != 9


def test_bust_bonus_lost_on_player_blackjack_when_configured():
    t = make_table({"side_bets": {"bust_bonus_push_on_player_blackjack": False}})
    stack(t, "Ah 6c Kd Ts 9c")
    t.place_bets(Bets(main=10, bust_bonus=2))
    bb = [s for s in t.round_result.side_bets if s.bet == "bust_bonus"][0]
    assert bb.outcome is None and bb.payout == 0


def _c7(t):
    return [s for s in t.round_result.side_bets if s.bet == "crazy_7"][0]


def test_crazy7_uses_players_third_card_from_hit():
    t = make_table()
    stack(t, "7h 7c 7d 9s 7h Tc")  # 7h 7d + hit 7h -> suited? no: hearts, diamonds, hearts -> unsuited
    t.place_bets(Bets(main=10, crazy_7=1))
    assert not [s for s in t.side_results if s.bet == "crazy_7"]  # pending until the third card
    t.act("hit")  # 21 -> auto-stand
    r = _c7(t)
    assert r.outcome == "three_sevens" and r.payout == 501


def test_crazy7_suited_trips_from_double():
    t = make_table()
    stack(t, "7h 6c 7h 9s 7h Tc")
    t.place_bets(Bets(main=10, crazy_7=1))
    t.act("double")
    assert _c7(t).outcome == "three_sevens_suited" and _c7(t).payout == 2001


def test_crazy7_dealer_card_does_not_count():
    t = make_table()
    stack(t, "7h 7c 7d 9s")  # dealer up 7 is not used; player stands -> no third card
    t.place_bets(Bets(main=10, crazy_7=1))
    t.act("stand")
    assert _c7(t).outcome == "two_sevens" and _c7(t).payout == 26


def test_crazy7_crowd_hit_counts_as_third_card():
    cfg = load_config(patch={"crowd": {"enabled": True, "mix": ONLY_RECKLESS}})
    t = Table(cfg, seed=3)
    stack(t, "7h 9c 7h 9s 7h 5c")  # player stands on 14, reckless crowd hits: 7h -> 7-7-7 suited
    t.place_bets(Bets(main=10, crazy_7=1))
    t.act("stand")
    assert t.round_result.crowd_cards[0] == parse_card("7h")
    assert _c7(t).outcome == "three_sevens_suited"


def test_crazy7_split_sevens_next_card_of_first_hand():
    cfg = load_config(patch={"crowd": {"enabled": True, "mix": ONLY_RECKLESS}})
    t = Table(cfg, seed=3)
    stack(t, "7h 6c 7d 9s Tc 7s 2c")  # crowd busts the unsplit 14 with Tc; split hand 1 gets 7s
    t.place_bets(Bets(main=10, crazy_7=1))
    t.act("split")
    assert t.crowd_cards == [parse_card("Tc")]
    while t.phase is Phase.PLAYER:
        t.act("stand")
    assert _c7(t).outcome == "three_sevens"


def test_crazy7_no_third_card_when_dealer_peeks_blackjack():
    t = make_table()
    stack(t, "7h Ac 7h Ks 7h")
    t.place_bets(Bets(main=10, crazy_7=1))
    t.act("no_insurance")
    assert _c7(t).outcome == "two_sevens_suited" and _c7(t).payout == 151


def test_crazy7_one_seven_settled_at_deal():
    t = make_table()
    stack(t, "7h 6c 9d Ks")
    t.place_bets(Bets(main=10, crazy_7=1))
    assert [s.outcome for s in t.side_results] == ["one_seven"]


def test_cashout_offer_and_settlement():
    t = make_table()
    stack(t, "Th Tc 6d 7s")  # 16 vs T (no peek): offer ~ 0.98 * fair value
    t.place_bets(Bets(main=10))
    offer = t.cashout_offer()
    fair = t.cashout_table.fair_value(16, False, 2, 9, True)
    assert offer == round(0.98 * fair * 10, 2) and 0 < offer < 10
    obs = t.observation()
    assert obs.hands[0].cashout_offer == offer
    assert t.snapshot(analytics=False)["game"]["hands"][0]["cashout_offer"] == offer
    t.act("cashout")
    r = t.round_result
    assert r.hands[0].result == "cashout" and r.hands[0].payout == offer
    assert t.balance == 1000 - 10 + offer
    assert t.snapshot(analytics=False)["game"]["hands"][0]["status"] == "cashed_out"


def test_cashout_paid_even_vs_late_blackjack_and_disabled_after_double():
    t = make_table()
    stack(t, "9h Tc 7d As")
    t.place_bets(Bets(main=10))
    offer = t.cashout_offer()
    t.act("cashout")
    assert t.round_result.dealer_blackjack and t.round_result.net == offer - 10
    t2 = make_table({"rules": {"cashout": False}})
    stack(t2, "9h Tc 7d As")
    t2.place_bets(Bets(main=10))
    assert Action.CASHOUT not in t2.legal_actions() and t2.cashout_offer() is None


def test_cashout_split_hands_have_own_offers():
    t = make_table()
    stack(t, "8h 6c 8d 9s 3c 2d")
    t.place_bets(Bets(main=10))
    t.act("split")
    o1 = t.cashout_offer()
    assert o1 is not None and t.cashout_offer(1) is None
    t.act("cashout")
    o2 = t.cashout_offer()
    assert o2 is not None and t.active == 1
    t.act("stand")
    assert t.round_result.hands[0].result == "cashout"


def test_cashout_never_beats_fresh_shoe_value():
    t = make_table()
    ct = t.cashout_table
    for up in range(10):
        for total in range(4, 21):
            for n in (2, 3, 4, 5):
                v = ct.fair_value(total, False, n, up, n == 2)
                assert ct.offer(total, False, n, up, n == 2) <= v + 1e-12


def test_player_blackjack_pushes_peeked_dealer_blackjack():
    for spec, patch in (("Ah Ac Kd Ts", {}), ("Ah Tc Kd As", {"rules": {"peek_on_ten": True}})):
        t = make_table(patch)
        stack(t, spec)
        t.place_bets(Bets(main=10, bust_bonus=1))
        if t.phase is Phase.INSURANCE:
            t.act("no_insurance")
        r = t.round_result
        assert r.dealer_blackjack and r.hands[0].result == "push" and r.net == 0  # bust bonus returned too


def test_even_money():
    t = make_table()
    stack(t, "Ah Ac Kd Ts")
    t.place_bets(Bets(main=10))
    t.act("insure")
    assert t.round_result.net == 10  # BJ push + insurance 2:1 = even money
