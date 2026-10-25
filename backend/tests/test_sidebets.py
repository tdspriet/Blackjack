import pytest

from onebj.cards import parse_card as P
from onebj.config import SideBetsConfig
from onebj.sidebets import (is_straight, payout, settle_21_3, settle_bust_bonus, settle_crazy_7,
                            settle_perfect_pairs)

SB = SideBetsConfig()


@pytest.mark.parametrize("a,b,outcome,odds", [
    ("Qh", "Qh", "perfect_pair", 25), ("Qh", "Qd", "coloured_pair", 12), ("Qs", "Qc", "coloured_pair", 12),
    ("Qh", "Qs", "mixed_pair", 6), ("Qh", "Kh", None, 0), ("Th", "Jh", None, 0),
])
def test_perfect_pairs(a, b, outcome, odds):
    assert settle_perfect_pairs(P(a), P(b), SB.perfect_pairs) == (outcome, odds)


@pytest.mark.parametrize("cards,outcome", [
    ("Ah 2c 3d", "straight"), ("Qh Kc Ad", "straight"), ("Kh Ac 2d", None), ("Ah 2h 3h", "straight_flush"),
    ("Qs Ks As", "straight_flush"), ("9h Th Jh", "straight_flush"), ("Th Jc Qd", "straight"),
    ("7h 7h 7h", "suited_trips"), ("7h 7d 7h", "three_of_a_kind"), ("2h 9h Kh", "flush"),
    ("7h 7h Kh", "flush"), ("Kh Kd Ks", "three_of_a_kind"), ("Th Kh Qd", None), ("Th Jh Kh", "flush"),
])
def test_21_3(cards, outcome):
    a, b, c = (P(x) for x in cards.split())
    res, odds = settle_21_3(a, b, c, SB.twenty_one_plus_three)
    assert res == outcome
    if outcome:
        assert odds == SB.twenty_one_plus_three[{"suited_trips": "suited_trips", "three_of_a_kind": "three_of_a_kind",
                                                 "straight_flush": "straight_flush", "straight": "straight",
                                                 "flush": "flush"}[outcome]]


def test_straight_ranks():
    assert is_straight(0, 1, 2) and is_straight(11, 12, 0) and not is_straight(12, 0, 1)
    assert not is_straight(9, 10, 10)


@pytest.mark.parametrize("cards,outcome,odds", [
    ("7h 2c -", "one_seven", 5), ("2h 7c 7d", None, 0), ("7h 7d -", "two_sevens", 25),
    ("7h 7h -", "two_sevens_suited", 150), ("7h 7h 5h", "two_sevens_suited", 150),
    ("7h 7d 7c", "three_sevens", 500), ("7h 7h 7d", "three_sevens", 500), ("7s 7s 7s", "three_sevens_suited", 2000),
    ("7h 9h 7h", "one_seven", 5),
])
def test_crazy_7(cards, outcome, odds):
    a, b, c = cards.split()
    res = settle_crazy_7(P(a), P(b), None if c == "-" else P(c), SB.crazy_7)
    assert res == (outcome, odds)


@pytest.mark.parametrize("n,bust,odds", [(3, True, 1), (4, True, 2), (5, True, 9), (6, True, 50), (7, True, 100),
                                         (8, True, 250), (11, True, 250), (5, False, None)])
def test_bust_bonus(n, bust, odds):
    res, o = settle_bust_bonus(n, bust, False, SB.bust_bonus)
    if odds is None:
        assert res is None
    else:
        assert o == odds and res == f"bust_{n}_cards"


def test_bust_bonus_player_blackjack_push():
    assert settle_bust_bonus(2, False, True, SB.bust_bonus, True) == ("push", 0.0)
    assert settle_bust_bonus(2, False, True, SB.bust_bonus, False) == (None, 0.0)
    assert payout(5, "push", 0) == 5 and payout(5, None, 0) == 0 and payout(5, "x", 25) == 130
