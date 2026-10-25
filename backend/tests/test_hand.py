from onebj.cards import CODES, card_code, card_json, cards_from, make_card, parse_card
from onebj.hand import Hand, hand_value, is_blackjack, is_charlie


def test_card_codes_and_parsing():
    assert card_code(make_card("10", "hearts")) == "10_of_hearts"
    assert card_code(make_card("K", "spades")) == "K_of_spades"
    assert parse_card("Ts") == parse_card("10_of_spades") == make_card("10", "spades")
    assert len(set(CODES)) == 52
    assert card_json(parse_card("Ah"), "dealer", face_up=False) == {"face_up": False, "code": "back", "owner": "dealer"}
    assert card_json(parse_card("Qd"), "crowd")["rank"] == "Q"


def test_hand_values():
    assert hand_value(cards_from("Ah 6c")) == (17, True)
    assert hand_value(cards_from("Ah 6c Kd")) == (17, False)
    assert hand_value(cards_from("Ah Ad")) == (12, True)
    assert hand_value(cards_from("Ah Ad Kd Qs")) == (22, False)
    assert hand_value(cards_from("Ah Kd")) == (21, True)
    assert hand_value(cards_from("9h 8d 5c")) == (22, False)


def test_blackjack_and_charlie():
    assert is_blackjack(cards_from("Ah Jd"))
    assert not is_blackjack(cards_from("Ah Jd"), from_split=True)
    assert not is_blackjack(cards_from("Ah 5d 5c"))
    assert is_charlie(cards_from("2h 2d 3c 3s Ah 4c"))  # 15 with six cards
    assert not is_charlie(cards_from("2h 2d 3c 3s Ah Tc Kd"))  # bust
    assert not is_charlie(cards_from("2h 2d 3c 3s Ah"))
    assert not is_charlie(cards_from("2h 2d 3c 3s Ah 4c"), charlie_cards=0)


def test_hand_class():
    h = Hand(10)
    for c in cards_from("Ah 5d"):
        h.add(c)
    assert (h.total, h.soft, h.blackjack) == (16, True, False)
    h.add(parse_card("Tc"))
    assert (h.total, h.soft, h.bust) == (16, False, False)
    bj = Hand(1)
    for c in cards_from("Kh As"):
        bj.add(c)
    assert bj.blackjack and bj.pair_bucket() == -1
    p = Hand(1)
    for c in cards_from("Kh Qs"):
        p.add(c)
    assert p.pair_bucket() == 9
