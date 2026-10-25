from onebj.strategy.basic import ILLUSTRIOUS_18, BasicStrategy, get_basic_strategy, hilo_action, should_insure
from onebj.strategy.solver import Solver, SolverRules
from onebj.config import load_config

COLS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]  # 2..9, T, A
# Standard 8-deck S17 DAS (peek, no Charlie) basic strategy — the textbook chart.
STANDARD_HARD = {
    8: "HHHHHHHHHH", 9: "HDDDDHHHHH", 10: "DDDDDDDDHH", 11: "DDDDDDDDDH", 12: "HHSSSHHHHH",
    13: "SSSSSHHHHH", 14: "SSSSSHHHHH", 15: "SSSSSHHHHH", 16: "SSSSSHHHHH", 17: "SSSSSSSSSS",
}
STANDARD_SOFT = {  # A2/A3 vs 5 is a coin flip in 8 decks (EV difference < 0.001) — excluded below
    14: "HHHDDHHHHH", 15: "HHDDDHHHHH", 16: "HHDDDHHHHH", 17: "HDDDDHHHHH", 18: "SDDDDSSHHH", 19: "SSSSSSSSSS",
}
STANDARD_PAIRS = {
    0: "PPPPPPPPPP", 1: "PPPPPP----", 2: "PPPPPP----", 3: "---PP-----", 4: "----------", 5: "PPPPP-----",
    6: "PPPPPP----", 7: "PPPPPPPPPP", 8: "PPPPP-PP--", 9: "----------",
}
SYM = {0: "S", 1: "H", 2: "D", 3: "D"}


def test_standard_chart_reproduced():
    bs = BasicStrategy(SolverRules(peek_ten=True, charlie=0), num_decks=8)
    for t, row in STANDARD_HARD.items():
        assert "".join(SYM[bs.hard[2][t][u]] for u in COLS) == row, t
    for t, row in STANDARD_SOFT.items():
        assert "".join(SYM[bs.soft[2][t][u]] for u in COLS) == row, t
    for x, row in STANDARD_PAIRS.items():
        assert "".join("P" if bs.split[x][u] else "-" for u in COLS) == row, x


def test_one_blackjack_specific_cells():
    bs = get_basic_strategy(load_config().rules, 8)
    T, A = 9, 0
    assert bs.action(11, False, 2, T) == "hit"  # no double vs 10 (late BJ takes all)
    assert bs.action(11, False, 2, 5) == "double"
    assert bs.action(16, False, 2, T, pair=7, can_split=True) == "hit"  # 8,8 vs T: no split
    assert bs.action(16, False, 2, A, pair=7, can_split=True) == "split"  # ace peeks
    assert bs.action(12, True, 2, T, pair=0, can_split=True) == "split"
    # Six Card Charlie: five-card hands hit more
    assert bs.action(16, False, 5, T) == "hit" and bs.action(15, False, 5, 6) == "hit"
    assert bs.action(17, False, 5, T) == "hit" and bs.action(17, False, 2, T) == "stand"
    assert bs.action(18, True, 4, 5) == "hit" and bs.action(20, True, 5, 9) == "hit"
    assert bs.action(21, True, 5, 9) == "stand"  # 21 is auto-stood


def test_solver_rtp_ranges():
    # sanity: standard S17 DAS split-once infinite deck ~ -0.57%; ONE rules ~ -0.49%
    assert -0.0062 < Solver(SolverRules(peek_ten=True, charlie=0)).round_ev() < -0.0052
    assert -0.0055 < Solver(SolverRules()).round_ev() < -0.0043


def test_hilo_deviations():
    bs = get_basic_strategy(load_config().rules, 8)
    assert hilo_action(bs, 16, False, 2, 9, True, -1, False, 1.0)[0] == "stand"
    assert hilo_action(bs, 16, False, 2, 9, True, -1, False, -1.0)[0] == "hit"
    assert hilo_action(bs, 16, False, 3, 9, False, -1, False, 5.0)[0] == "hit"  # 2-card indices only
    assert hilo_action(bs, 12, False, 2, 1, True, -1, False, 4.0)[0] == "stand"
    assert hilo_action(bs, 20, False, 2, 4, True, 9, True, 6.0)[0] == "split"
    assert hilo_action(bs, 20, False, 2, 4, True, 9, True, 2.0)[0] == "stand"
    assert should_insure(3.0) and not should_insure(2.9)
    assert len([d for d in ILLUSTRIOUS_18 if d.kind != "insurance"]) >= 17
