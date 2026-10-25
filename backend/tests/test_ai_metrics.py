"""sim.Metrics classifies cashed-out hands by payout vs stake (engine fix made for the NN evaluation)."""

from conftest import make_table, stack
from onebj.engine import Action, Bets, Phase
from onebj.sim import Metrics


def _cashout_round(spec: str):
    t = make_table()
    stack(t, spec)
    t.place_bets(Bets(main=10))
    assert t.phase is Phase.PLAYER
    offer = t.cashout_offer()
    t.act(Action.CASHOUT)
    return t.round_result, offer


def test_cashout_losing_offer_counts_as_loss():
    rr, offer = _cashout_round("Ts 9h 6c 7d")  # 16 v T: offer < stake
    assert offer < 10
    m = Metrics()
    m.add(rr)
    assert (m.wins, m.losses, m.pushes, m.cashouts) == (0, 1, 0, 1)
    assert abs(m.cashout_net - (offer - 10)) < 1e-9
    d = m.to_dict()
    assert d["cashouts"] == 1 and abs(d["cashout_net"] - (offer - 10)) < 1e-9


def test_cashout_winning_offer_counts_as_win():
    rr, offer = _cashout_round("Ts 6h Tc 7d")  # 20 v 6: offer > stake
    assert offer > 10
    m = Metrics()
    m.add(rr)
    assert (m.wins, m.losses, m.cashouts) == (1, 0, 1)
    other = Metrics()
    other.merge(m)
    assert other.cashouts == 1 and other.wins == 1
