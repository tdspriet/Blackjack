"""Composition-dependent analytics for the next round, from what the player has seen.

Inputs are unseen counts — per card (52 entries: rank and suit are both visible when a card is seen)
or per rank bucket. Everything here is fast (~2 ms for :func:`table_analytics` on a new composition).

* Perfect Pairs, 21+3: **exact** EVs (drawing the relevant cards without replacement from the unseen
  pool; all unseen cards — remaining shoe, burn cards, half-1 residue — are exchangeable). Crazy 7:
  exact under the assumption that a third card is drawn unless the dealer peeks a blackjack.
* Dealer final-outcome distribution per up card: S17/H17 dealer recursion by number of cards on the
  current composition (up card removed, later draws with replacement).
* Bust Bonus: exact dealer bust-by-card-count distribution without replacement from the pool, times
  P(no shared blackjack) (+ the returned stake on a player blackjack); the only approximation is the
  independence of "player has blackjack" and the dealer's hand (fresh shoe: 94.109% vs 94.117% exact).
* Insurance EV: exact given the composition (P(hole is a 10-value card)).
* Main-game edge estimate: ``base + sum(EoR_r * excess_removed_per_deck_r)``. EoRs and the base are
  computed for the configured rules with the infinite-deck solver (:mod:`onebj.strategy.solver`) by
  removing one card of each rank from a 52-card deck (by the envelope theorem the first-order effect
  equals the fixed-basic-strategy EoR); ``FINITE_DECK_CORRECTION`` adds the measured difference
  between an 8-deck shoe and the infinite deck (simulation, see docs/RULES.md).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .cards import BUCKETS, NUM_CARDS
from .config import GameConfig, RulesConfig
from .strategy.solver import SolverRules, dealer_outcomes, solve

if TYPE_CHECKING:
    from .engine import Table

SEVEN = 6
_STRAIGHTS = [(r, r + 1, r + 2) for r in range(11)] + [(11, 12, 0)]
#: 8-deck minus infinite-deck main-game EV with this engine's basic strategy (from simulation)
FINITE_DECK_CORRECTION: dict[int, float] = {8: 0.0008}


def _per_rank_suit(unseen: Sequence[int]) -> list[list[int]]:
    """``n[rank][suit]`` from per-card counts (card = rank * 4 + suit)."""
    return [[unseen[r * 4 + s] for s in range(4)] for r in range(13)]


def perfect_pairs_probs(unseen: Sequence[int]) -> dict[str, float]:
    """P(perfect / coloured / mixed pair) for the player's first two cards."""
    N = sum(unseen)
    n2 = N * (N - 1)
    perfect = coloured = mixed = 0.0
    for r in range(13):
        h, d, c, s = (unseen[r * 4 + k] for k in range(4))
        perfect += h * (h - 1) + d * (d - 1) + c * (c - 1) + s * (s - 1)
        coloured += 2 * (h * d + c * s)
        mixed += 2 * (h + d) * (c + s)
    return {"perfect": perfect / n2, "coloured": coloured / n2, "mixed": mixed / n2}


def perfect_pairs_ev(unseen: Sequence[int], paytable: Mapping[str, float]) -> float:
    p = perfect_pairs_probs(unseen)
    return sum(p[k] * (paytable[k] + 1) for k in p) - 1


def twenty_one_plus_three_probs(unseen: Sequence[int]) -> dict[str, float]:
    """P of each (highest) 21+3 hand for player's 2 cards + dealer up card."""
    N = sum(unseen)
    n3 = N * (N - 1) * (N - 2)
    rs = _per_rank_suit(unseen)
    m = [sum(x) for x in rs]
    suit_tot = [sum(rs[r][s] for r in range(13)) for s in range(4)]
    suited_trips = sum(x * (x - 1) * (x - 2) for row in rs for x in row)
    trips_any = sum(x * (x - 1) * (x - 2) for x in m)
    sf = 6 * sum(rs[a][s] * rs[b][s] * rs[c][s] for a, b, c in _STRAIGHTS for s in range(4))
    straight_any = 6 * sum(m[a] * m[b] * m[c] for a, b, c in _STRAIGHTS)
    flush_any = sum(x * (x - 1) * (x - 2) for x in suit_tot)
    return {"suited_trips": suited_trips / n3, "straight_flush": sf / n3,
            "three_of_a_kind": (trips_any - suited_trips) / n3, "straight": (straight_any - sf) / n3,
            "flush": (flush_any - sf - suited_trips) / n3}


def twenty_one_plus_three_ev(unseen: Sequence[int], paytable: Mapping[str, float]) -> float:
    p = twenty_one_plus_three_probs(unseen)
    return sum(p[k] * (paytable[k] + 1) for k in p) - 1


def crazy_7_probs(unseen: Sequence[int], peek_on_ace: bool = True) -> dict[str, float]:
    """P of each Crazy 7 tier on the player's first three cards.

    The third card is assumed to be drawn whenever the dealer has not peeked a blackjack (true for
    basic strategy — 7-7 is split or hit — and with the crowd); a peeked blackjack needs an ace up and a
    ten in the hole, drawn from the same pool. Exact for that model.
    """
    N = sum(unseen)
    k = [unseen[SEVEN * 4 + s] for s in range(4)]
    K = sum(k)
    n2 = N * (N - 1)
    M = N - 2
    f = 1.0 / M  # P(a given 7 is the third card)
    if peek_on_ace:
        A, T = sum(unseen[0:4]), sum(unseen[36:52])
        f *= 1.0 - A * T / ((M - 1) * (M - 2))
    ss = sum(x * (x - 1) for x in k)
    two_suited, two_any = ss / n2, K * (K - 1) / n2
    three_suited = sum(x * (x - 1) * (x - 2) for x in k) / n2 * f
    three_any = K * (K - 1) * (K - 2) / n2 * f
    three_first_suited = ss * (K - 2) / n2 * f
    return {"three_suited": three_suited, "three_unsuited": three_any - three_suited,
            "pair_suited": two_suited - three_first_suited,
            "pair_unsuited": (two_any - two_suited) - (three_any - three_first_suited),
            "one": K * (N - K) / n2}


def crazy_7_ev(unseen: Sequence[int], paytable: Mapping[str, float], peek_on_ace: bool = True) -> float:
    p = crazy_7_probs(unseen, peek_on_ace)
    return sum(p[k] * (paytable[k] + 1) for k in p) - 1


def _probs_without(buckets: Sequence[int], up: int) -> tuple[float, ...] | None:
    b = list(buckets)
    if b[up] <= 0:
        return None
    b[up] -= 1
    tot = sum(b)
    return tuple(x / tot for x in b)


@lru_cache(maxsize=512)
def dealer_distribution(unseen_buckets: tuple[int, ...], h17: bool = False) -> dict[int, dict[str, Any]]:
    """Per up-card bucket: ``{"p_up", "totals": {17..21, "bust"}, "bust_by_cards": {n: p},
    "blackjack": p}`` (unconditional on the hole card; the up card is removed from the pool)."""
    N = sum(unseen_buckets)
    out: dict[int, dict[str, Any]] = {}
    for u in range(10):
        p = _probs_without(unseen_buckets, u)
        if p is None:
            out[u] = {"p_up": 0.0, "totals": {}, "bust_by_cards": {}, "blackjack": 0.0, "bust": 0.0}
            continue
        o = dealer_outcomes(u, p, h17)
        totals: dict[Any, float] = {t: 0.0 for t in (17, 18, 19, 20, 21)}
        totals["bust"] = 0.0
        bust_by: dict[int, float] = {}
        for (t, n), pr in o.items():
            if t == 22:
                totals["bust"] += pr
                bust_by[n] = bust_by.get(n, 0.0) + pr
            else:
                totals[t] += pr
        bj = p[9] if u == 0 else (p[0] if u == 9 else 0.0)
        out[u] = {"p_up": unseen_buckets[u] / N, "totals": totals, "bust_by_cards": bust_by,
                  "blackjack": bj, "bust": totals["bust"]}
    return out


@lru_cache(maxsize=256)
def dealer_bust_by_cards_exact(unseen_buckets: tuple[int, ...], h17: bool = False) -> dict[int, float]:
    """P(dealer busts with n cards) drawing the whole dealer hand without replacement from the pool
    (exact for the dealer's marginal distribution: unknown cards dealt to players are exchangeable).
    Key 0 holds P(no bust). Memoised on the multiset of dealer cards drawn so far (~1 ms)."""
    counts = unseen_buckets
    N = sum(counts)
    memo: dict[tuple[int, ...], dict[int, float]] = {}

    def rec(removed: tuple[int, ...], hard: int, ace: bool, n: int) -> dict[int, float]:
        r = memo.get(removed)
        if r is not None:
            return r
        left = N - n
        out: dict[int, float] = {}
        for b in range(10):
            c = counts[b] - removed[b]
            if c <= 0:
                continue
            p = c / left
            h2 = hard + b + 1
            a2 = ace or b == 0
            n2 = n + 1
            if h2 > 21:
                out[n2] = out.get(n2, 0.0) + p
                continue
            soft = a2 and h2 <= 11
            t = h2 + 10 if soft else h2
            if n2 >= 2 and (t > 17 or (t == 17 and not (h17 and soft))):
                out[0] = out.get(0, 0.0) + p
                continue
            sub = rec(removed[:b] + (removed[b] + 1,) + removed[b + 1:], h2, a2, n2)
            for k, v in sub.items():
                out[k] = out.get(k, 0.0) + p * v
        memo[removed] = out
        return out

    return rec((0,) * 10, 0, False, 0)


def bust_bonus_probs(unseen_buckets: Sequence[int], h17: bool = False) -> dict[str, Any]:
    """P(dealer busts with n cards) (exact dealer marginal) and P(shared player blackjack)."""
    by = {n: p for n, p in dealer_bust_by_cards_exact(tuple(unseen_buckets), h17).items() if n}
    N = sum(unseen_buckets)
    a, t = unseen_buckets[0], unseen_buckets[9]
    p_pbj = 2 * a * t / (N * (N - 1))
    return {"bust_by_cards": by, "player_blackjack": p_pbj}


def bust_bonus_ev(unseen_buckets: Sequence[int], paytable: Mapping[str, float],
                  push_on_player_blackjack: bool = True, h17: bool = False) -> float:
    """Bust Bonus EV: the dealer plays unless the shared hand is a blackjack (then the stake is returned,
    official rule, or lost with ``push_on_player_blackjack=False``)."""
    pr = bust_bonus_probs(unseen_buckets, h17)
    keys = sorted(int(k) for k in paytable)
    ret = 0.0
    for n, p in pr["bust_by_cards"].items():
        k = min(n, keys[-1])
        if k >= keys[0]:
            ret += p * (paytable[str(k)] + 1)
    pbj = pr["player_blackjack"]
    ret *= 1 - pbj
    if push_on_player_blackjack:
        ret += pbj
    return ret - 1


def insurance_ev(unseen_buckets: Sequence[int], payout: float = 2.0) -> float:
    """EV per unit of insurance stake given the dealer shows an ace (counts must exclude the up ace:
    pass the current unseen counts while the ace is showing; before the deal it is an approximation)."""
    N = sum(unseen_buckets)
    p = unseen_buckets[9] / N if N else 0.0
    return p * payout - (1 - p)


# ---------------------------------------------------------------------- main-game edge model
@lru_cache(maxsize=16)
def edge_model(rules: SolverRules) -> tuple[float, tuple[float, ...]]:
    """(infinite-deck base EV, EoR per bucket) for ``rules``; EoR = EV change per card removed from
    one deck (T entry is per single 10-value card)."""
    base = solve(rules).round_ev()
    eor = []
    for r in range(10):
        counts = [4.0] * 9 + [16.0]
        counts[r] -= 1
        tot = sum(counts)
        eor.append(solve(rules, tuple(c / tot for c in counts)).round_ev() - base)
    return base, tuple(eor)


def main_edge_estimate(unseen_buckets: Sequence[int], rules: RulesConfig | SolverRules,
                       num_decks: int = 8) -> float:
    """Approximate player edge (EV per unit main bet, flat basic strategy) for the given composition."""
    sr = rules if isinstance(rules, SolverRules) else SolverRules.from_config(rules)
    base, eor = edge_model(sr)
    N = sum(unseen_buckets)
    if N <= 0:
        return base
    decks = N / 52
    edge = base + FINITE_DECK_CORRECTION.get(num_decks, 0.0)
    for r in range(10):
        expected = N * (4 if r == 9 else 1) / 13
        edge += eor[r] * (expected - unseen_buckets[r]) / decks
    return edge


def side_bet_evs(unseen_cards: Sequence[int], cfg: GameConfig) -> dict[str, float]:
    """EV per unit stake of all four side bets for the next round (~1.5 ms)."""
    sb = cfg.side_bets
    buckets = cards_to_buckets(unseen_cards)
    bb = bust_bonus_ev(buckets, sb.bust_bonus, sb.bust_bonus_push_on_player_blackjack,
                       cfg.rules.dealer_hits_soft_17)
    return {
        "perfect_pairs": perfect_pairs_ev(unseen_cards, sb.perfect_pairs),
        "twenty_one_plus_three": twenty_one_plus_three_ev(unseen_cards, sb.twenty_one_plus_three),
        "crazy_7": crazy_7_ev(unseen_cards, sb.crazy_7, cfg.rules.peek_on_ace),
        "bust_bonus": bb,
    }


def cards_to_buckets(unseen_cards: Sequence[int]) -> list[int]:
    b = [0] * 10
    for c in range(NUM_CARDS):
        b[min(c // 4, 9)] += unseen_cards[c]
    return b


def table_analytics(table: "Table") -> dict[str, Any]:
    """The ``shoe.analytics`` object of SPEC §6 for the table's current knowledge."""
    cfg = table.cfg
    shoe = table.shoe
    cards = shoe.unseen_cards()
    buckets = shoe.unseen_buckets()
    out: dict[str, Any] = side_bet_evs(cards, cfg)
    out["insurance"] = insurance_ev(buckets, cfg.rules.insurance_payout)
    out["main_edge_estimate"] = main_edge_estimate(buckets, cfg.rules, shoe.num_decks)
    dist = dealer_distribution(tuple(buckets), cfg.rules.dealer_hits_soft_17)
    out["p_dealer_bust_by_upcard"] = {BUCKETS[u]: dist[u]["bust"] for u in range(10)}
    out["dealer_outcomes_by_upcard"] = {
        BUCKETS[u]: {str(k): v for k, v in dist[u]["totals"].items()} for u in range(10)}
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}
