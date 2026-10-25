import random
import time

import pytest

from onebj import analytics as A
from onebj.config import load_config
from onebj.engine import Table
from onebj.sidebets import settle_21_3, settle_crazy_7, settle_perfect_pairs
from onebj.strategy.solver import SolverRules

CFG = load_config()


def skewed_pool(seed: int = 0) -> list[int]:
    """A depleted 8-deck composition: random removals biased towards small cards."""
    rng = random.Random(seed)
    counts = [8] * 52
    for _ in range(150):
        c = rng.randrange(52) if rng.random() < 0.5 else rng.randrange(4, 24)
        if counts[c]:
            counts[c] -= 1
    return counts


def sample_ev(counts, n, fn):
    rng = random.Random(1)
    pool = [c for c in range(52) for _ in range(counts[c])]
    tot = 0.0
    for _ in range(n):
        tot += fn(rng.sample(pool, 5))
    return tot / n


def ret(res):
    out, odds = res
    return -1.0 if out is None else (0.0 if out == "push" else odds)


@pytest.mark.parametrize("seed", [0, 1])
def test_side_bet_evs_match_monte_carlo(seed):
    counts = skewed_pool(seed)
    sb = CFG.side_bets
    n = 400_000
    # draw order: P1, up, P2, hole, third card
    pp = sample_ev(counts, n, lambda d: ret(settle_perfect_pairs(d[0], d[2], sb.perfect_pairs)))
    t3 = sample_ev(counts, n, lambda d: ret(settle_21_3(d[0], d[2], d[1], sb.twenty_one_plus_three)))

    assert abs(pp - A.perfect_pairs_ev(counts, sb.perfect_pairs)) < 0.012
    assert abs(t3 - A.twenty_one_plus_three_ev(counts, sb.twenty_one_plus_three)) < 0.012


def test_crazy7_tier_probabilities_match_monte_carlo():
    """Crazy 7 EV is too noisy for MC (2000:1), so compare tier frequencies on a 7-rich pool."""
    counts = [8] * 52
    for c in range(52):
        if c // 4 != 6:
            counts[c] = 2
    names = {"one_seven": "one", "two_sevens": "pair_unsuited", "two_sevens_suited": "pair_suited",
             "three_sevens": "three_unsuited", "three_sevens_suited": "three_suited"}
    probs = A.crazy_7_probs(counts)
    pool = [c for c in range(52) for _ in range(counts[c])]
    rng = random.Random(5)
    n = 300_000
    freq = {k: 0 for k in probs}
    for _ in range(n):
        d = rng.sample(pool, 5)  # P1, up, P2, hole, third card
        peek_bj = d[1] // 4 == 0 and d[3] // 4 >= 9
        out, _ = settle_crazy_7(d[0], d[2], None if peek_bj else d[4], CFG.side_bets.crazy_7)
        if out:
            freq[names[out]] += 1
    for k, p in probs.items():
        se = (p * (1 - p) / n) ** 0.5
        assert abs(freq[k] / n - p) < 5 * se + 1e-5, (k, freq[k] / n, p)


def test_probabilities_exact_full_shoe():
    full = [8] * 52
    pp = A.perfect_pairs_probs(full)
    assert abs(pp["perfect"] - 7 / 415) < 1e-12
    # published RTPs at a fresh 8-deck shoe
    assert abs(1 + A.perfect_pairs_ev(full, CFG.side_bets.perfect_pairs) - 0.9590) < 0.0001
    assert abs(1 + A.twenty_one_plus_three_ev(full, CFG.side_bets.twenty_one_plus_three) - 0.9630) < 0.0001
    assert abs(1 + A.crazy_7_ev(full, CFG.side_bets.crazy_7) - 0.9477) < 0.0001


def test_dealer_distribution_vs_monte_carlo():
    counts = skewed_pool(3)
    buckets = A.cards_to_buckets(counts)
    dist = A.dealer_distribution(tuple(buckets))
    pool = [min(c // 4, 9) for c in range(52) for _ in range(counts[c])]
    rng = random.Random(2)
    n = 200_000
    bust_by = {}
    bust_up = [0] * 10
    up_n = [0] * 10
    for _ in range(n):
        d = rng.sample(pool, 12)
        hard, ace, k = 0, False, 0
        for b in d:
            hard += b + 1
            ace = ace or b == 0
            k += 1
            t = hard + 10 if ace and hard <= 11 else hard
            if k >= 2 and t >= 17:
                break
        up_n[d[0]] += 1
        if hard > 21:
            bust_by[k] = bust_by.get(k, 0) + 1
            bust_up[d[0]] += 1
    p_bust = sum(dist[u]["p_up"] * dist[u]["bust"] for u in range(10))
    assert abs(sum(bust_by.values()) / n - p_bust) < 0.006
    bb = A.bust_bonus_probs(buckets)["bust_by_cards"]
    for k in (3, 4, 5):
        assert abs(bust_by.get(k, 0) / n - bb[k]) < 0.004
    for u in (1, 5, 9):
        assert abs(bust_up[u] / up_n[u] - dist[u]["bust"]) < 0.02


def test_insurance_and_edge_direction():
    full = A.cards_to_buckets([8] * 52)
    assert abs(A.insurance_ev(full) - (3 * 128 / 416 - 1)) < 1e-12
    rich = full[:]
    for b in range(1, 6):
        rich[b] -= 12  # 60 low cards gone: Hi-Lo TC ~ +8.7
    poor = full[:]
    poor[9] -= 40
    poor[0] -= 10
    sr = SolverRules.from_config(CFG.rules)
    base = A.main_edge_estimate(full, sr)
    assert A.main_edge_estimate(rich, sr) > base + 0.02
    assert A.main_edge_estimate(poor, sr) < base - 0.02
    assert -0.008 < base < 0.0
    assert A.insurance_ev(rich) > 0


def test_table_analytics_fast():
    t = Table(CFG, seed=1)
    A.table_analytics(t)  # warm the edge model
    t.shoe.mark_seen(t.shoe.draw())
    t0 = time.perf_counter()
    a = A.table_analytics(t)
    assert time.perf_counter() - t0 < 0.02
    for k in ("perfect_pairs", "twenty_one_plus_three", "crazy_7", "bust_bonus", "insurance",
              "main_edge_estimate", "p_dealer_bust_by_upcard"):
        assert k in a

