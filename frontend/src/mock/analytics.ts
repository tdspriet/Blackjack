import type { GameConfig, RankBucket, ShoeAnalytics } from '../protocol'
import { BUCKETS } from './cards'
import { dealerDist, type BucketProbs } from './strategy'

/** counts[c] for card ids 0..51 (8 per id in a fresh 8-deck shoe) → bucket probabilities */
export function bucketProbs(counts: number[]): BucketProbs {
  const p = new Array(10).fill(0)
  let n = 0
  for (let c = 0; c < 52; c++) {
    p[Math.min(9, c % 13)] += counts[c]
    n += counts[c]
  }
  return p.map((x) => (n > 0 ? x / n : 0.1))
}

export function perfectPairsEv(counts: number[], cfg: GameConfig): number {
  const pt = cfg.side_bets.perfect_pairs
  let N = 0
  for (const x of counts) N += x
  const denom = N * (N - 1)
  let perfect = 0
  let coloured = 0
  let mixed = 0
  for (let r = 0; r < 13; r++) {
    for (let s1 = 0; s1 < 4; s1++) {
      const a = counts[s1 * 13 + r]
      for (let s2 = 0; s2 < 4; s2++) {
        const b = counts[s2 * 13 + r]
        if (s1 === s2) perfect += a * (a - 1)
        else if (s1 < 2 === s2 < 2) coloured += a * b
        else mixed += a * b
      }
    }
  }
  perfect /= denom
  coloured /= denom
  mixed /= denom
  const win = perfect + coloured + mixed
  return pt.perfect * perfect + pt.coloured * coloured + pt.mixed * mixed - (1 - win)
}

export function classify21p3(r: number[], s: number[]): keyof GameConfig['side_bets']['twenty_one_plus_three'] | null {
  const flush = s[0] === s[1] && s[1] === s[2]
  const trips = r[0] === r[1] && r[1] === r[2]
  if (trips) return flush ? 'suited_trips' : 'three_of_a_kind'
  const sorted = [...r].sort((a, b) => a - b)
  const distinct = sorted[0] !== sorted[1] && sorted[1] !== sorted[2]
  const straight =
    distinct && (sorted[2] - sorted[0] === 2 || (sorted[0] === 0 && sorted[1] === 11 && sorted[2] === 12))
  if (straight && flush) return 'straight_flush'
  if (straight) return 'straight'
  if (flush) return 'flush'
  return null
}

export function twentyOnePlus3Ev(counts: number[], cfg: GameConfig): number {
  const pt = cfg.side_bets.twenty_one_plus_three
  let N = 0
  for (const x of counts) N += x
  const denom = N * (N - 1) * (N - 2)
  let ev = 0
  let pWin = 0
  const rs = [0, 0, 0]
  const ss = [0, 0, 0]
  for (let i = 0; i < 52; i++) {
    const ni = counts[i]
    if (!ni) continue
    for (let j = i; j < 52; j++) {
      const nj = counts[j]
      if (!nj) continue
      for (let k = j; k < 52; k++) {
        const nk = counts[k]
        if (!nk) continue
        let ways: number
        if (i === j && j === k) ways = ni * (ni - 1) * (ni - 2)
        else if (i === j) ways = 3 * ni * (ni - 1) * nk
        else if (j === k) ways = 3 * ni * nj * (nj - 1)
        else ways = 6 * ni * nj * nk
        if (ways <= 0) continue
        rs[0] = i % 13
        rs[1] = j % 13
        rs[2] = k % 13
        ss[0] = (i / 13) | 0
        ss[1] = (j / 13) | 0
        ss[2] = (k / 13) | 0
        const cls = classify21p3(rs, ss)
        if (cls) {
          const p = ways / denom
          pWin += p
          ev += p * pt[cls]
        }
      }
    }
  }
  return ev - (1 - pWin)
}

export function crazy7Ev(counts: number[], cfg: GameConfig): number {
  const pt = cfg.side_bets.crazy_7
  let N = 0
  for (const x of counts) N += x
  const s7 = [0, 1, 2, 3].map((s) => counts[s * 13 + 6])
  const n7 = s7.reduce((a, b) => a + b, 0)
  const d1 = N
  const d2 = N * (N - 1)
  const d3 = N * (N - 1) * (N - 2)
  const P1 = n7 / d1
  const P2 = (n7 * (n7 - 1)) / d2
  const P2s = s7.reduce((a, x) => a + x * (x - 1), 0) / d2
  const T = (n7 * (n7 - 1) * (n7 - 2)) / d3
  const Ts = s7.reduce((a, x) => a + x * (x - 1) * (x - 2), 0) / d3
  const A = s7.reduce((a, x) => a + x * (x - 1) * (n7 - 2), 0) / d3 // first two suited & third a 7
  const one = P1 - P2
  const pairS = P2s - A
  const pairU = P2 - P2s - (T - A)
  const threeU = T - Ts
  return pt.one * one + pt.pair_unsuited * pairU + pt.pair_suited * pairS + pt.three_unsuited * threeU + pt.three_suited * Ts - (1 - P1)
}

/** P(dealer busts with exactly n cards) for n = 2..9 (index n), infinite-deck from bucket probs. */
export function dealerBustByCards(p: BucketProbs): number[] {
  const out = new Array(12).fill(0)
  const rec = (sum: number, ace: boolean, n: number, prob: number) => {
    if (prob < 1e-9) return
    const best = ace && sum + 10 <= 21 ? sum + 10 : sum
    if (best > 21) {
      out[Math.min(11, n)] += prob
      return
    }
    if (best >= 17 && n >= 2) return
    for (let b = 0; b < 10; b++) {
      // dealer blackjack (A + T in first two) never busts — handled by best=21 stand
      rec(sum + b + 1, ace || b === 0, n + 1, prob * p[b])
    }
  }
  rec(0, false, 0, 1)
  return out
}

export function bustBonusEv(p: BucketProbs, cfg: GameConfig): number {
  const pt = cfg.side_bets.bust_bonus
  const by = dealerBustByCards(p)
  let ev = 0
  let pBust = 0
  for (let n = 3; n < by.length; n++) {
    const odds = pt[String(Math.min(8, n))] ?? 0
    ev += by[n] * odds
    pBust += by[n]
  }
  return ev - (1 - pBust)
}

export function computeAnalytics(counts: number[], cfg: GameConfig, hiloTc: number, skip21p3 = false): ShoeAnalytics {
  const p = bucketProbs(counts)
  const byUp: Partial<Record<RankBucket, number>> = {}
  for (let b = 0; b < 10; b++) byUp[BUCKETS[b]] = dealerDist(b, p).bust
  const pT = p[9]
  return {
    perfect_pairs: perfectPairsEv(counts, cfg),
    twenty_one_plus_three: skip21p3 ? NaN : twentyOnePlus3Ev(counts, cfg),
    crazy_7: crazy7Ev(counts, cfg),
    bust_bonus: bustBonusEv(p, cfg),
    insurance: pT * (1 + cfg.rules.insurance_payout) - 1,
    main_edge_estimate: -0.0072 + 0.005 * hiloTc,
    p_dealer_bust_by_upcard: byUp,
  }
}
