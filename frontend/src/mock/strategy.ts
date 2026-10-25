import type { PlayAction } from '../protocol'
import { handTotal, valueOf } from './cards'

/**
 * Basic strategy for 8 decks, S17, DAS, no surrender (SPEC rules), plus a small
 * composition-dependent EV calculator (infinite-deck approximation using the current
 * unseen composition) that drives the mock "neural network" outputs.
 */

type Code = 'H' | 'S' | 'D' | 'Ds' | 'P'

/** dealer up value: 2..10, 11 = Ace */
function hard(t: number, d: number): Code {
  if (t <= 8) return 'H'
  if (t === 9) return d >= 3 && d <= 6 ? 'D' : 'H'
  if (t === 10) return d <= 9 ? 'D' : 'H'
  if (t === 11) return d <= 10 ? 'D' : 'H'
  if (t === 12) return d >= 4 && d <= 6 ? 'S' : 'H'
  if (t <= 16) return d <= 6 ? 'S' : 'H'
  return 'S'
}

function soft(t: number, d: number): Code {
  if (t <= 14) return d === 5 || d === 6 ? 'D' : 'H'
  if (t <= 16) return d >= 4 && d <= 6 ? 'D' : 'H'
  if (t === 17) return d >= 3 && d <= 6 ? 'D' : 'H'
  if (t === 18) return d >= 3 && d <= 6 ? 'Ds' : d <= 8 ? 'S' : 'H'
  return 'S'
}

/** pair value 1(A)..10 */
function pair(v: number, d: number): boolean {
  switch (v) {
    case 1:
    case 8:
      return true
    case 10:
    case 5:
      return false
    case 9:
      return d !== 7 && d !== 10 && d !== 11
    case 7:
      return d <= 7
    case 6:
      return d <= 6
    case 4:
      return d === 5 || d === 6
    default:
      return d <= 7 // 2,2 and 3,3
  }
}

export const upValue = (upCard: number) => (valueOf(upCard) === 1 ? 11 : valueOf(upCard))

export function basicAction(cards: number[], upCard: number, canDouble: boolean, canSplit: boolean): PlayAction {
  const d = upValue(upCard)
  if (canSplit && cards.length === 2 && valueOf(cards[0]) === valueOf(cards[1]) && pair(valueOf(cards[0]), d)) {
    return 'split'
  }
  const { total, soft: isSoft } = handTotal(cards)
  const code = isSoft ? soft(total, d) : hard(total, d)
  if (code === 'D') return canDouble ? 'double' : 'hit'
  if (code === 'Ds') return canDouble ? 'double' : 'stand'
  if (code === 'P') return 'split'
  return code === 'H' ? 'hit' : 'stand'
}

// ───────────────────────── composition-dependent EVs ─────────────────────────

/** p[b] for buckets 0=A, 1..8 = 2..9, 9 = T */
export type BucketProbs = number[]

export interface DealerDist {
  /** P(final 17), …, P(final 21) */
  final: number[]
  bust: number
  bj: number
}

/** Dealer final-total distribution given the up-card bucket (S17, no-hole-card-peek on tens, peek on aces). */
export function dealerDist(upBucket: number, p: BucketProbs, hitSoft17 = false): DealerDist {
  const memo = new Map<number, number[]>()
  // returns [17,18,19,20,21,bust]
  const rec = (sum: number, ace: boolean): number[] => {
    const best = ace && sum + 10 <= 21 ? sum + 10 : sum
    const isSoft = ace && sum + 10 <= 21
    if (best > 21) return [0, 0, 0, 0, 0, 1]
    if (best >= 17 && !(hitSoft17 && isSoft && best === 17)) {
      const r = [0, 0, 0, 0, 0, 0]
      r[best - 17] = 1
      return r
    }
    const key = sum * 2 + (ace ? 1 : 0)
    const m = memo.get(key)
    if (m) return m
    const out = [0, 0, 0, 0, 0, 0]
    for (let b = 0; b < 10; b++) {
      if (p[b] <= 0) continue
      const sub = rec(sum + b + 1, ace || b === 0)
      for (let k = 0; k < 6; k++) out[k] += p[b] * sub[k]
    }
    memo.set(key, out)
    return out
  }
  const upVal = upBucket + 1
  const final = [0, 0, 0, 0, 0]
  let bust = 0
  let bj = 0
  // second (hole) card
  let norm = 1
  if (upBucket === 0) norm = 1 - p[9] // peeked: no ten under an ace
  for (let b = 0; b < 10; b++) {
    let pb = p[b]
    if (upBucket === 0 && b === 9) continue
    pb /= norm
    if (upBucket === 9 && b === 0) {
      bj += pb // late-discovered dealer blackjack (no peek on tens)
      continue
    }
    const sub = rec(upVal + b + 1, upBucket === 0 || b === 0)
    for (let k = 0; k < 5; k++) final[k] += pb * sub[k]
    bust += pb * sub[5]
  }
  return { final, bust, bj }
}

function standEv(t: number, dd: DealerDist): number {
  if (t > 21) return -1
  let ev = dd.bust - dd.bj
  for (let k = 0; k < 5; k++) {
    const f = 17 + k
    if (t > f) ev += dd.final[k]
    else if (t < f) ev -= dd.final[k]
  }
  // player totals below 17 lose to every dealer final total
  return ev
}

export interface ActionEvs {
  stand: number
  hit: number
  double: number | null
  split: number | null
}

/** EV per unit of original bet for each action (infinite-deck approximation, composition-weighted). */
export function actionEvs(cards: number[], upBucket: number, p: BucketProbs, canDouble: boolean, canSplit: boolean): ActionEvs {
  const dd = dealerDist(upBucket, p)
  const memo = new Map<number, number>()
  const value = (sum: number, ace: boolean, n: number): number => {
    const best = ace && sum + 10 <= 21 ? sum + 10 : sum
    if (best > 21) return -1
    if (n >= 6) return 1 // six card charlie
    const key = (sum * 2 + (ace ? 1 : 0)) * 8 + n
    const m = memo.get(key)
    if (m !== undefined) return m
    const s = standEv(best, dd)
    let h = 0
    for (let b = 0; b < 10; b++) h += p[b] * value(sum + b + 1, ace || b === 0, n + 1)
    const v = Math.max(s, h)
    memo.set(key, v)
    return v
  }
  let sum = 0
  let ace = false
  for (const c of cards) {
    sum += valueOf(c)
    if (valueOf(c) === 1) ace = true
  }
  const best = ace && sum + 10 <= 21 ? sum + 10 : sum
  const n = cards.length
  const stand = standEv(best, dd)
  let hit = 0
  for (let b = 0; b < 10; b++) hit += p[b] * value(sum + b + 1, ace || b === 0, n + 1)
  let dbl: number | null = null
  if (canDouble) {
    dbl = 0
    for (let b = 0; b < 10; b++) {
      const s2 = sum + b + 1
      const a2 = ace || b === 0
      const b2 = a2 && s2 + 10 <= 21 ? s2 + 10 : s2
      dbl += p[b] * 2 * standEv(b2, dd)
    }
  }
  let split: number | null = null
  if (canSplit && n === 2) {
    const v = valueOf(cards[0])
    let one = 0
    for (let b = 0; b < 10; b++) {
      const s2 = v + b + 1
      const a2 = v === 1 || b === 0
      if (v === 1) {
        one += p[b] * standEv(a2 && s2 + 10 <= 21 ? s2 + 10 : s2, dd)
      } else {
        const st = value(s2, a2, 2)
        let d2 = 0
        for (let c = 0; c < 10; c++) {
          const s3 = s2 + c + 1
          const a3 = a2 || c === 0
          d2 += p[c] * 2 * standEv(a3 && s3 + 10 <= 21 ? s3 + 10 : s3, dd)
        }
        one += p[b] * Math.max(st, d2)
      }
    }
    split = 2 * one
  }
  return { stand, hit, double: dbl, split }
}
