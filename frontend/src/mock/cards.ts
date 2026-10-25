import type { Card, Owner, Rank, RankBucket, Suit } from '../protocol'

/**
 * Mock card encoding: int 0..51. rank index r = c % 13 (0=A, 1=2 … 9=10, 10=J, 11=Q, 12=K),
 * suit index s = floor(c / 13) (0 hearts, 1 diamonds, 2 clubs, 3 spades).
 */
export const RANKS: Rank[] = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
export const SUITS: Suit[] = ['hearts', 'diamonds', 'clubs', 'spades']
export const BUCKETS: RankBucket[] = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T']

export const rankIdx = (c: number) => c % 13
export const suitIdx = (c: number) => Math.floor(c / 13)
/** 0=A, 1..8 = 2..9, 9 = ten-valued */
export const bucketOf = (c: number) => Math.min(9, c % 13)
/** Blackjack value with ace = 1. */
export const valueOf = (c: number) => Math.min(10, (c % 13) + 1)
export const isRed = (c: number) => suitIdx(c) < 2

export function cardJson(c: number, owner: Owner, faceUp = true): Card {
  if (!faceUp) return { code: 'back', face_up: false, owner }
  const rank = RANKS[rankIdx(c)]
  const suit = SUITS[suitIdx(c)]
  return { rank, suit, code: `${rank}_of_${suit}`, face_up: true, owner }
}

export function parseCode(code: string): number {
  const [rank, , suit] = code.split('_')
  const r = RANKS.indexOf(rank as Rank)
  const s = SUITS.indexOf(suit as Suit)
  if (r < 0 || s < 0) throw new Error(`bad card code ${code}`)
  return s * 13 + r
}

export interface Total {
  total: number
  soft: boolean
}

export function handTotal(cards: number[]): Total {
  let t = 0
  let aces = 0
  for (const c of cards) {
    const v = valueOf(c)
    t += v
    if (v === 1) aces++
  }
  if (aces > 0 && t + 10 <= 21) return { total: t + 10, soft: true }
  return { total: t, soft: false }
}
