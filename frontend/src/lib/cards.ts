import type { Card } from '../protocol'

export const cardUrl = (code: string) => `/cards/${code}.png`

const VALUES: Record<string, number> = { A: 1, J: 10, Q: 10, K: 10 }
export function cardValue(c: Card): number {
  if (!c.face_up || !c.rank) return 0
  return VALUES[c.rank] ?? Number(c.rank)
}

export function handValue(cards: Card[]): { total: number; soft: boolean } {
  let t = 0
  let aces = 0
  for (const c of cards) {
    const v = cardValue(c)
    t += v
    if (v === 1) aces++
  }
  if (aces && t + 10 <= 21) return { total: t + 10, soft: true }
  return { total: t, soft: false }
}

/** "7/17" for live soft totals, like the real game's badge. */
export function totalLabel(total: number, soft: boolean, final: boolean): string {
  if (soft && !final && total <= 21 && total !== 21) return `${total - 10}/${total}`
  return String(total)
}

export function preloadCards() {
  const ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
  const suits = ['hearts', 'diamonds', 'clubs', 'spades']
  const codes = ['back', ...ranks.flatMap((r) => suits.map((s) => `${r}_of_${s}`))]
  for (const c of codes) {
    const img = new Image()
    img.src = cardUrl(c)
  }
}
