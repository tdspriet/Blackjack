import { useLayoutEffect, useRef } from 'react'
import type { Card } from '../../protocol'
import { cardUrl } from '../../lib/cards'

/** Cards that already played their deal-in animation (per round), so re-layouts (e.g. split) don't replay it. */
const dealt = new Set<string>()
let dealtRound = -1

export interface PlayingCardProps {
  card: Card
  round: number
  width: number
  style?: React.CSSProperties
  className?: string
  title?: string
  dealMs?: number
}

export function PlayingCard({ card, round, width, style, className, title, dealMs = 380 }: PlayingCardProps) {
  const ref = useRef<HTMLDivElement>(null)
  const prevCode = useRef<string | null>(null)
  const height = Math.round(width * 1.452)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    if (round !== dealtRound) {
      dealt.clear()
      dealtRound = round
    }
    const key = `${round}:${card.code}:${card.owner ?? ''}`
    const prev = prevCode.current
    prevCode.current = card.code
    // hole card turned over
    if (prev === 'back' && card.code !== 'back') {
      dealt.add(key)
      el.animate(
        [
          { transform: 'perspective(600px) rotateY(90deg) scale(1.04)', filter: 'brightness(1.3)' },
          { transform: 'perspective(600px) rotateY(0deg)', filter: 'none' },
        ],
        { duration: Math.max(120, dealMs * 0.9), easing: 'cubic-bezier(.2,.7,.2,1)' },
      )
      return
    }
    if (prev !== null || dealt.has(key)) return
    dealt.add(key)
    const origin = document.querySelector('[data-deal-origin="active"]')
    if (!origin || dealMs <= 0) return
    const o = origin.getBoundingClientRect()
    const r = el.getBoundingClientRect()
    const dx = o.left + o.width / 2 - (r.left + r.width / 2)
    const dy = o.top + o.height / 2 - (r.top + r.height / 2)
    el.animate(
      [
        { transform: `translate(${dx}px, ${dy}px) rotate(-28deg) scale(0.45)`, opacity: 0.2 },
        { opacity: 1, offset: 0.35 },
        { transform: 'translate(0,0) rotate(0) scale(1)', opacity: 1 },
      ],
      { duration: dealMs, easing: 'cubic-bezier(.25,.8,.25,1)', fill: 'backwards' },
    )
  }, [card.code, card.owner, round, dealMs])

  return (
    <div
      ref={ref}
      className={`pcard ${card.face_up ? '' : 'down'} ${className ?? ''}`}
      style={{ width, height, ...style }}
      title={title}
    >
      <img src={cardUrl(card.face_up ? card.code : 'back')} alt={card.face_up ? card.code.replace(/_/g, ' ') : 'face-down card'} draggable={false} />
    </div>
  )
}
