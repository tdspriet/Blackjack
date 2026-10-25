import type { ShoeState } from '../../protocol'
import { useLab } from '../../store'

export function Felt() {
  return (
    <>
      <div className="studio" />
      <div className="felt" />
      <svg className="felt-text" viewBox="0 0 1000 700" preserveAspectRatio="xMidYMid meet" aria-hidden>
        <defs>
          <path id="arcL1" d="M 120 300 Q 250 352 395 342" />
          <path id="arcL2" d="M 135 330 Q 262 380 395 370" />
          <path id="arcR1" d="M 605 342 Q 750 352 880 300" />
          <path id="arcR2" d="M 605 370 Q 738 380 865 330" />
        </defs>
        <g className="ft">
          <text>
            <textPath href="#arcL1" startOffset="50%" textAnchor="middle">
              Blackjack pays 3 to 2
            </textPath>
          </text>
          <text>
            <textPath href="#arcL2" startOffset="50%" textAnchor="middle">
              6 player cards win
            </textPath>
          </text>
          <text>
            <textPath href="#arcR1" startOffset="50%" textAnchor="middle">
              Dealer must draw at 16
            </textPath>
          </text>
          <text>
            <textPath href="#arcR2" startOffset="50%" textAnchor="middle">
              and stand on all 17&apos;s
            </textPath>
          </text>
        </g>
      </svg>
    </>
  )
}

/** One of the two acrylic shoe containers (half 1 left, half 2 right). The active one is the deal origin. */
export function ShoeDevice({ side, half, shoe }: { side: 'left' | 'right'; half: number; shoe: ShoeState }) {
  const memo = useLab((s) => s.halfMemo)
  const active = shoe.half === half
  const size = shoe.half_size
  let remaining: number
  let cutLeft: number | null = null
  let state: string
  if (active) {
    remaining = size - shoe.dealt_in_half
    cutLeft = Math.max(0, shoe.cut_at - shoe.dealt_in_half)
    state = cutLeft > 0 ? `${cutLeft} to cut` : 'cut card out'
  } else if (half === 2) {
    remaining = size
    state = 'waiting'
  } else {
    const dealt = memo?.dealt ?? Math.round(size * 0.75)
    remaining = size - dealt
    state = 'residue · unseen'
  }
  const frac = Math.max(0, Math.min(1, remaining / size))
  const cutFrac = cutLeft !== null && cutLeft > 0 ? Math.max(0, Math.min(1, (remaining - cutLeft) / size)) : null
  return (
    <div className={`shoe-dev ${side} ${active ? 'active' : ''}`} title={`Half ${half}: ${remaining} cards in the container (${state})`}>
      <div className="sd-box" data-deal-origin={active ? 'active' : undefined}>
        <div className="sd-stack" style={{ width: `${frac * 100}%` }} />
        {cutFrac !== null && <div className="sd-cut" style={{ left: `${cutFrac * 100}%` }} />}
        {!active && half === 1 && <div className="sd-cut idle" />}
      </div>
    </div>
  )
}
