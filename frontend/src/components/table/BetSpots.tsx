import { BET_LABELS, type BetKey, type Bets, type SideBetResult } from '../../protocol'
import { signedPct } from '../../lib/format'
import { Chip } from './Chip'

const SPOTS: { key: BetKey; label: string[]; x: number; y: number; size: number }[] = [
  { key: 'bust_bonus', label: ['BUST', 'BONUS'], x: -2, y: -40, size: 70 },
  { key: 'twenty_one_plus_three', label: ['21+3'], x: -1, y: -12, size: 74 },
  { key: 'main', label: ['MAIN'], x: 0, y: 0, size: 104 },
  { key: 'perfect_pairs', label: ['PERFECT', 'PAIRS'], x: 1, y: -12, size: 74 },
  { key: 'crazy_7', label: ['CRAZY', '7'], x: 2, y: -40, size: 70 },
]

interface Props {
  bets: Bets
  ghost: boolean
  editable: boolean
  chips: number[]
  results: SideBetResult[]
  mainResult: { bet: number; payout: number } | null
  ev: Partial<Record<BetKey, number>>
  onAdd(k: BetKey): void
  onRemove(k: BetKey): void
  /** AI-recommended bets (advice mode) shown as small tags on the spots. */
  advice?: Partial<Bets>
}

export function BetSpots({ bets, ghost, editable, chips, results, mainResult, ev, onAdd, onRemove, advice }: Props) {
  return (
    <div className="spots">
      {SPOTS.map((s) => {
        const amount = bets[s.key]
        const res = s.key === 'main' ? null : results.find((r) => r.bet === s.key)
        const won = s.key === 'main' ? !!mainResult && mainResult.payout > mainResult.bet : !!res && res.payout > res.stake
        const lost = s.key === 'main' ? !!mainResult && mainResult.payout < mainResult.bet && mainResult.payout === 0 : !!res && res.payout === 0
        const e = ev[s.key]
        const tip =
          s.key === 'main'
            ? 'Main bet'
            : `${BET_LABELS[s.key]}${e !== undefined ? ` — EV at the current composition ${signedPct(e)}` : ''}`
        return (
          <div
            key={s.key}
            className={`spot spot-${s.key} ${editable ? 'editable' : ''} ${amount ? 'has-bet' : ''} ${won ? 'won' : ''} ${lost ? 'lost' : ''} ${ghost ? 'ghost' : ''}`}
            style={{
              width: s.size,
              height: s.size,
              transform: `translate(calc(${s.x} * var(--spot-gap) - 50%), ${s.y}px)`,
            }}
            title={editable ? `${tip}\nClick: add chip · Right-click / Shift-click: remove` : tip}
            onClick={(ev2) => {
              if (!editable) return
              if (ev2.shiftKey) onRemove(s.key)
              else onAdd(s.key)
            }}
            onContextMenu={(ev2) => {
              ev2.preventDefault()
              if (editable) onRemove(s.key)
            }}
            role={editable ? 'button' : undefined}
            aria-label={`${s.label.join(' ')} bet €${amount}`}
          >
            <div className="spot-ring" />
            {!amount && (
              <div className="spot-label">
                {s.label.map((l) => (
                  <span key={l}>{l}</span>
                ))}
              </div>
            )}
            {amount > 0 && (
              <div className="spot-chip">
                <Chip value={amount} chips={chips} size={s.key === 'main' ? 56 : 44} />
              </div>
            )}
            {won && (
              <div className="spot-win tabular">
                +€{s.key === 'main' ? (mainResult!.payout - mainResult!.bet).toFixed(0) : res!.payout.toFixed(0)}
              </div>
            )}
            {advice?.[s.key] ? (
              <div className="spot-ai tabular" title="AI-recommended bet for this spot">
                AI €{advice[s.key]}
              </div>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}
