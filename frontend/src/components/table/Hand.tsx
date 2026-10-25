import type { Card, HandResult, HandState } from '../../protocol'
import { handValue, totalLabel } from '../../lib/cards'
import { signedMoney } from '../../lib/format'
import { PlayingCard } from './PlayingCard'

const RESULT_LABEL: Record<HandResult, string> = {
  win: 'WIN',
  lose: 'LOSE',
  push: 'PUSH',
  blackjack: 'BLACKJACK',
  charlie: '6-CARD CHARLIE',
  cashout: 'CASHED OUT',
}

interface HandViewProps {
  hand: HandState
  crowdCards: Card[]
  round: number
  width: number
  active: boolean
  split: boolean
  index: number
  dealMs: number
  phaseFinal: boolean
}

const CROWD_TIP =
  'Crowd card — ONE Blackjack is a shared hand. You had already stood, but other players at the table kept hitting, so the dealer kept drawing to the shared hand. Not part of your hand, but it is a seen card (it counts).'

export function HandView({ hand, crowdCards, round, width, active, split, index, dealMs, phaseFinal }: HandViewProps) {
  const ownCards = hand.cards
  // Robustness: if the backend already appended crowd cards to the hand (owner:"crowd"), don't add them twice.
  const hasInlineCrowd = ownCards.some((c) => c.owner === 'crowd')
  const all = hasInlineCrowd ? ownCards : [...ownCards, ...crowdCards]
  const offX = Math.round(width * 0.3)
  const offY = Math.round(width * 0.13)
  const n = all.length
  const stackW = width + (n - 1) * offX
  const stackH = Math.round(width * 1.452) + (n - 1) * offY

  const isBj = hand.status === 'blackjack' || hand.result === 'blackjack'
  const bust = hand.status === 'bust' || hand.total > 21
  const charlie = hand.status === 'charlie' || hand.result === 'charlie'
  const final = phaseFinal || hand.status !== 'active'
  const badge = isBj ? 'BJ' : charlie ? '6★' : totalLabel(hand.total, hand.soft, final)
  const crowdOnly = all.filter((c) => c.owner === 'crowd')
  const crowdTotal = crowdOnly.length ? handValue(all) : null

  return (
    <div className={`hand ${active ? 'active' : ''} ${split ? 'split' : ''}`} style={{ width: stackW + 40 }}>
      {hand.result && (
        <div className={`result-chip r-${hand.result}`}>
          {RESULT_LABEL[hand.result]}
          {(hand.result === 'win' || hand.result === 'blackjack' || hand.result === 'charlie') && <span>{signedMoney(hand.payout - hand.bet)}</span>}
          {hand.result === 'cashout' && <span>€{hand.payout.toFixed(2)}</span>}
        </div>
      )}
      <div className="hand-stack" style={{ width: stackW, height: stackH }}>
        {n > 0 && (
          <div className={`total-badge ${isBj ? 'bj' : ''} ${bust ? 'bust' : ''} ${charlie ? 'charlie' : ''}`} style={{ top: stackH - Math.round(width * 1.452) / 2 - 13 }}>
            {badge}
          </div>
        )}
        {all.map((c, i) => {
          const crowd = c.owner === 'crowd'
          return (
            <div
              key={`${round}-${index}-${i}`}
              className={`hand-slot ${crowd ? 'crowd' : ''}`}
              style={{ left: i * offX, top: (n - 1 - i) * offY, zIndex: i + 1 }}
              title={crowd ? CROWD_TIP : undefined}
            >
              <PlayingCard card={c} round={round} width={width} dealMs={dealMs} />
            </div>
          )
        })}
        {crowdTotal && (
          <div className="crowd-total" style={{ left: stackW + 6, top: 0 }} title={CROWD_TIP}>
            crowd {crowdTotal.total > 21 ? 'bust' : crowdTotal.total}
          </div>
        )}
      </div>
    </div>
  )
}

/** Mini "crowd hand" shown next to split hands: the original pair plus the crowd's hits. */
export function CrowdHand({ pair, crowdCards, round, width, dealMs }: { pair: Card[]; crowdCards: Card[]; round: number; width: number; dealMs: number }) {
  const all = [...pair, ...crowdCards]
  const offX = Math.round(width * 0.3)
  const offY = Math.round(width * 0.13)
  const n = all.length
  const h = Math.round(width * 1.452)
  const t = handValue(all)
  return (
    <div className="hand crowd-hand" title="The crowd keeps playing the unsplit shared hand before your split hands are dealt.">
      <div className="hand-stack" style={{ width: width + (n - 1) * offX, height: h + (n - 1) * offY }}>
        <div className="total-badge crowdb" style={{ top: (n - 1) * offY + h / 2 - 13 }}>
          {t.total > 21 ? 'X' : t.total}
        </div>
        {all.map((c, i) => (
          <div key={`${round}-crowd-${i}`} className="hand-slot crowd" style={{ left: i * offX, top: (n - 1 - i) * offY, zIndex: i + 1 }}>
            <PlayingCard card={c} round={round} width={width} dealMs={dealMs} />
          </div>
        ))}
      </div>
    </div>
  )
}
