import { useEffect, useRef, useState } from 'react'
import { useLab, totalOf } from '../../store'
import type { BetKey, Bets, GameState } from '../../protocol'
import { money } from '../../lib/format'
import { totalLabel } from '../../lib/cards'
import { PlayingCard } from './PlayingCard'
import { CrowdHand, HandView } from './Hand'
import { BetSpots } from './BetSpots'
import { Dock } from './Dock'
import { Felt, ShoeDevice } from './Felt'
import { Paytable } from './Paytable'
import './table.css'

/** Measures the stage so card sizes scale with the available space. */
function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [size, setSize] = useState({ w: 800, h: 700 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, size] as const
}

/** The single central status line, as in the real game. */
function statusLine(g: GameState, pending: number): string | null {
  const dealing = g.legal_actions.length === 0
  switch (g.phase) {
    case 'betting':
      return 'PLACE YOUR BETS'
    case 'insurance':
      return 'INSURANCE?'
    case 'player':
      return dealing ? null : 'MAKE YOUR DECISION'
    case 'crowd':
    case 'dealer':
      return null
    case 'settled':
      if (pending > 0) return 'PLACE YOUR BETS'
      if (g.round_net !== null && g.round_net > 0.004) return `YOU WIN ${money(g.round_net)}`
      return 'WAIT FOR THE NEXT GAME'
  }
}

export function GameTable() {
  const st = useLab((s) => s.state)!
  const hello = useLab((s) => s.hello)!
  const pending = useLab((s) => s.pending)
  const addChip = useLab((s) => s.addChip)
  const removeChip = useLab((s) => s.removeChip)
  const autoplayDelay = useLab((s) => s.autoplayDelay)
  const adviceOn = useLab((s) => s.adviceOn)
  const [stageRef, size] = useSize<HTMLDivElement>()

  const g = st.game
  const shoe = st.shoe
  const round = g.round_id
  const autoplay = st.ai.autoplay
  const dealMs = autoplay ? Math.max(90, Math.min(380, autoplayDelay * 0.45)) : 380

  const cardW = Math.round(Math.max(56, Math.min(92, size.h * 0.112, size.w * 0.1)))
  const dealerW = Math.round(cardW * 0.92)

  const betweenRounds = g.phase === 'betting' || g.phase === 'settled'
  const editable = betweenRounds && g.legal_actions.includes('deal') && !autoplay
  const pendingTotal = totalOf(pending)
  const showPending = betweenRounds && (pendingTotal > 0 || g.phase === 'betting')
  const displayBets: Bets = showPending ? pending : g.bets
  const mainBet = g.hands.reduce((a, h) => a + h.bet, 0)
  const mainPayout = g.hands.reduce((a, h) => a + h.payout, 0)
  const mainResult = g.phase === 'settled' && !showPending && g.hands.length ? { bet: mainBet, payout: mainPayout } : null
  // side-bet EVs only as a hover hint, and only when advice is on
  const sideEv: Partial<Record<BetKey, number>> = adviceOn
    ? {
        perfect_pairs: shoe.analytics.perfect_pairs,
        twenty_one_plus_three: shoe.analytics.twenty_one_plus_three,
        crazy_7: shoe.analytics.crazy_7,
        bust_bonus: shoe.analytics.bust_bonus,
      }
    : {}
  const adviceBets = adviceOn && st.ai.advice?.kind === 'bet' ? (st.ai.advice.extra?.bets as Partial<Bets> | undefined) : undefined

  const split = g.hands.length > 1
  const status = statusLine(g, pendingTotal)
  const dealerFinal = g.phase === 'settled' || g.phase === 'dealer'
  const dealerTop = Math.round(size.h * 0.12)
  const statusTop = dealerTop + Math.round(dealerW * 1.452) + 40

  return (
    <main className="table-col">
      <div className="stage" ref={stageRef} style={{ ['--spot-gap' as string]: `${Math.min(118, size.w * 0.135)}px` }}>
        <Felt />
        <ShoeDevice side="left" half={1} shoe={shoe} />
        <ShoeDevice side="right" half={2} shoe={shoe} />

        <div className="table-info">
          <div className="saldo tabular">
            <div>
              <div className="saldo-k">SALDO</div>
              <div className="saldo-v">{money(g.balance)}</div>
            </div>
            <div className="saldo-sep" />
            <div>
              <div className="saldo-k">TOTALE INZET</div>
              <div className="saldo-v">{money(betweenRounds && showPending ? pendingTotal : g.total_bet)}</div>
            </div>
          </div>
          <div className="ti-row">
            <Paytable cfg={hello.config} />
            <span className="ti-limits tabular">
              €{hello.config.limits.main_min}–{hello.config.limits.main_max.toLocaleString()}
            </span>
          </div>
        </div>

        <div className="dealer-area" style={{ top: dealerTop }}>
          {g.dealer.cards.length > 0 && (
            <div className={`total-badge dealer ${g.dealer.bust ? 'bust' : ''}`}>
              {g.dealer.blackjack ? 'BJ' : totalLabel(g.dealer.total, g.dealer.soft, dealerFinal)}
            </div>
          )}
          <div className="dealer-cards">
            {g.dealer.cards.map((c, i) => (
              <div key={`${round}-d-${i}`} className="dealer-slot" style={{ marginLeft: i ? -Math.round(dealerW * 0.18) : 0, zIndex: i }}>
                <PlayingCard card={c} round={round} width={dealerW} dealMs={dealMs} />
              </div>
            ))}
          </div>
        </div>

        {g.crowd.enabled && (
          <div
            className="crowd-counter tabular"
            style={{ top: Math.round(size.h * 0.27) }}
            title="Shared hand: players who still want a card / players who stand. The shared hand keeps receiving cards while anyone hits."
          >
            <PersonIcon /> {g.crowd.hitters}
            <HitIcon />
            <span className="cc-sep" />
            {g.crowd.standers}
            <StandIcon />
          </div>
        )}

        {status && (
          <div className="status-line" style={{ top: statusTop }} key={status}>
            {status}
          </div>
        )}

        <div className="hands-area" style={{ bottom: Math.round(Math.max(150, size.h * 0.235)) }}>
          {g.hands.map((h, i) => (
            <HandView
              key={`${round}-h${i}`}
              hand={h}
              index={i}
              crowdCards={split ? [] : g.crowd.cards}
              round={round}
              width={split ? Math.round(cardW * 0.9) : cardW}
              active={split && i === g.active_hand && g.phase === 'player'}
              split={split}
              dealMs={dealMs}
              phaseFinal={g.phase === 'settled' || g.phase === 'dealer'}
            />
          ))}
          {split && g.crowd.cards.length > 0 && (
            <CrowdHand pair={[g.hands[0].cards[0], g.hands[1].cards[0]]} crowdCards={g.crowd.cards} round={round} width={Math.round(cardW * 0.62)} dealMs={dealMs} />
          )}
        </div>

        <BetSpots
          bets={displayBets}
          ghost={!showPending && betweenRounds}
          editable={editable}
          chips={hello.config.limits.chips}
          results={showPending ? [] : g.side_bet_results}
          mainResult={mainResult}
          ev={sideEv}
          onAdd={(k) => addChip(k)}
          onRemove={(k) => removeChip(k)}
          advice={editable ? adviceBets : undefined}
        />
      </div>
      <Dock />
    </main>
  )
}

function PersonIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 16 16" aria-hidden>
      <circle cx="8" cy="5" r="3.2" fill="currentColor" />
      <path d="M2 15c0-3.3 2.7-5.5 6-5.5s6 2.2 6 5.5z" fill="currentColor" />
    </svg>
  )
}
function HitIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 16 16" aria-label="hit">
      <path d="M8 2v12M2 8h12" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
    </svg>
  )
}
function StandIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 16 16" aria-label="stand">
      <path
        d="M5 14V6.5a1 1 0 0 1 2 0V5a1 1 0 0 1 2 0v.5a1 1 0 0 1 2 0V7a1 1 0 0 1 2 0v4c0 2-1.5 3.5-3.5 3.5H8C6.5 14.5 5.5 14 5 14z"
        fill="currentColor"
      />
    </svg>
  )
}
