import { useLab, totalOf } from '../../store'
import { EMPTY_BETS, type Action, type Bets } from '../../protocol'
import { money } from '../../lib/format'
import { Chip } from './Chip'

const ACTIONS: { a: Action; label: string; icon: string; cls: string }[] = [
  { a: 'hit', label: 'Hit', icon: '+', cls: 'hit' },
  { a: 'stand', label: 'Stand', icon: '−', cls: 'stand' },
  { a: 'double', label: 'Double', icon: '2×', cls: 'double' },
  { a: 'split', label: 'Split', icon: '⇄', cls: 'split' },
]

export function Dock() {
  const st = useLab((s) => s.state)!
  const hello = useLab((s) => s.hello)!
  const s = useLab()
  const game = st.game
  const legal = game.legal_actions
  const advice = s.adviceOn ? st.ai.advice : null
  const chips = hello.config.limits.chips
  const pendingTotal = totalOf(s.pending)

  if (st.ai.autoplay) {
    return (
      <div className="dock">
        <div className="dock-status">
          {st.ai.agent} is playing
          <button className="btn small" onClick={() => s.send({ cmd: 'autoplay', on: false })}>
            Stop
          </button>
        </div>
      </div>
    )
  }

  if (legal.includes('deal')) {
    const advBets = advice?.kind === 'bet' ? (advice.extra?.bets as Partial<Bets> | undefined) : undefined
    return (
      <div className="dock betting">
        <div className="chip-rack">
          {chips.map((c) => (
            <Chip key={c} value={c} size={34} selected={s.chip === c} onClick={() => s.set('chip', c)} disabled={c > game.balance} />
          ))}
        </div>
        <div className="bet-buttons">
          <button className="btn ghost" onClick={s.clearBets} disabled={pendingTotal === 0}>
            Clear
          </button>
          <button className="btn ghost" onClick={s.rebet} disabled={!s.lastBets}>
            Rebet
          </button>
          <button className="btn ghost" onClick={s.doubleBets} disabled={pendingTotal === 0}>
            ×2
          </button>
          {advBets && (
            <button className="btn ghost" title={`Use the bets recommended by ${st.ai.agent}`} onClick={() => s.set('pending', { ...EMPTY_BETS, ...advBets })}>
              Use AI
            </button>
          )}
        </div>
        <div className="dock-spacer" />
        <button className="deal-btn" onClick={s.deal} disabled={pendingTotal === 0 && !s.lastBets} title="Deal (Space)">
          Deal
        </button>
      </div>
    )
  }

  if (game.phase === 'insurance' && legal.length) {
    const rec = advice?.kind === 'insurance' ? advice.chosen : null
    return (
      <div className="dock actions">
        <span className="dock-q">Insurance {money(game.bets.main / 2)}?</span>
        <ActBtn cls="insure" icon="✓" label="Insure" disabled={!legal.includes('insure')} advised={rec === 'insure'} onClick={() => s.send({ cmd: 'action', action: 'insure' })} />
        <ActBtn cls="noins" icon="✕" label="No" advised={rec === 'no_insurance'} onClick={() => s.send({ cmd: 'action', action: 'no_insurance' })} />
      </div>
    )
  }

  if (game.phase === 'player' && legal.length) {
    const rec = advice?.kind === 'play' ? advice.chosen : null
    const offer = game.hands[game.active_hand]?.cashout_offer
    return (
      <div className="dock actions">
        {ACTIONS.map((x) => (
          <ActBtn
            key={x.a}
            cls={x.cls}
            icon={x.icon}
            label={x.label}
            disabled={!legal.includes(x.a)}
            advised={rec === x.a}
            onClick={() => s.send({ cmd: 'action', action: x.a })}
          />
        ))}
        <ActBtn
          cls="cashout"
          icon="€"
          label={offer != null ? `Cash out ${money(offer)}` : 'Cash out'}
          disabled={!legal.includes('cashout') || offer == null}
          advised={rec === 'cashout'}
          onClick={() => s.send({ cmd: 'action', action: 'cashout' })}
        />
      </div>
    )
  }

  return <div className="dock" />
}

function ActBtn(p: { cls: string; icon: string; label: string; disabled?: boolean; advised?: boolean; onClick(): void }) {
  return (
    <button className={`act-btn ${p.cls} ${p.advised ? 'advised' : ''}`} disabled={p.disabled} onClick={p.onClick}>
      <span className="act-icon">{p.icon}</span>
      <span className="act-label tabular">{p.label}</span>
    </button>
  )
}
