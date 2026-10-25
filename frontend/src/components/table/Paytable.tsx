import { useState } from 'react'
import type { GameConfig } from '../../protocol'

/** Official RTPs from the in-game help (docs/OFFICIAL_RULES_NL.txt). Not part of the config. */
const RTP = { main: '99.59%', twenty_one_plus_three: '96.30%', bust_bonus: '94.12%', perfect_pairs: '95.90%', crazy_7: '94.77%', cashout: '98.00%' }

export function Paytable({ cfg }: { cfg: GameConfig }) {
  const [open, setOpen] = useState(false)
  const sb = cfg.side_bets
  const t = sb.twenty_one_plus_three
  const pp = sb.perfect_pairs
  const c7 = sb.crazy_7
  const bb = Object.entries(sb.bust_bonus).sort((a, b) => Number(b[0]) - Number(a[0]))
  return (
    <div className="paytable-wrap">
      <button className="pt-btn" onClick={() => setOpen((o) => !o)} aria-expanded={open} title="Paytables & rules">
        i
      </button>
      {open && (
        <div className="paytable" role="dialog" aria-label="Paytables">
          <div className="pt-head">
            <b>Paytables</b>
            <button className="btn small ghost" onClick={() => setOpen(false)}>
              Close
            </button>
          </div>
          <div className="pt-grid">
            <PtBlock title="21+3" rtp={RTP.twenty_one_plus_three} rows={[['Suited trips', t.suited_trips], ['Straight flush', t.straight_flush], ['Three of a kind', t.three_of_a_kind], ['Straight', t.straight], ['Flush', t.flush]]} />
            <PtBlock title="Bust Bonus" rtp={RTP.bust_bonus} rows={bb.map(([n, o]) => [`Dealer busts with ${n}${n === '8' ? '+' : ''} cards`, o])} note="Bet returned if you have Blackjack." />
            <PtBlock title="Perfect Pairs" rtp={RTP.perfect_pairs} rows={[['Perfect pair', pp.perfect], ['Coloured pair', pp.coloured], ['Mixed pair', pp.mixed]]} />
            <PtBlock title="Crazy 7" rtp={RTP.crazy_7} rows={[['7-7-7 suited', c7.three_suited], ['7-7-7', c7.three_unsuited], ['7-7 suited', c7.pair_suited], ['7-7', c7.pair_unsuited], ['7', c7.one]]} note="Your own first three cards; the dealer card does not count." />
          </div>
          <div className="pt-foot">
            Main game RTP {RTP.main} · Blackjack pays {cfg.rules.blackjack_payout === 1.5 ? '3:2' : `${cfg.rules.blackjack_payout}:1`} · Insurance{' '}
            {cfg.rules.insurance_payout}:1 · Six Card Charlie wins · Cash-out RTP {((cfg.rules.cashout_rtp ?? 0.98) * 100).toFixed(2)}%
          </div>
        </div>
      )}
    </div>
  )
}

function PtBlock({ title, rtp, rows, note }: { title: string; rtp: string; rows: [string, number][]; note?: string }) {
  return (
    <div className="pt-block">
      <div className="pt-title">
        {title} <span>RTP {rtp}</span>
      </div>
      {rows.map(([k, v]) => (
        <div className="pt-row" key={k}>
          <span>{k}</span>
          <b className="tabular">{v}:1</b>
        </div>
      ))}
      {note && <div className="pt-note">{note}</div>}
    </div>
  )
}
