import { useLab } from '../../store'
import { COUNT_LABELS, COUNT_SYSTEMS, RANK_BUCKETS, type RankBucket, type ShoeState } from '../../protocol'
import { signed, signedPct, pct } from '../../lib/format'
import { diverging } from '../../lib/colors'
import { LineChart, type Marker } from '../charts/LineChart'
import './count.css'

export function CountPanel() {
  return (
    <aside className="side-panel right">
      <div className="panel-scroll">
        <CountReadout />
        <ShoeViz />
        <Composition />
        <Analytics />
        <CountHistory />
      </div>
    </aside>
  )
}

function CountReadout() {
  const sys = useLab((s) => s.countSystem)
  const set = useLab((s) => s.set)
  const shoe = useLab((s) => s.state!.shoe)
  const total = useLab((s) => (s.hello?.config.shoe.num_decks ?? 8) * 52)
  const rc = shoe.running[sys] ?? 0
  const tc = shoe.true[sys] ?? 0
  const seen = total - shoe.cards_unseen
  return (
    <div className="section">
      <div className="seg sys-seg">
        {COUNT_SYSTEMS.map((k) => (
          <button key={k} className={sys === k ? 'on' : ''} onClick={() => set('countSystem', k)}>
            {k === 'wong_halves' ? 'Halves' : COUNT_LABELS[k]}
          </button>
        ))}
      </div>
      <div className="readout">
        <div>
          <div className="ro-label">Running count</div>
          <div className="ro-value tabular">{signed(rc, Number.isInteger(rc) ? 0 : 1)}</div>
        </div>
        <div title="Running count ÷ decks unseen (all unseen cards, both halves)">
          <div className="ro-label">True count</div>
          <div className={`ro-value tabular ${tc >= 1 ? 'pos' : tc <= -1 ? 'neg' : ''}`}>{signed(tc, 1)}</div>
        </div>
      </div>
      <div className="ro-line tabular">
        {shoe.decks_unseen.toFixed(2)} decks unseen · {pct(seen / total, 0)} penetration
      </div>
    </div>
  )
}

// ───────────────────────── shoe visual ─────────────────────────

function halfOneDealt(shoe: ShoeState, total: number, memoDealt: number | undefined): number {
  if (memoDealt !== undefined) return memoDealt
  // derive: everything seen minus what was seen from half 2 (+ the burn card of half 1)
  const seen = total - shoe.cards_unseen
  const seen2 = Math.max(0, shoe.dealt_in_half - shoe.burned)
  return Math.max(0, Math.min(shoe.half_size, seen - seen2 + shoe.burned))
}

function ShoeViz() {
  const shoe = useLab((s) => s.state!.shoe)
  const memo = useLab((s) => s.halfMemo)
  const total = useLab((s) => (s.hello?.config.shoe.num_decks ?? 8) * 52)
  const size = shoe.half_size
  const halves = [1, 2].map((h) => {
    if (h === shoe.half) return { h, dealt: shoe.dealt_in_half, cut: shoe.cut_at, state: 'dealing' as const }
    if (h === 1) return { h, dealt: halfOneDealt(shoe, total, memo?.dealt), cut: memo?.cutAt ?? null, state: 'done' as const }
    return { h, dealt: 0, cut: null, state: 'waiting' as const }
  })
  return (
    <div className="section">
      <div className="section-head">
        <span className="section-title">
          Shoe {shoe.shoe_number}
          <span
            className="hint"
            title={`Both halves come from the same 8-deck shuffle, so the count carries over between halves. Cards behind half 1's cut card are never seen; they stay in the unseen pool, so the true count divides by all ${shoe.cards_unseen} unseen cards. Dark red = burn card, light = dealt, hatched = behind the cut card, red line = cut card.`}
          >
            ?
          </span>
        </span>
      </div>
      {halves.map(({ h, dealt, cut, state }) => {
        const burn = shoe.burned
        const residue = state === 'done' ? size - dealt : 0
        const toCut = state === 'dealing' && cut !== null ? Math.max(0, cut - dealt) : null
        return (
          <div key={h} className={`half ${state}`}>
            <div className="half-head">
              <span className="half-name">Half {h}</span>
              <span className="half-state tabular">
                {state === 'dealing' && (toCut! > 0 ? `${toCut} to cut` : 'cut card out')}
                {state === 'done' && `${residue} never seen`}
                {state === 'waiting' && 'waiting'}
              </span>
            </div>
            <div className="half-bar" role="img" aria-label={`Half ${h}`}>
              {state !== 'waiting' && (
                <>
                  <div className="hb-seg burn" style={{ left: 0, width: `${(burn / size) * 100}%` }} title={`${burn} burned (unseen)`} />
                  <div className="hb-seg dealt" style={{ left: `${(burn / size) * 100}%`, width: `${(Math.max(0, dealt - burn) / size) * 100}%` }} title={`${Math.max(0, dealt - burn)} dealt (seen)`} />
                  {state === 'done' && <div className="hb-seg residue" style={{ left: `${(dealt / size) * 100}%`, width: `${(residue / size) * 100}%` }} title={`${residue} cards behind the cut card — never seen`} />}
                  {state === 'dealing' && cut !== null && (
                    <div className="hb-seg behind" style={{ left: `${(cut / size) * 100}%`, width: `${((size - cut) / size) * 100}%` }} title={`${size - cut} cards behind the cut card will not be dealt`} />
                  )}
                  {cut !== null && (
                    <div className="hb-cut" style={{ left: `${(cut / size) * 100}%` }} title={`cut card after ${cut} cards`}>
                    </div>
                  )}
                </>
              )}
              {state === 'waiting' && <div className="hb-seg waiting" style={{ left: 0, width: '100%' }} />}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ───────────────────────── composition ─────────────────────────

function Composition() {
  const shoe = useLab((s) => s.state!.shoe)
  const decks = useLab((s) => s.hello?.config.shoe.num_decks ?? 8)
  const total = decks * 52
  const rows = RANK_BUCKETS.map((b: RankBucket) => {
    const full = (b === 'T' ? 16 : 4) * decks
    const exp = (full * shoe.cards_unseen) / total
    const unseen = shoe.unseen_counts[b] ?? 0
    const rel = exp > 0 ? (unseen - exp) / exp : 0
    return { b, unseen, exp, rel, diff: unseen - exp }
  })
  const scale = Math.max(0.08, ...rows.map((r) => Math.abs(r.rel)))
  return (
    <div className="section">
      <div className="section-head">
        <span className="section-title" title="Unseen cards per rank vs. expected at this depth. Orange = rich, blue = depleted; tens and aces rich favour the player.">
          Remaining composition
        </span>
      </div>
      <div className="comp">
        {rows.map((r) => {
          const w = (Math.abs(r.rel) / scale) * 50
          return (
            <div className="comp-row" key={r.b} title={`${r.b}: ${r.unseen} unseen, ${r.exp.toFixed(1)} expected (${signedPct(r.rel, 1)})`}>
              <span className="comp-rank">{r.b}</span>
              <span className="comp-n tabular">
                {r.unseen}
                <small>/{r.exp.toFixed(0)}</small>
              </span>
              <span className="comp-track">
                <span className="out-zero" />
                <span
                  className="comp-bar"
                  style={{ left: r.rel >= 0 ? '50%' : `${50 - w}%`, width: `${w}%`, background: diverging(r.rel >= 0 ? 0.85 : -0.85) }}
                />
              </span>
              <span className="comp-pct tabular">{signed(r.diff, 1)}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ───────────────────────── analytics ─────────────────────────

/** Full-shoe reference EVs (official RTPs from the in-game help). */
const BASELINE: Record<string, number> = {
  perfect_pairs: -0.041,
  twenty_one_plus_three: -0.037,
  crazy_7: -0.0523,
  bust_bonus: -0.0588,
  insurance: 3 * (128 / 416) - 1,
  main_edge_estimate: -0.0041,
}

function Analytics() {
  const a = useLab((s) => s.state!.shoe.analytics)
  const rows: { k: string; label: string; v: number }[] = [
    { k: 'main_edge_estimate', label: 'Main game', v: a.main_edge_estimate },
    { k: 'perfect_pairs', label: 'Perfect Pairs', v: a.perfect_pairs },
    { k: 'twenty_one_plus_three', label: '21+3', v: a.twenty_one_plus_three },
    { k: 'crazy_7', label: 'Crazy 7', v: a.crazy_7 },
    { k: 'bust_bonus', label: 'Bust Bonus', v: a.bust_bonus },
    { k: 'insurance', label: 'Insurance', v: a.insurance },
  ]
  return (
    <div className="section">
      <div className="section-head">
        <span className="section-title" title="Expected value per unit staked, given the current unseen composition">
          EV now
        </span>
      </div>
      <table className="kv tabular ev-kv">
        <tbody>
          {rows.map((r) => {
            const ok = typeof r.v === 'number' && isFinite(r.v)
            const base = BASELINE[r.k]
            return (
              <tr key={r.k} title={`Full shoe: ${signedPct(base, 2)}`}>
                <td>{r.label}</td>
                <td className={ok && r.v > 0 ? 'pos' : ''}>{ok ? signedPct(r.v, 2) : '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ───────────────────────── history ─────────────────────────

function CountHistory() {
  const shoe = useLab((s) => s.state!.shoe)
  const memo = useLab((s) => s.halfMemo)
  const pts: [number, number][] = shoe.history.map((h) => [h.round, h.tc])
  const markers: Marker[] = []
  const firstHalf2 = shoe.history.find((h) => h.half === 2)
  if (firstHalf2) markers.push({ x: firstHalf2.round, label: 'half 2' })
  else if (memo && memo.shoe === shoe.shoe_number) markers.push({ x: memo.switchRound, label: 'half 2' })
  return (
    <div className="section">
      <div className="section-head">
        <span className="section-title">True count (Hi-Lo)</span>
      </div>
      <LineChart
        series={[{ id: 'tc', label: 'True count', color: 'var(--gold)', points: pts }]}
        height={120}
        xLabel="round"
        yFormat={(y) => signed(y, Math.abs(y) < 10 && !Number.isInteger(y) ? 1 : 0)}
        tipYFormat={(y) => signed(y, 2)}
        refY={0}
        markers={markers}
        yMinSpan={4}
        emptyText="Count history appears after the first round of this shoe"
      />
    </div>
  )
}
