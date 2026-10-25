import { useLab } from '../../store'
import { BET_LABELS, type Metrics } from '../../protocol'
import { money, num, pct, signedMoney, signedPct, compact, moneyCompact } from '../../lib/format'
import { LineChart } from '../charts/LineChart'

const SIM_SIZES = [1000, 10000, 100000]

export function MetricsView() {
  const session = useLab((s) => s.state!.metrics)
  const sim = useLab((s) => s.sim)
  const source = useLab((s) => s.metricsSource)
  const set = useLab((s) => s.set)
  const useSim = source === 'sim' && !!sim
  const m: Metrics = useSim ? sim!.metrics : session
  return (
    <>
      <SimControls />
      <div className="section">
        <div className="section-head">
          <div className="seg">
            <button className={!useSim ? 'on' : ''} onClick={() => set('metricsSource', 'session')}>
              Session
            </button>
            <button className={useSim ? 'on' : ''} disabled={!sim} onClick={() => set('metricsSource', 'sim')}>
              Simulation
            </button>
          </div>
          <span className="muted small tabular">{num(m.rounds)} rounds</span>
        </div>
        <LineChart
          series={[{ id: 'bankroll', label: 'Bankroll', color: 'var(--accent)', points: m.bankroll, area: true }]}
          height={160}
          xLabel="round"
          xFormat={(x) => compact(x)}
          yFormat={moneyCompact}
          tipYFormat={(y) => money(y)}
          refY={m.bankroll[0]?.[1] ?? null}
          emptyText="Play or simulate some rounds"
        />
        <KeyValues m={m} />
      </div>
      <div className="section">
        <div className="section-head">
          <span className="section-title">Side bets</span>
        </div>
        <SideBetTable m={m} />
      </div>
    </>
  )
}

function SimControls() {
  const st = useLab((s) => s.state)!
  const sim = useLab((s) => s.sim)
  const simRounds = useLab((s) => s.simRounds)
  const send = useLab((s) => s.send)
  const set = useLab((s) => s.set)
  const running = !!sim?.running
  const eta = sim && running && sim.metrics.rounds_per_sec ? Math.max(0, (sim.total - sim.done) / sim.metrics.rounds_per_sec) : null
  return (
    <div className="section">
      <div className="sim-row">
        <span className="section-title">Simulate</span>
        <div className="seg">
          {SIM_SIZES.map((n) => (
            <button key={n} className={simRounds === n ? 'on' : ''} onClick={() => set('simRounds', n)} disabled={running}>
              {n / 1000}k
            </button>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        {running ? (
          <button
            className="btn"
            onClick={() => {
              send({ cmd: 'stop' })
              if (sim) set('sim', { ...sim, running: false })
            }}
          >
            Stop
          </button>
        ) : (
          <button
            className="btn"
            disabled={st.ai.autoplay}
            onClick={() => {
              set('sim', { done: 0, total: simRounds, metrics: st.metrics, running: true, agent: st.ai.agent, startedAt: Date.now() })
              set('metricsSource', 'sim')
              send({ cmd: 'simulate', rounds: simRounds, agent: st.ai.agent })
            }}
            title={`Headless simulation with ${st.ai.agent}`}
          >
            Run
          </button>
        )}
      </div>
      {sim && (
        <div className="sim-progress">
          <div className="progress">
            <div style={{ width: `${(sim.done / Math.max(1, sim.total)) * 100}%` }} />
          </div>
          <div className="sim-meta tabular">
            <span>
              {num(sim.done)} / {num(sim.total)} · {sim.agent}
            </span>
            <span>
              {sim.metrics.rounds_per_sec ? `${num(sim.metrics.rounds_per_sec)}/s` : ''}
              {eta !== null ? ` · ${eta < 1 ? '<1' : Math.round(eta)}s` : ''}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

function KeyValues({ m }: { m: Metrics }) {
  const n = Math.max(1, m.wins + m.losses + m.pushes)
  const tone = (x: number) => (x > 0 ? 'pos' : x < 0 ? 'neg' : '')
  const rows: [string, React.ReactNode, string?][] = [
    ['Net', signedMoney(m.net), tone(m.net)],
    ['Wagered', money(m.wagered)],
    ['EV per round', signedMoney(m.ev_per_round), tone(m.ev_per_round)],
    ['Edge', signedPct(m.edge, 2), tone(m.edge)],
    ['Win / loss / push', `${pct(m.wins / n, 1)} / ${pct(m.losses / n, 1)} / ${pct(m.pushes / n, 1)}`],
    ['Blackjacks', num(m.blackjacks)],
    ['Six-card Charlies', num(m.charlies)],
    ['Max drawdown', money(m.max_drawdown)],
    ['Agreement with basic', m.agreement_basic === null ? '—' : pct(m.agreement_basic, 1)],
  ]
  if (m.rounds_per_sec) rows.push(['Rounds per second', num(m.rounds_per_sec)])
  return (
    <table className="kv tabular">
      <tbody>
        {rows.map(([k, v, cls]) => (
          <tr key={k}>
            <td>{k}</td>
            <td className={cls}>{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function SideBetTable({ m }: { m: Metrics }) {
  const keys = ['perfect_pairs', 'twenty_one_plus_three', 'crazy_7', 'bust_bonus', 'insurance'] as const
  const rows = keys.map((k) => ({ k, v: m.side_bets[k] })).filter((r) => r.v && r.v.wagered > 0)
  if (!rows.length) return <div className="muted small">No side bets placed yet.</div>
  return (
    <table className="data">
      <thead>
        <tr>
          <th>Bet</th>
          <th>Wagered</th>
          <th>Net</th>
          <th>RTP</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ k, v }) => (
          <tr key={k}>
            <td>{BET_LABELS[k]}</td>
            <td>{money(v!.wagered)}</td>
            <td className={v!.net > 0 ? 'pos' : v!.net < 0 ? 'neg' : ''}>{signedMoney(v!.net)}</td>
            <td>{pct((v!.wagered + v!.net) / v!.wagered, 1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
