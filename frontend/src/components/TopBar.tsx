import { useLab } from '../store'
import './TopBar.css'

export function TopBar() {
  const conn = useLab((s) => s.conn)
  const hello = useLab((s) => s.hello)
  const game = useLab((s) => s.state?.game)
  const ai = useLab((s) => s.state?.ai)
  const send = useLab((s) => s.send)
  const connect = useLab((s) => s.connect)

  const between = !game || game.phase === 'betting' || game.phase === 'settled'
  const crowdOn = hello?.config.crowd.enabled ?? true
  const disconnected = conn.mode === 'live' && conn.status !== 'open'

  const dotTitle =
    conn.mode === 'mock'
      ? `Mock mode — ${conn.note ?? 'in-browser mock server'}`
      : conn.mode === 'live'
        ? conn.status === 'open'
          ? 'Connected to the backend'
          : `Reconnecting… ${conn.note ?? ''}`
        : 'Connecting…'
  const dotCls = conn.mode === 'mock' ? 'mock' : conn.mode === 'live' && conn.status === 'open' ? 'live' : 'warn'

  return (
    <header className="topbar">
      <div className="brand">
        <span className={`conn-dot ${dotCls}`} title={dotTitle} />
        <span className="brand-name">ONE Blackjack Lab</span>
        {conn.mode === 'mock' && (
          <span className="mock-tag" title={dotTitle}>
            mock
          </span>
        )}
      </div>

      <div className="tb-spacer" />

      {disconnected && (
        <button className="btn small" onClick={() => void connect(false)}>
          Reconnect
        </button>
      )}

      {hello && ai && (
        <select className="select" value={ai.agent} onChange={(e) => send({ cmd: 'set_agent', agent: e.target.value })} title="Agent">
          {hello.agents.map((a) => (
            <option key={a} value={a}>
              {a}
            </option>
          ))}
        </select>
      )}

      <label className="toggle" title="Simulate the shared-hand crowd">
        <input
          type="checkbox"
          checked={crowdOn}
          disabled={!between}
          onChange={(e) => send({ cmd: 'set_config', patch: { crowd: { enabled: e.target.checked } } })}
        />
        <span className="track" />
        Crowd
      </label>

      <button className="btn ghost" disabled={!between} onClick={() => send({ cmd: 'new_shoe' })} title="Shuffle a fresh 8-deck shoe">
        New shoe
      </button>
      <button
        className="btn ghost"
        onClick={() => {
          if (confirm('Reset bankroll and session metrics?')) send({ cmd: 'reset', bankroll: hello?.config.bankroll.start })
        }}
      >
        Reset
      </button>
      <div className="tb-help" tabIndex={0}>
        ?
        <div className="tb-help-pop">
          <div className="tb-help-title">Keyboard</div>
          <div>
            <span className="kbd">H</span> hit · <span className="kbd">S</span> stand · <span className="kbd">D</span> double · <span className="kbd">P</span> split ·{' '}
            <span className="kbd">C</span> cash out
          </div>
          <div>
            <span className="kbd">I</span> insure · <span className="kbd">N</span> no insurance · <span className="kbd">Space</span> deal / rebet
          </div>
          <div className="tb-help-title">Betting</div>
          <div>Click a spot to add the selected chip; right-click or shift-click removes it.</div>
          <div className="tb-help-title">Table</div>
          <div>
            Everyone shares one hand. Dimmed cards were dealt to the crowd after you stood — not yours, but counted. The two containers are
            the halves of one 8-deck shuffle.
          </div>
          <div className="tb-help-title">Connection</div>
          <div className="tb-help-row">
            <span>{dotTitle}</span>
            {conn.mode === 'mock' ? (
              <button className="btn small" onClick={() => void connect(false)}>
                Connect live
              </button>
            ) : conn.mode === 'live' ? (
              <button className="btn small" onClick={() => void connect(true)}>
                Use mock
              </button>
            ) : null}
          </div>
        </div>
      </div>
    </header>
  )
}
