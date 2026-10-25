import { useState } from 'react'
import { useLab } from '../../store'
import { NetworkGraph } from './NetworkGraph'
import { BetDetails, FeatureList, OutputBars, TraceHeader } from './TraceView'
import { MetricsView } from './MetricsView'
import { TrainingView } from './TrainingView'
import './nn.css'

type Tab = 'network' | 'metrics' | 'training'

export function NNPanel() {
  const [tab, setTab] = useState<Tab>('network')
  return (
    <aside className="side-panel">
      <AiControls />
      <div className="tabs">
        {(['network', 'metrics', 'training'] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>
      <div className="panel-scroll">
        {tab === 'network' && <NetworkTab />}
        {tab === 'metrics' && <MetricsView />}
        {tab === 'training' && <TrainingView />}
      </div>
    </aside>
  )
}

function AiControls() {
  const st = useLab((s) => s.state)!
  const sim = useLab((s) => s.sim)
  const delay = useLab((s) => s.autoplayDelay)
  const adviceOn = useLab((s) => s.adviceOn)
  const send = useLab((s) => s.send)
  const set = useLab((s) => s.set)
  const autoplay = st.ai.autoplay
  const simRunning = !!sim?.running
  const canStep = !autoplay && !simRunning && st.game.legal_actions.length > 0

  return (
    <div className="ai-controls">
      <button className="btn" disabled={!canStep} onClick={() => send({ cmd: 'ai_step' })} title="Let the AI make the next decision">
        AI step
      </button>
      <label className="toggle" title="AI plays continuously">
        <input type="checkbox" checked={autoplay} disabled={simRunning} onChange={(e) => send({ cmd: 'autoplay', on: e.target.checked, delay_ms: delay })} />
        <span className="track" />
        Autoplay
      </label>
      <input
        type="range"
        min={0}
        max={100}
        className="speed"
        value={Math.round(100 - (Math.log(delay / 30) / Math.log(2000 / 30)) * 100)}
        onChange={(e) => {
          const t = (100 - Number(e.target.value)) / 100
          const d = Math.round(30 * Math.pow(2000 / 30, t))
          set('autoplayDelay', d)
          if (autoplay) send({ cmd: 'autoplay', on: true, delay_ms: d })
        }}
        title={`Autoplay speed · ${delay} ms per step`}
        aria-label="Autoplay speed"
      />
      <label className="toggle" title="Mark the AI's recommendation while you play">
        <input
          type="checkbox"
          checked={adviceOn}
          onChange={(e) => {
            set('adviceOn', e.target.checked)
            send({ cmd: 'advice', on: e.target.checked })
          }}
        />
        <span className="track" />
        Advice
      </label>
    </div>
  )
}

function NetworkTab() {
  const net = useLab((s) => s.hello?.network ?? null)
  const ai = useLab((s) => s.state!.ai)
  const adviceOn = useLab((s) => s.adviceOn)
  const [featOpen, setFeatOpen] = useState(false)

  const useAdvice = adviceOn && !!ai.advice
  const trace = useAdvice ? ai.advice : ai.last_trace
  const isNN = ai.agent.startsWith('nn')

  return (
    <>
      <div className="section">
        <div className="section-head">
          <span className="section-title" title="Top-K strongest weights per layer pair. Node colour = activation (blue negative, orange positive); edge brightness = |weight × activation|.">
            Activations
          </span>
          <span className="muted small">{trace ? (useAdvice ? 'advice' : 'last decision') : ''}</span>
        </div>
        {net ? <NetworkGraph net={net} trace={isNN ? trace : null} /> : <div className="empty">No network description from the backend.</div>}
      </div>

      <div className="section">
        {trace ? (
          <>
            <TraceHeader trace={trace} />
            <OutputBars trace={trace} />
            {trace.kind === 'bet' && <BetDetails trace={trace} />}
          </>
        ) : (
          <div className="empty">No decision yet — press AI step or turn on Advice.</div>
        )}
      </div>

      {trace && (
        <div className="section">
          <button className={`collapser ${featOpen ? 'open' : ''}`} onClick={() => setFeatOpen((o) => !o)} aria-expanded={featOpen}>
            <span className="chev">▶</span>
            <span className="section-title">Input features</span>
            <span className="muted small" style={{ marginLeft: 'auto' }}>
              {trace.features.length}
            </span>
          </button>
          {featOpen && (
            <div style={{ marginTop: 12 }}>
              <FeatureList features={trace.features} />
            </div>
          )}
        </div>
      )}
    </>
  )
}
