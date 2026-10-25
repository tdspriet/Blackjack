import { useEffect, useState } from 'react'
import { fetchCheckpoints, fetchTraining } from '../../api'
import type { CheckpointInfo, TrainingLogRow } from '../../protocol'
import { useLab } from '../../store'
import { compact, signedPct } from '../../lib/format'
import { LineChart, type Series } from '../charts/LineChart'

function series(rows: TrainingLogRow[], key: string, color: string, label: string): Series {
  return {
    id: key,
    label,
    color,
    points: rows.filter((r) => typeof r[key] === 'number' && isFinite(r[key] as number)).map((r) => [r.step, r[key] as number]),
  }
}

export function TrainingView() {
  const mode = useLab((s) => s.conn.mode)
  const [cks, setCks] = useState<CheckpointInfo[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [sel, setSel] = useState<string | null>(null)
  const [log, setLog] = useState<TrainingLogRow[] | null>(null)
  const [logErr, setLogErr] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    setCks(null)
    setErr(null)
    fetchCheckpoints()
      .then((c) => {
        if (!alive) return
        setCks(c)
        setSel((s) => s ?? c[0]?.name ?? null)
      })
      .catch((e: unknown) => alive && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [mode])

  useEffect(() => {
    if (!sel) return
    let alive = true
    setLog(null)
    setLogErr(null)
    fetchTraining(sel)
      .then((l) => alive && setLog(l))
      .catch((e: unknown) => alive && setLogErr(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [sel])

  const extraKeys = log
    ? [...new Set(log.flatMap((r) => Object.keys(r)))].filter((k) => !['step', 'loss', 'eval_ev'].includes(k) && log.some((r) => typeof r[k] === 'number'))
    : []

  return (
    <>
      <div className="section">
        <div className="section-head">
          <span className="section-title">Checkpoints</span>
        </div>
        {err && <div className="empty neg">Could not load checkpoints: {err}</div>}
        {!err && !cks && <div className="empty">Loading…</div>}
        {cks && cks.length === 0 && <div className="empty">No checkpoints yet — train one with onebj.ai.train.</div>}
        {cks && cks.length > 0 && (
          <div className="ck-list">
            {cks.map((c) => (
              <button key={c.name} className={`ck ${sel === c.name ? 'on' : ''}`} onClick={() => setSel(c.name)}>
                <div className="ck-name">{c.name}</div>
                <div className="ck-meta muted">{c.created ? new Date(c.created).toLocaleString() : ''}</div>
                {c.metrics && (
                  <div className="ck-metrics tabular">
                    {Object.entries(c.metrics)
                      .slice(0, 4)
                      .map(([k, v]) => (
                        <span key={k}>
                          {k.replace(/_/g, ' ')} <b>{typeof v === 'number' ? (Math.abs(v) < 1 ? v.toFixed(4) : v.toFixed(1)) : String(v)}</b>
                        </span>
                      ))}
                  </div>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
      {sel && (
        <>
          <div className="section">
            <div className="section-head">
              <span className="section-title">Loss · {sel}</span>
            </div>
            {logErr ? (
              <div className="empty neg">{logErr}</div>
            ) : (
              <LineChart series={log ? [series(log, 'loss', 'var(--accent)', 'loss')] : []} height={150} xLabel="step" xFormat={compact} yFormat={(y) => y.toFixed(3)} emptyText={log ? 'No loss values' : 'Loading…'} />
            )}
          </div>
          <div className="section">
            <div className="section-head">
              <span className="section-title">Eval EV per round</span>
            </div>
            <LineChart
              series={log ? [series(log, 'eval_ev', 'var(--pos)', 'eval EV')] : []}
              height={150}
              xLabel="step"
              xFormat={compact}
              yFormat={(y) => signedPct(y, 1)}
              tipYFormat={(y) => signedPct(y, 3)}
              refY={0}
              emptyText={log ? 'No eval values' : 'Loading…'}
            />
          </div>
          {extraKeys.map((k) => (
            <div className="section" key={k}>
              <div className="section-head">
                <span className="section-title">{k.replace(/_/g, ' ')}</span>
              </div>
              <LineChart series={[series(log!, k, 'var(--blue)', k)]} height={110} xLabel="step" xFormat={compact} yFormat={(y) => (Math.abs(y) < 10 ? y.toFixed(2) : compact(y))} />
            </div>
          ))}
        </>
      )}
    </>
  )
}
