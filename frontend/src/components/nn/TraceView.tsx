import { useMemo } from 'react'
import { BET_LABELS, type Bets, type Trace, type TraceFeature } from '../../protocol'
import { money, signed, signedPct } from '../../lib/format'
import { diverging } from '../../lib/colors'

const prettyLabel = (l: string) => (BET_LABELS as Record<string, string>)[l] ?? l.replace(/_/g, ' ')

// ───────────────────────── header ─────────────────────────

export function TraceHeader({ trace }: { trace: Trace }) {
  const ref = trace.reference?.basic_strategy ?? null
  const dev = trace.reference?.hilo_deviation ?? null
  const agrees = ref !== null && ref === trace.chosen
  const kind = trace.kind === 'play' ? 'Play' : trace.kind === 'bet' ? 'Bet' : 'Insurance'
  return (
    <div className="trace-head">
      <span className="th-kind">{kind}</span>
      <span className="th-chosen">{trace.kind === 'bet' ? stakeText(trace) : prettyLabel(trace.chosen)}</span>
      {ref !== null && trace.kind !== 'bet' && (
        <span
          className={`th-ref ${agrees ? 'agree' : 'disagree'}`}
          title={`Basic strategy: ${prettyLabel(ref)}${dev ? ` · Hi-Lo index play: ${prettyLabel(dev)}` : ''}`}
        >
          {agrees ? '✓ basic' : `✗ basic: ${prettyLabel(ref)}`}
        </span>
      )}
    </div>
  )
}

// ───────────────────────── outputs ─────────────────────────

export function OutputBars({ trace }: { trace: Trace }) {
  const { labels, values, legal } = trace.outputs
  const isBet = trace.kind === 'bet'
  const finite = values.filter((v): v is number => v !== null && isFinite(v))
  const maxAbs = Math.max(1e-6, ...finite.map(Math.abs))
  return (
    <div className="outputs" title={trace.outputs.unit ?? undefined}>
      {labels.map((l, i) => {
        const v = values[i]
        const ok = legal[i] && v !== null
        const chosen = isBet ? false : l === trace.chosen
        const w = v === null ? 0 : (Math.abs(v) / maxAbs) * 50
        return (
          <div key={l} className={`out-row ${ok ? '' : 'illegal'} ${chosen ? 'chosen' : ''}`}>
            <span className="out-label">
              {prettyLabel(l)}
            </span>
            <span className="out-track">
              <span className="out-zero" />
              {v !== null && (
                <span
                  className={`out-bar ${v >= 0 ? 'p' : 'n'}`}
                  style={v >= 0 ? { left: '50%', width: `${w}%` } : { left: `${50 - w}%`, width: `${w}%` }}
                />
              )}
            </span>
            <span className={`out-val tabular ${v !== null && v > 0 ? 'pos' : ''}`}>
              {v === null ? (legal[i] ? '—' : 'illegal') : isBet ? signedPct(v, 2) : signed(v, 3)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

export function BetDetails({ trace }: { trace: Trace }) {
  const frac = trace.extra?.bet_fraction
  if (frac === undefined) return null
  return <div className="bet-details muted tabular">Stake = {(frac * 100).toFixed(2)}% of bankroll</div>
}

function stakeText(trace: Trace): string {
  const bets = trace.extra?.bets as Partial<Bets> | undefined
  if (!bets) return trace.chosen
  const entries = Object.entries(bets).filter(([, v]) => v)
  return entries.length ? entries.map(([k, v]) => `${prettyLabel(k)} ${money(v as number)}`).join(' + ') : 'no bet'
}

// ───────────────────────── features ─────────────────────────

type Group = 'hand' | 'dealer' | 'composition' | 'count'
const GROUP_ORDER: Group[] = ['hand', 'dealer', 'composition', 'count']
const GROUP_LABEL: Record<Group, string> = { hand: 'Hand', dealer: 'Dealer', composition: 'Composition', count: 'Count & shoe' }

function groupOf(name: string): Group {
  const n = name.toLowerCase()
  if (/dealer|^up_|upcard/.test(n)) return 'dealer'
  if (/^comp|unseen_|remain|composition/.test(n)) return 'composition'
  if (/(^|_)(tc|rc)(_|$)|count|deck|half|cut|shoe|pen|bankroll|insurance/.test(n)) return 'count'
  return 'hand'
}

interface Family {
  kind: 'family'
  prefix: string
  items: { suffix: string; f: TraceFeature }[]
  onehot: boolean
}
interface Single {
  kind: 'single'
  f: TraceFeature
}

function buildGroups(features: TraceFeature[]) {
  const byPrefix = new Map<string, { suffix: string; f: TraceFeature }[]>()
  for (const f of features) {
    const m = /^(.*)_([A-Za-z0-9]{1,2})$/.exec(f.name)
    if (m) {
      const list = byPrefix.get(m[1]) ?? []
      list.push({ suffix: m[2], f })
      byPrefix.set(m[1], list)
    }
  }
  const families = new Set([...byPrefix.entries()].filter(([, l]) => l.length >= 5).map(([p]) => p))
  const groups = new Map<Group, (Family | Single)[]>()
  const seen = new Set<string>()
  for (const f of features) {
    const m = /^(.*)_([A-Za-z0-9]{1,2})$/.exec(f.name)
    const g = groupOf(f.name)
    const list = groups.get(g) ?? []
    if (m && families.has(m[1])) {
      if (!seen.has(m[1])) {
        seen.add(m[1])
        const items = byPrefix.get(m[1])!
        list.push({ kind: 'family', prefix: m[1], items, onehot: items.every((x) => x.f.value === 0 || x.f.value === 1) })
      }
    } else list.push({ kind: 'single', f })
    groups.set(g, list)
  }
  return groups
}

export function FeatureList({ features }: { features: TraceFeature[] }) {
  const groups = useMemo(() => buildGroups(features), [features])
  return (
    <div className="features">
      {GROUP_ORDER.filter((g) => groups.get(g)?.length).map((g) => (
        <div key={g} className="feat-group">
          <div className="feat-group-title">{GROUP_LABEL[g]}</div>
          {groups.get(g)!.map((it) =>
            it.kind === 'single' ? (
              <FeatureRow key={it.f.name} f={it.f} />
            ) : (
              <FamilyRow key={it.prefix} fam={it} />
            ),
          )}
        </div>
      ))}
    </div>
  )
}

function fmtRaw(raw: TraceFeature['raw']) {
  if (raw === undefined || raw === null) return ''
  if (typeof raw === 'boolean') return raw ? 'yes' : 'no'
  return String(raw)
}

function FeatureRow({ f }: { f: TraceFeature }) {
  const v = Math.max(-1, Math.min(1, f.value))
  const w = Math.abs(v) * 50
  return (
    <div className="feat-row" title={`${f.name} = ${f.value}${f.raw !== undefined ? ` (raw ${fmtRaw(f.raw)})` : ''}`}>
      <span className="feat-name">{f.name}</span>
      <span className="feat-raw tabular">{fmtRaw(f.raw)}</span>
      <span className="feat-track">
        <span className="out-zero" />
        <span className="feat-bar" style={{ left: v >= 0 ? '50%' : `${50 - w}%`, width: `${w}%`, background: diverging(v) }} />
      </span>
      <span className="feat-val tabular">{f.value.toFixed(2)}</span>
    </div>
  )
}

function FamilyRow({ fam }: { fam: Family }) {
  if (fam.onehot) {
    const on = fam.items.find((x) => x.f.value === 1)
    return (
      <div className="feat-row fam">
        <span className="feat-name">{fam.prefix}</span>
        <span className="onehot">
          {fam.items.map((x) => (
            <span key={x.suffix} className={`oh ${x.f.value === 1 ? 'on' : ''}`} title={x.f.name}>
              {x.suffix}
            </span>
          ))}
        </span>
        <span className="feat-val tabular">{on ? on.suffix : '—'}</span>
      </div>
    )
  }
  const mx = Math.max(1e-6, ...fam.items.map((x) => Math.abs(x.f.value)))
  return (
    <div className="feat-row fam">
      <span className="feat-name">{fam.prefix}</span>
      <span className="vec">
        {fam.items.map((x) => {
          const t = x.f.value / mx
          return (
            <span key={x.suffix} className="vec-cell" title={`${x.f.name} = ${x.f.value.toFixed(3)}${x.f.raw !== undefined ? ` (raw ${fmtRaw(x.f.raw)})` : ''}`}>
              <span className="vec-bar-wrap">
                <span className="vec-bar" style={{ height: `${Math.abs(t) * 50}%`, [t >= 0 ? 'bottom' : 'top']: '50%', background: diverging(t) }} />
              </span>
              <span className="vec-lab">{x.suffix}</span>
            </span>
          )
        })}
      </span>
      <span className="feat-val tabular" />
    </div>
  )
}
