import { useEffect, useMemo, useRef, useState } from 'react'
import type { NetworkDescription, Trace } from '../../protocol'
import { diverging } from '../../lib/colors'

interface NodePos {
  x: number
  y: number
  s: number // square size
}
interface LayerGeom {
  name: string
  size: number
  labels?: string[]
  nodes: NodePos[]
  head: boolean
  cx: number
}

const H = 260
const TOP = 14
const BOTTOM = 30

function useWidth() {
  const ref = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(360)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

function layout(net: NetworkDescription, width: number): LayerGeom[] {
  const trunk = net.layers
  const heads = net.heads ?? []
  const nCols = trunk.length + (heads.length ? 1 : 0)
  const left = 22
  const right = width - (heads.length ? 92 : 22)
  const step = nCols > 1 ? (right - left) / (nCols - 1) : 0
  const avail = H - TOP - BOTTOM
  const geoms: LayerGeom[] = trunk.map((l, li) => {
    const cx = left + li * step
    const n = Math.max(1, l.size)
    const maxRows = Math.max(1, Math.floor(avail / 6.5))
    const cols = Math.ceil(n / maxRows)
    const rows = Math.ceil(n / cols)
    const cell = Math.min(11, avail / rows, 40 / Math.max(1, cols - 0.5) + 2)
    const gridH = rows * cell
    const y0 = TOP + (avail - gridH) / 2 + cell / 2
    const nodes: NodePos[] = []
    for (let i = 0; i < n; i++) {
      const c = Math.floor(i / rows)
      const r = i % rows
      nodes.push({ x: cx + (c - (cols - 1) / 2) * cell, y: y0 + r * cell, s: Math.max(2.5, cell - 1.6) })
    }
    return { name: l.name, size: l.size, labels: l.labels, nodes, head: false, cx }
  })
  if (heads.length) {
    const cx = left + (nCols - 1) * step
    const gap = 20
    const sp = 17
    const total = heads.reduce((a, h) => a + h.size * sp, 0) + gap * (heads.length - 1)
    let y = TOP + Math.max(0, (avail - total) / 2) + sp / 2
    for (const h of heads) {
      const nodes: NodePos[] = []
      for (let i = 0; i < h.size; i++) {
        nodes.push({ x: cx, y, s: 11 })
        y += sp
      }
      y += gap
      geoms.push({ name: h.name, size: h.size, labels: h.labels, nodes, head: true, cx })
    }
  }
  return geoms
}

export function NetworkGraph({ net, trace }: { net: NetworkDescription; trace: Trace | null }) {
  const [ref, width] = useWidth()
  const [hover, setHover] = useState<{ layer: string; i: number; x: number; y: number } | null>(null)
  const geoms = useMemo(() => layout(net, width), [net, width])
  const byName = useMemo(() => new Map(geoms.map((g) => [g.name, g])), [geoms])

  const acts = useMemo(() => {
    const m = new Map<string, { a: number[]; scale: number }>()
    for (const l of trace?.layers ?? []) {
      let mx = 0
      for (const v of l.activations) mx = Math.max(mx, Math.abs(v))
      m.set(l.name, { a: l.activations, scale: mx || 1 })
    }
    // heads: fall back to trace outputs when the trace has no explicit head layer
    if (trace) {
      const head = geoms.find((g) => g.head && g.labels && trace.outputs.labels.every((lab) => g.labels!.includes(lab)))
      if (head && !m.has(head.name)) {
        const a = head.labels!.map((lab) => {
          const k = trace.outputs.labels.indexOf(lab)
          return k >= 0 ? (trace.outputs.values[k] ?? 0) : 0
        })
        m.set(head.name, { a, scale: Math.max(1e-6, ...a.map(Math.abs)) })
      }
    }
    return m
  }, [trace, geoms])

  const activeHead = useMemo(() => {
    if (!trace) return null
    for (const g of geoms) if (g.head && acts.has(g.name)) return g.name
    return null
  }, [trace, geoms, acts])

  const edges = useMemo(() => {
    const out: { x1: number; y1: number; x2: number; y2: number; w: number; flow: number; key: string; head?: string }[] = []
    for (const [key, list] of Object.entries(net.edges ?? {})) {
      const [a, b] = key.split('->')
      const ga = byName.get(a)
      const gb = byName.get(b)
      if (!ga || !gb || !list.length) continue
      const maxW = Math.max(...list.map((e) => Math.abs(e[2]))) || 1
      const actA = acts.get(a)
      let flows = list.map(([i, , w]) => (actA ? Math.abs(w * (actA.a[i] ?? 0)) : 0))
      const maxF = Math.max(...flows) || 1
      flows = flows.map((f) => f / maxF)
      list.forEach(([i, j, w], k) => {
        const pa = ga.nodes[i]
        const pb = gb.nodes[j]
        if (!pa || !pb) return
        out.push({ x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, w: w / maxW, flow: actA ? flows[k] : -1, key: `${key}:${k}`, head: gb.head ? gb.name : undefined })
      })
    }
    return out
  }, [net, byName, acts])

  const chosenIdx = (g: LayerGeom) => (trace && g.labels ? g.labels.indexOf(trace.chosen) : -1)

  const tip = (() => {
    if (!hover) return null
    const g = byName.get(hover.layer)
    if (!g) return null
    const a = acts.get(g.name)?.a[hover.i]
    const label = g.labels?.[hover.i]
    const feat = g.name === net.layers[0]?.name && trace ? trace.features[hover.i] : undefined
    return (
      <div className="viz-tip" style={{ left: Math.min(width - 170, hover.x + 10), top: Math.max(0, hover.y - 10) }}>
        <div className="t">
          {g.name} · #{hover.i}
        </div>
        {label && <div>{label}</div>}
        <div>
          activation <b>{a === undefined ? '—' : a.toFixed(3)}</b>
        </div>
        {feat?.raw !== undefined && feat.raw !== null && <div className="muted">raw {String(feat.raw)}</div>}
      </div>
    )
  })()

  return (
    <div className="netgraph" ref={ref}>
      <svg width={width} height={H} role="img" aria-label="Network topology with live activations">
        <g className="ng-edges">
          {edges.map((e) => {
            const pos = e.w > 0
            const dimHead = e.head && activeHead && e.head !== activeHead
            const op = dimHead ? 0.02 : e.flow < 0 ? 0.05 + 0.16 * Math.abs(e.w) : 0.03 + 0.55 * e.flow * Math.abs(e.w)
            return (
              <line
                key={e.key}
                x1={e.x1}
                y1={e.y1}
                x2={e.x2}
                y2={e.y2}
                stroke={pos ? 'var(--pos)' : 'var(--neg)'}
                strokeWidth={e.flow > 0.6 ? 1.2 : 0.7}
                strokeOpacity={op}
              />
            )
          })}
        </g>
        {geoms.map((g) => {
          const a = acts.get(g.name)
          const ci = chosenIdx(g)
          const dim = g.head && activeHead !== null && g.name !== activeHead
          return (
            <g key={g.name} opacity={dim ? 0.35 : 1}>
              {g.nodes.map((p, i) => {
                const v = a ? (a.a[i] ?? 0) / a.scale : 0
                const fill = a ? diverging(v) : '#2c2c34'
                const on = hover?.layer === g.name && hover.i === i
                return g.head ? (
                  <g key={i}>
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={6}
                      fill={fill}
                      stroke={i === ci && !dim ? 'var(--gold)' : 'rgba(255,255,255,0.18)'}
                      strokeWidth={i === ci && !dim ? 2 : 1}
                      className="ng-node"
                      onMouseEnter={() => setHover({ layer: g.name, i, x: p.x, y: p.y })}
                      onMouseLeave={() => setHover(null)}
                    />
                    <text x={p.x + 11} y={p.y} dominantBaseline="middle" className={`ng-head-label ${i === ci && !dim ? 'chosen' : ''}`}>
                      {(g.labels?.[i] ?? `#${i}`).replace(/_/g, ' ').replace('twenty one plus three', '21+3')}
                    </text>
                  </g>
                ) : (
                  <rect
                    key={i}
                    x={p.x - p.s / 2}
                    y={p.y - p.s / 2}
                    width={p.s}
                    height={p.s}
                    rx={Math.min(2, p.s / 3)}
                    fill={fill}
                    stroke={on ? '#fff' : 'none'}
                    className="ng-node"
                    onMouseEnter={() => setHover({ layer: g.name, i, x: p.x, y: p.y })}
                    onMouseLeave={() => setHover(null)}
                  />
                )
              })}
              {!g.head && (
                <text x={g.cx} y={H - 12} textAnchor="middle" className="ng-caption">
                  {g.name} <tspan className="ng-size">{g.size}</tspan>
                </text>
              )}
            </g>
          )
        })}
        {geoms.some((g) => g.head) && (
          <text x={geoms.find((g) => g.head)!.cx} y={H - 12} textAnchor="middle" className="ng-caption">
            heads
          </text>
        )}
      </svg>
      {tip}
    </div>
  )
}
