import { useEffect, useMemo, useRef, useState } from 'react'
import './charts.css'

export interface Series {
  id: string
  label: string
  color: string
  points: [number, number][]
  area?: boolean
}

export interface Marker {
  x: number
  label: string
}

interface Props {
  series: Series[]
  height?: number
  xFormat?: (x: number) => string
  yFormat?: (y: number) => string
  tipYFormat?: (y: number) => string
  xLabel?: string
  refY?: number | null
  markers?: Marker[]
  emptyText?: string
  yMinSpan?: number
}

function niceTicks(min: number, max: number, count = 4): number[] {
  if (!isFinite(min) || !isFinite(max)) return []
  if (min === max) {
    min -= 1
    max += 1
  }
  const span = max - min
  const raw = span / count
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const norm = raw / mag
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag
  const out: number[] = []
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  return out
}

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [w, setW] = useState(320)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(120, e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

/** Small dependency-free SVG line chart with crosshair tooltip (one y-axis, by design). */
export function LineChart({
  series,
  height = 150,
  xFormat = (x) => String(Math.round(x)),
  yFormat = (y) => String(Math.round(y * 100) / 100),
  tipYFormat,
  xLabel,
  refY = null,
  markers = [],
  emptyText = 'No data yet',
  yMinSpan = 0,
}: Props) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)
  const pad = { l: 42, r: 10, t: 8, b: 20 }
  const iw = Math.max(10, width - pad.l - pad.r)
  const ih = Math.max(10, height - pad.t - pad.b)

  const dom = useMemo(() => {
    let x0 = Infinity
    let x1 = -Infinity
    let y0 = Infinity
    let y1 = -Infinity
    for (const s of series)
      for (const [x, y] of s.points) {
        if (y === null || !isFinite(y)) continue
        x0 = Math.min(x0, x)
        x1 = Math.max(x1, x)
        y0 = Math.min(y0, y)
        y1 = Math.max(y1, y)
      }
    if (refY !== null && isFinite(y0)) {
      y0 = Math.min(y0, refY)
      y1 = Math.max(y1, refY)
    }
    if (y1 - y0 < yMinSpan) {
      const mid = (y0 + y1) / 2
      y0 = mid - yMinSpan / 2
      y1 = mid + yMinSpan / 2
    }
    const padY = (y1 - y0) * 0.08 || 1
    return { x0, x1: x1 === x0 ? x0 + 1 : x1, y0: y0 - padY, y1: y1 + padY }
  }, [series, refY, yMinSpan])

  const has = series.some((s) => s.points.length > 0) && isFinite(dom.x0)
  const sx = (x: number) => pad.l + ((x - dom.x0) / (dom.x1 - dom.x0)) * iw
  const sy = (y: number) => pad.t + (1 - (y - dom.y0) / (dom.y1 - dom.y0)) * ih
  const yTicks = has ? niceTicks(dom.y0, dom.y1, 4) : []
  const xTicks = has ? niceTicks(dom.x0, dom.x1, Math.max(2, Math.floor(iw / 80))) : []

  const base = series[0]?.points ?? []
  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    if (!base.length) return
    const r = (e.target as SVGRectElement).getBoundingClientRect()
    const x = dom.x0 + ((e.clientX - r.left) / r.width) * (dom.x1 - dom.x0)
    let lo = 0
    let hi = base.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (base[mid][0] < x) lo = mid + 1
      else hi = mid
    }
    if (lo > 0 && Math.abs(base[lo - 1][0] - x) < Math.abs(base[lo][0] - x)) lo--
    setHover(lo)
  }

  const hx = hover !== null && base[hover] ? base[hover][0] : null

  return (
    <div className="lc" ref={ref} style={{ height }}>
      {!has ? (
        <div className="lc-empty">{emptyText}</div>
      ) : (
        <svg width={width} height={height}>
          {yTicks.map((t) => (
            <g key={`y${t}`}>
              <line x1={pad.l} x2={pad.l + iw} y1={sy(t)} y2={sy(t)} className="lc-grid" />
              <text x={pad.l - 6} y={sy(t)} className="lc-tick" textAnchor="end" dominantBaseline="middle">
                {yFormat(t)}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={`x${t}`} x={sx(t)} y={height - 5} className="lc-tick" textAnchor="middle">
              {xFormat(t)}
            </text>
          ))}
          {refY !== null && <line x1={pad.l} x2={pad.l + iw} y1={sy(refY)} y2={sy(refY)} className="lc-ref" />}
          {markers
            .filter((m) => m.x >= dom.x0 && m.x <= dom.x1)
            .map((m) => (
              <g key={`m${m.x}${m.label}`}>
                <line x1={sx(m.x)} x2={sx(m.x)} y1={pad.t} y2={pad.t + ih} className="lc-marker" />
                <text x={sx(m.x) + 4} y={pad.t + 9} className="lc-marker-label">
                  {m.label}
                </text>
              </g>
            ))}
          {series.map((s) => {
            if (!s.points.length) return null
            const d = s.points.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join('')
            const last = s.points[s.points.length - 1]
            return (
              <g key={s.id}>
                {s.area && (
                  <path
                    d={`${d}L${sx(last[0]).toFixed(1)},${sy(refY ?? dom.y0).toFixed(1)}L${sx(s.points[0][0]).toFixed(1)},${sy(refY ?? dom.y0).toFixed(1)}Z`}
                    fill={s.color}
                    opacity={0.1}
                  />
                )}
                <path d={d} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
                <circle cx={sx(last[0])} cy={sy(last[1])} r={4} fill={s.color} stroke="var(--panel-2)" strokeWidth={2} />
              </g>
            )
          })}
          {hx !== null && (
            <g pointerEvents="none">
              <line x1={sx(hx)} x2={sx(hx)} y1={pad.t} y2={pad.t + ih} className="lc-cross" />
              {series.map((s) => {
                const p = s.points.find((q) => q[0] === hx)
                return p ? <circle key={s.id} cx={sx(p[0])} cy={sy(p[1])} r={4} fill={s.color} stroke="var(--panel-2)" strokeWidth={2} /> : null
              })}
            </g>
          )}
          <rect
            x={pad.l}
            y={pad.t}
            width={iw}
            height={ih}
            fill="transparent"
            onMouseMove={onMove}
            onMouseLeave={() => setHover(null)}
          />
        </svg>
      )}
      {has && hx !== null && (
        <div className="viz-tip" style={{ left: Math.min(width - 150, Math.max(0, sx(hx) + 10)), top: 4 }}>
          <div className="t">
            {xLabel ? `${xLabel} ` : ''}
            {xFormat(hx)}
          </div>
          {series.map((s) => {
            const p = s.points.find((q) => q[0] === hx)
            return p ? (
              <div key={s.id} className="lc-tip-row">
                <i style={{ background: s.color }} />
                {series.length > 1 && <span>{s.label}</span>}
                <b>{(tipYFormat ?? yFormat)(p[1])}</b>
              </div>
            ) : null
          })}
        </div>
      )}
    </div>
  )
}
