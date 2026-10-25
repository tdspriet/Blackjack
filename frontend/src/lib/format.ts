const eur = new Intl.NumberFormat('en-IE', { style: 'currency', currency: 'EUR', minimumFractionDigits: 2, maximumFractionDigits: 2 })
const eur0 = new Intl.NumberFormat('en-IE', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 })
const int = new Intl.NumberFormat('en-US')

export const money = (x: number | null | undefined) => (x === null || x === undefined || !isFinite(x) ? '—' : eur.format(x))
export const money0 = (x: number) => eur0.format(x)
export const signedMoney = (x: number | null | undefined) => {
  if (x === null || x === undefined || !isFinite(x)) return '—'
  if (Math.abs(x) < 0.005) return eur.format(0)
  return (x > 0 ? '+' : '−') + eur.format(Math.abs(x))
}
export const num = (x: number) => int.format(x)
export const compact = (x: number) => {
  const a = Math.abs(x)
  if (a >= 1e6) return (x / 1e6).toFixed(a >= 1e7 ? 0 : 1) + 'M'
  if (a >= 1e3) return (x / 1e3).toFixed(a >= 1e4 ? 0 : 1) + 'K'
  return String(Math.round(x * 100) / 100)
}
/** "−€1.2K" style compact money for axes / tiles. */
export const moneyCompact = (x: number) => {
  if (!isFinite(x)) return '—'
  const a = Math.abs(x)
  const body = a >= 1000 ? compact(a) : a.toFixed(a >= 100 || Number.isInteger(a) ? 0 : 2)
  return (x < 0 ? '−€' : '€') + body
}
export const signedMoneyCompact = (x: number) => {
  if (!isFinite(x)) return '—'
  if (Math.abs(x) < 1000) return signedMoney(x)
  return (x > 0 ? '+' : '−') + '€' + compact(Math.abs(x))
}
export const pct = (x: number | null | undefined, digits = 1) =>
  x === null || x === undefined || !isFinite(x) ? '—' : (x * 100).toFixed(digits) + '%'
export const signedPct = (x: number | null | undefined, digits = 2) => {
  if (x === null || x === undefined || !isFinite(x)) return '—'
  const v = x * 100
  return (v > 0 ? '+' : v < 0 ? '−' : '±') + Math.abs(v).toFixed(digits) + '%'
}
export const signed = (x: number | null | undefined, digits = 2) => {
  if (x === null || x === undefined || !isFinite(x)) return '—'
  return (x > 0 ? '+' : x < 0 ? '−' : '±') + Math.abs(x).toFixed(digits)
}
export const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))
export const titleCase = (s: string) => s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
