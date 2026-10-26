/** Diverging scale: red (−) ↔ neutral ↔ green (+). t in [-1, 1]. */
const NEG = [239, 107, 107] // --neg
const MID = [44, 44, 52] // neutral on dark surface
const POS = [60, 207, 127] // --pos

export function diverging(t: number): string {
  const x = Math.max(-1, Math.min(1, isFinite(t) ? t : 0))
  const a = x < 0 ? NEG : POS
  const k = Math.pow(Math.abs(x), 0.8)
  const c = MID.map((m, i) => Math.round(m + (a[i] - m) * k))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

export const CHIP_COLORS: Record<number, { base: string; edge: string; text: string }> = {
  1: { base: '#e9e6df', edge: '#9a958b', text: '#222' },
  2: { base: '#7cc4f0', edge: '#2d7fb8', text: '#0b2234' },
  5: { base: '#d8373b', edge: '#7c1418', text: '#fff' },
  10: { base: '#2f6fd6', edge: '#173d7d', text: '#fff' },
  25: { base: '#2e9a52', edge: '#155a2d', text: '#fff' },
  100: { base: '#2a2a2e', edge: '#9a9aa2', text: '#fff' },
  500: { base: '#8a3fd1', edge: '#4b1a7a', text: '#fff' },
  1000: { base: '#e3a92b', edge: '#8a5a07', text: '#2a1a00' },
}

export function chipColor(amount: number, chips: number[]) {
  const sorted = [...chips].sort((a, b) => a - b)
  let pick = sorted[0] ?? 1
  for (const c of sorted) if (c <= amount) pick = c
  return CHIP_COLORS[pick] ?? CHIP_COLORS[1]
}
