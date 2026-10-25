import type { Bets, NetworkDescription, PlayAction, ShoeAnalytics, Trace, TraceFeature } from '../protocol'
import { BUCKETS, handTotal, valueOf, bucketOf } from './cards'
import type { MockTable } from './engine'
import { actionEvs, basicAction, upValue } from './strategy'
import { bucketProbs } from './analytics'
import { Rng } from './rng'

/**
 * A fake-but-consistent "neural network": a real MLP forward pass with fixed random
 * weights (48 → 128 → 64 → heads) whose activations therefore respond to the inputs,
 * while the output heads are driven by composition-dependent EVs so decisions look sane.
 */

const PLAY_LABELS: PlayAction[] = ['stand', 'hit', 'double', 'split', 'cashout']
const BET_LABELS = ['main', 'perfect_pairs', 'twenty_one_plus_three', 'crazy_7', 'bust_bonus', 'insurance'] as const

export const FEATURE_NAMES: string[] = [
  'player_total',
  'soft',
  'num_cards',
  'can_double',
  'can_split',
  'is_split',
  'split_aces',
  ...BUCKETS.map((b) => `pair_${b}`),
  ...BUCKETS.map((b) => `dealer_up_${b}`),
  ...BUCKETS.map((b) => `comp_${b}`),
  'decks_unseen',
  'tc_hilo',
  'tc_omega2',
  'tc_wong_halves',
  'tc_zen',
  'rc_ko',
  'half',
  'cards_to_cut',
  'shoe_progress',
  'insurance_offered',
  'bankroll',
]

const SIZES = { input: FEATURE_NAMES.length, h1: 128, h2: 64 }

function matrix(rng: Rng, rows: number, cols: number, scale: number): number[][] {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => rng.normal() * scale))
}

function topK(w: number[][], k: number): [number, number, number][] {
  // w[j][i]: weight from i (source) to j (target)
  const all: [number, number, number][] = []
  w.forEach((row, j) => row.forEach((v, i) => all.push([i, j, Math.round(v * 1000) / 1000])))
  all.sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]))
  return all.slice(0, k)
}

export class MockNet {
  W1: number[][]
  b1: number[]
  W2: number[][]
  b2: number[]
  Wp: number[][]
  Wb: number[][]
  constructor(seed = 99) {
    const rng = new Rng(seed)
    this.W1 = matrix(rng, SIZES.h1, SIZES.input, 1.6 / Math.sqrt(SIZES.input))
    this.b1 = Array.from({ length: SIZES.h1 }, () => rng.normal() * 0.15)
    this.W2 = matrix(rng, SIZES.h2, SIZES.h1, 1.8 / Math.sqrt(SIZES.h1))
    this.b2 = Array.from({ length: SIZES.h2 }, () => rng.normal() * 0.1)
    this.Wp = matrix(rng, 5, SIZES.h2, 1 / Math.sqrt(SIZES.h2))
    this.Wb = matrix(rng, 6, SIZES.h2, 1 / Math.sqrt(SIZES.h2))
  }

  describe(name: string): NetworkDescription {
    return {
      name,
      checkpoint: 'checkpoints/v1-mock.pt',
      layers: [
        { name: 'input', size: SIZES.input, labels: FEATURE_NAMES },
        { name: 'trunk.0', size: SIZES.h1 },
        { name: 'trunk.1', size: SIZES.h2 },
      ],
      heads: [
        { name: 'head.play', size: 5, labels: [...PLAY_LABELS], from: 'trunk.1' },
        { name: 'head.bet', size: 6, labels: [...BET_LABELS], from: 'trunk.1' },
      ],
      edges: {
        'input->trunk.0': topK(this.W1, 260),
        'trunk.0->trunk.1': topK(this.W2, 260),
        'trunk.1->head.play': topK(this.Wp, 48),
        'trunk.1->head.bet': topK(this.Wb, 60),
      },
      training: { algorithm: 'all-action MC rollouts (mock)', steps: 200_000, created: '2026-10-25' },
    }
  }

  forward(x: number[]) {
    const h1 = this.W1.map((row, j) => Math.max(0, row.reduce((a, w, i) => a + w * x[i], this.b1[j])))
    const h2 = this.W2.map((row, j) => Math.max(0, row.reduce((a, w, i) => a + w * h1[i], this.b2[j])))
    const play = this.Wp.map((row) => Math.tanh(row.reduce((a, w, i) => a + w * h2[i], 0)))
    const bet = this.Wb.map((row) => Math.tanh(row.reduce((a, w, i) => a + w * h2[i], 0)))
    return { h1, h2, play, bet }
  }
}

// ───────────────────────── features ─────────────────────────

export function features(t: MockTable, insuranceOffered = false): TraceFeature[] {
  const h = t.hands[t.active] ?? t.hands[0]
  const cards = h?.cards ?? []
  const tot = handTotal(cards)
  const up = t.dealer[0]
  const legal = t.legalActions()
  const pairBucket = cards.length === 2 && valueOf(cards[0]) === valueOf(cards[1]) ? bucketOf(cards[0]) : -1
  const unseenB = t.unseenBuckets()
  const N = t.cardsUnseen
  const full = t.cfg.shoe.num_decks * 52
  const out: TraceFeature[] = []
  const push = (name: string, value: number, raw?: number | string | boolean | null) =>
    out.push({ name, value: Math.round(value * 1000) / 1000, raw })
  push('player_total', cards.length ? tot.total / 21 : 0, cards.length ? tot.total : null)
  push('soft', tot.soft ? 1 : 0, tot.soft)
  push('num_cards', cards.length / 6, cards.length)
  push('can_double', legal.includes('double') ? 1 : 0, legal.includes('double'))
  push('can_split', legal.includes('split') ? 1 : 0, legal.includes('split'))
  push('is_split', t.hands.length > 1 ? 1 : 0, t.hands.length > 1)
  push('split_aces', h?.splitAces ? 1 : 0, !!h?.splitAces)
  BUCKETS.forEach((b, i) => push(`pair_${b}`, pairBucket === i ? 1 : 0))
  BUCKETS.forEach((b, i) => push(`dealer_up_${b}`, up !== undefined && bucketOf(up) === i ? 1 : 0))
  BUCKETS.forEach((b, i) => {
    const fullB = (i === 9 ? 16 : 4) * t.cfg.shoe.num_decks
    const v = N ? (unseenB[i] / N - fullB / full) * 10 : 0
    push(`comp_${b}`, v, unseenB[i])
  })
  push('decks_unseen', N / full, Math.round((N / 52) * 100) / 100)
  push('tc_hilo', t.trueCount('hilo') / 5, Math.round(t.trueCount('hilo') * 100) / 100)
  push('tc_omega2', t.trueCount('omega2') / 8, Math.round(t.trueCount('omega2') * 100) / 100)
  push('tc_wong_halves', t.trueCount('wong_halves') / 5, Math.round(t.trueCount('wong_halves') * 100) / 100)
  push('tc_zen', t.trueCount('zen') / 8, Math.round(t.trueCount('zen') * 100) / 100)
  push('rc_ko', t.running.ko / 30, t.running.ko)
  push('half', t.half - 1, t.half)
  push('cards_to_cut', Math.max(0, t.cutAt - t.pos) / t.halfSize, Math.max(0, t.cutAt - t.pos))
  push('shoe_progress', 1 - N / full, Math.round((1 - N / full) * 1000) / 10 + '%')
  push('insurance_offered', insuranceOffered ? 1 : 0, insuranceOffered)
  push('bankroll', Math.tanh(t.balance / t.cfg.bankroll.start - 1), Math.round(t.balance))
  return out
}

// ───────────────────────── agents ─────────────────────────

export interface MockAgent {
  name: string
  decideBets(t: MockTable, a: ShoeAnalytics | null, withTrace: boolean): { bets: Bets; trace: Trace | null }
  decideInsurance(t: MockTable, withTrace: boolean): { insure: boolean; trace: Trace | null }
  decideAction(t: MockTable, withTrace: boolean): { action: PlayAction; trace: Trace | null }
}

const BASE_UNIT = 10

function clampBets(t: MockTable, b: Bets): Bets {
  const L = t.cfg.limits
  b.main = Math.max(L.main_min, Math.min(L.main_max, b.main))
  const sides = ['perfect_pairs', 'twenty_one_plus_three', 'crazy_7', 'bust_bonus'] as const
  let total = b.main + sides.reduce((a, k) => a + b[k], 0)
  if (total > t.balance) {
    for (const k of sides) b[k] = 0
    b.main = Math.max(L.main_min, Math.min(b.main, Math.floor(t.balance)))
    total = b.main
  }
  return b
}

function playContext(t: MockTable) {
  const h = t.hands[t.active]
  const legal = t.legalActions()
  const p = bucketProbs(t.unseen)
  const evs = actionEvs(h.cards, bucketOf(t.dealer[0]), p, legal.includes('double'), legal.includes('split'))
  const basic = basicAction(h.cards, t.dealer[0], legal.includes('double'), legal.includes('split'))
  const offer = t.cashoutOffer(h)
  // cash-out value is known exactly: offer / bet − 1 (per unit of the hand's bet)
  const cashout = offer !== null && legal.includes('cashout') ? offer / h.bet - 1 : null
  return { h, legal, evs, basic, cashout }
}

function hiloDeviation(t: MockTable, cards: number[], up: number, legal: string[]): PlayAction | null {
  const tc = t.trueCount('hilo')
  const { total, soft } = handTotal(cards)
  const d = upValue(up)
  const pairT = cards.length === 2 && valueOf(cards[0]) === 10 && valueOf(cards[1]) === 10
  if (pairT && legal.includes('split') && ((d === 5 && tc >= 5) || (d === 6 && tc >= 4))) return 'split'
  if (soft) return null
  if (total === 16 && d === 10 && tc >= 0) return 'stand'
  if (total === 15 && d === 10 && tc >= 4) return 'stand'
  if (total === 12 && d === 3 && tc >= 2) return 'stand'
  if (total === 12 && d === 2 && tc >= 3) return 'stand'
  if (total === 13 && d === 2 && tc < -1) return 'hit'
  if (total === 12 && d === 4 && tc < 0) return 'hit'
  if (legal.includes('double')) {
    if (total === 10 && d === 10 && tc >= 4) return 'double'
    if (total === 11 && d === 11 && tc >= 1) return 'double'
    if (total === 9 && d === 2 && tc >= 1) return 'double'
    if (total === 9 && d === 7 && tc >= 3) return 'double'
  }
  return null
}

function playTrace(
  t: MockTable,
  values: (number | null)[],
  chosen: PlayAction,
  basic: PlayAction,
  dev: PlayAction | null,
  net: MockNet | null,
  feats: TraceFeature[],
): Trace {
  const legal = t.legalActions()
  const layers = net ? netLayers(net, feats, 'play', values) : []
  return {
    kind: 'play',
    features: feats,
    layers,
    outputs: {
      labels: [...PLAY_LABELS],
      values: values.map((v) => (v === null ? null : Math.round(v * 10000) / 10000)),
      legal: PLAY_LABELS.map((l) => legal.includes(l)),
      unit: 'EV per unit bet',
    },
    chosen,
    reference: { basic_strategy: basic, hilo_deviation: dev },
  }
}

function netLayers(net: MockNet, feats: TraceFeature[], head: 'play' | 'bet', headValues: (number | null)[]) {
  const x = feats.map((f) => f.value)
  const f = net.forward(x)
  const r = (a: number[]) => a.map((v) => Math.round(v * 1000) / 1000)
  return [
    { name: 'input', size: x.length, activations: r(x) },
    { name: 'trunk.0', size: f.h1.length, activations: r(f.h1) },
    { name: 'trunk.1', size: f.h2.length, activations: r(f.h2) },
    head === 'play'
      ? { name: 'head.play', size: 5, activations: headValues.map((v) => (v === null ? 0 : Math.round(v * 1000) / 1000)) }
      : { name: 'head.bet', size: 6, activations: headValues.map((v) => (v === null ? 0 : Math.round(v * 1000) / 1000)) },
  ]
}

function betTrace(
  t: MockTable,
  net: MockNet | null,
  pred: Record<string, number>,
  bets: Bets,
  frac: number,
): Trace {
  const feats = features(t)
  const values = BET_LABELS.map((k) => pred[k] ?? 0)
  const placed: string[] = [`main €${bets.main}`]
  for (const k of ['perfect_pairs', 'twenty_one_plus_three', 'crazy_7', 'bust_bonus'] as const) if (bets[k]) placed.push(k)
  return {
    kind: 'bet',
    features: feats,
    layers: net ? netLayers(net, feats, 'bet', values) : [],
    outputs: {
      labels: [...BET_LABELS],
      values: values.map((v) => Math.round(v * 10000) / 10000),
      legal: BET_LABELS.map((k) => k !== 'insurance'),
      unit: 'predicted EV per unit staked',
    },
    chosen: placed.join(' + '),
    reference: { basic_strategy: `flat €${BASE_UNIT}`, hilo_deviation: null },
    extra: { predicted_ev: { ...pred }, bet_fraction: Math.round(frac * 10000) / 10000, bets: { ...bets } },
  }
}

function insuranceTrace(t: MockTable, net: MockNet | null, ev: number, insure: boolean): Trace {
  const feats = features(t, true)
  const values = [0, Math.round(ev * 0.5 * 10000) / 10000]
  return {
    kind: 'insurance',
    features: feats,
    layers: net ? netLayers(net, feats, 'bet', [0, 0, 0, 0, 0, values[1]]) : [],
    outputs: { labels: ['no_insurance', 'insure'], values, legal: [true, t.legalActions().includes('insure')], unit: 'EV per unit main bet' },
    chosen: insure ? 'insure' : 'no_insurance',
    reference: { basic_strategy: 'no_insurance', hilo_deviation: t.trueCount('hilo') >= 3 ? 'insure' : null },
  }
}

function insuranceEv(t: MockTable) {
  const p = bucketProbs(t.unseen)
  return p[9] * (1 + t.cfg.rules.insurance_payout) - 1
}

export class NNMockAgent implements MockAgent {
  name: string
  net: MockNet
  constructor(name: string, net: MockNet) {
    this.name = name
    this.net = net
  }
  decideBets(t: MockTable, a: ShoeAnalytics | null, withTrace: boolean) {
    const tc = t.trueCount('hilo')
    const noise = withTrace ? this.net.forward(features(t).map((f) => f.value)).bet : [0, 0, 0, 0, 0, 0]
    const pred: Record<string, number> = {
      main: -0.0065 + 0.0052 * tc + noise[0] * 0.002,
      perfect_pairs: (a?.perfect_pairs ?? -0.041) + noise[1] * 0.004,
      twenty_one_plus_three: (a?.twenty_one_plus_three ?? -0.037) + noise[2] * 0.004,
      crazy_7: (a?.crazy_7 ?? -0.057) + noise[3] * 0.004,
      bust_bonus: (a?.bust_bonus ?? -0.062) + noise[4] * 0.004,
      insurance: (a?.insurance ?? insuranceEv(t)) + noise[5] * 0.004,
    }
    // half-Kelly-ish spread on the predicted main edge (variance ≈ 1.33 per unit)
    const bank = t.sizingBankroll ?? t.balance
    const units = pred.main > 0 ? Math.min(8, 1 + Math.round(pred.main * 300)) : 1
    const frac = (units * BASE_UNIT) / Math.max(1, bank)
    const bets: Bets = { main: units * BASE_UNIT, perfect_pairs: 0, twenty_one_plus_three: 0, crazy_7: 0, bust_bonus: 0 }
    for (const k of ['perfect_pairs', 'twenty_one_plus_three', 'crazy_7', 'bust_bonus'] as const) {
      if (pred[k] > 0.01) bets[k] = 5
    }
    clampBets(t, bets)
    return { bets, trace: withTrace ? betTrace(t, this.net, pred, bets, frac) : null }
  }
  decideInsurance(t: MockTable, withTrace: boolean) {
    const ev = insuranceEv(t)
    const insure = ev > 0 && t.legalActions().includes('insure')
    return { insure, trace: withTrace ? insuranceTrace(t, this.net, ev, insure) : null }
  }
  decideAction(t: MockTable, withTrace: boolean) {
    const { legal, evs, basic, h, cashout } = playContext(t)
    const feats = withTrace ? features(t) : null
    const jitter = feats ? this.net.forward(feats.map((f) => f.value)).play : [0, 0, 0, 0, 0]
    const vals: (number | null)[] = [
      evs.stand + jitter[0] * 0.012,
      legal.includes('hit') ? evs.hit + jitter[1] * 0.012 : null,
      legal.includes('double') && evs.double !== null ? evs.double + jitter[2] * 0.012 : null,
      legal.includes('split') && evs.split !== null ? evs.split + jitter[3] * 0.012 : null,
      cashout,
    ]
    let best = 0
    vals.forEach((v, i) => {
      if (v !== null && v > (vals[best] ?? -Infinity)) best = i
    })
    // The mock "network" only deviates from basic strategy when its composition-aware EV
    // clearly beats it (keeps agreement realistic, ~95%); cash-out needs a clear edge too.
    const bi = PLAY_LABELS.indexOf(basic)
    const margin = PLAY_LABELS[best] === 'cashout' ? 0.02 : 0.03
    if (bi >= 0 && vals[bi] !== null && (vals[best] as number) - (vals[bi] as number) < margin) best = bi
    const action = PLAY_LABELS[best]
    const dev = hiloDeviation(t, h.cards, t.dealer[0], legal)
    return { action, trace: feats ? playTrace(t, vals, action, basic, dev, this.net, feats) : null }
  }
}

export class HiLoAgent implements MockAgent {
  name = 'hilo'
  decideBets(t: MockTable, a: ShoeAnalytics | null, withTrace: boolean) {
    const tc = Math.floor(t.trueCount('hilo'))
    const units = tc <= 1 ? 1 : Math.min(12, (tc - 1) * 2)
    const bets: Bets = { main: units * BASE_UNIT, perfect_pairs: 0, twenty_one_plus_three: 0, crazy_7: 0, bust_bonus: 0 }
    if (a) for (const k of ['perfect_pairs', 'twenty_one_plus_three', 'crazy_7', 'bust_bonus'] as const) if (a[k] > 0) bets[k] = 5
    clampBets(t, bets)
    const pred: Record<string, number> = {
      main: -0.0065 + 0.005 * t.trueCount('hilo'),
      perfect_pairs: a?.perfect_pairs ?? 0,
      twenty_one_plus_three: a?.twenty_one_plus_three ?? 0,
      crazy_7: a?.crazy_7 ?? 0,
      bust_bonus: a?.bust_bonus ?? 0,
      insurance: a?.insurance ?? 0,
    }
    return { bets, trace: withTrace ? betTrace(t, null, pred, bets, 0) : null }
  }
  decideInsurance(t: MockTable, withTrace: boolean) {
    const insure = t.trueCount('hilo') >= 3 && t.legalActions().includes('insure')
    return { insure, trace: withTrace ? insuranceTrace(t, null, insuranceEv(t), insure) : null }
  }
  decideAction(t: MockTable, withTrace: boolean) {
    const { legal, evs, basic, h, cashout } = playContext(t)
    const dev = hiloDeviation(t, h.cards, t.dealer[0], legal)
    const action = dev ?? basic
    const vals = [evs.stand, legal.includes('hit') ? evs.hit : null, legal.includes('double') ? evs.double : null, legal.includes('split') ? evs.split : null, cashout]
    return { action, trace: withTrace ? playTrace(t, vals, action, basic, dev, null, features(t)) : null }
  }
}

export class BasicAgent implements MockAgent {
  name = 'basic'
  decideBets(t: MockTable, a: ShoeAnalytics | null, withTrace: boolean) {
    const bets = clampBets(t, { main: BASE_UNIT, perfect_pairs: 0, twenty_one_plus_three: 0, crazy_7: 0, bust_bonus: 0 })
    const pred: Record<string, number> = { main: -0.0072, ...(a ? { perfect_pairs: a.perfect_pairs, twenty_one_plus_three: a.twenty_one_plus_three, crazy_7: a.crazy_7, bust_bonus: a.bust_bonus, insurance: a.insurance } : {}) }
    return { bets, trace: withTrace ? betTrace(t, null, pred, bets, 0) : null }
  }
  decideInsurance(t: MockTable, withTrace: boolean) {
    return { insure: false, trace: withTrace ? insuranceTrace(t, null, insuranceEv(t), false) : null }
  }
  decideAction(t: MockTable, withTrace: boolean) {
    const { legal, evs, basic, cashout } = playContext(t)
    const vals = [evs.stand, legal.includes('hit') ? evs.hit : null, legal.includes('double') ? evs.double : null, legal.includes('split') ? evs.split : null, cashout]
    return { action: basic, trace: withTrace ? playTrace(t, vals, basic, basic, null, null, features(t)) : null }
  }
}
