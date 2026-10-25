import type {
  AiState,
  ClientCmd,
  GameConfig,
  HelloMsg,
  MainSideBet,
  Metrics,
  ServerMsg,
  ShoeAnalytics,
  ShoeState,
  StateMsg,
  Trace,
  CountSystem,
} from '../protocol'
import { DEFAULT_CONFIG } from './config'
import { MockTable, round2, type RoundSummary } from './engine'
import { BasicAgent, HiLoAgent, MockNet, NNMockAgent, type MockAgent } from './network'
import { computeAnalytics, twentyOnePlus3Ev } from './analytics'
import { parseCode } from './cards'
import { basicAction } from './strategy'

/**
 * In-browser stand-in for the FastAPI WebSocket server (SPEC §6). It speaks the exact
 * same JSON messages so the rest of the app cannot tell the difference.
 */

const NN_NAME = 'nn:v1-mock'
const SYSTEMS: CountSystem[] = ['hilo', 'ko', 'omega2', 'wong_halves', 'zen']

class MetricsTracker {
  rounds = 0
  net = 0
  wagered = 0
  wins = 0
  losses = 0
  pushes = 0
  blackjacks = 0
  charlies = 0
  bankroll: [number, number][] = []
  stride = 1
  side: Partial<Record<MainSideBet | 'insurance', { wagered: number; net: number }>> = {}
  agreeN = 0
  agreeTotal = 0
  peak: number
  maxDD = 0
  rps: number | null = null
  constructor(public start: number) {
    this.peak = start
    this.bankroll.push([0, start])
  }
  decision(agrees: boolean) {
    this.agreeTotal++
    if (agrees) this.agreeN++
  }
  record(s: RoundSummary) {
    this.rounds++
    this.net += s.net
    this.wagered += s.wagered
    if (s.outcome === 'win') this.wins++
    else if (s.outcome === 'lose') this.losses++
    else this.pushes++
    if (s.blackjack) this.blackjacks++
    this.charlies += s.charlies
    for (const [k, v] of Object.entries(s.side)) {
      const key = k as MainSideBet | 'insurance'
      const cur = this.side[key] ?? { wagered: 0, net: 0 }
      cur.wagered += v!.wagered
      cur.net += v!.net
      this.side[key] = cur
    }
    const bal = this.start + this.net
    this.peak = Math.max(this.peak, bal)
    this.maxDD = Math.max(this.maxDD, this.peak - bal)
    if (this.rounds % this.stride === 0) {
      this.bankroll.push([this.rounds, round2(bal)])
      if (this.bankroll.length > 1500) {
        this.bankroll = this.bankroll.filter((_, i) => i % 2 === 0)
        this.stride *= 2
      }
    }
  }
  json(): Metrics {
    const bal = this.start + this.net
    const last = this.bankroll[this.bankroll.length - 1]
    const bankroll = last && last[0] === this.rounds ? this.bankroll : [...this.bankroll, [this.rounds, round2(bal)] as [number, number]]
    const side: Metrics['side_bets'] = {}
    for (const [k, v] of Object.entries(this.side)) side[k as MainSideBet] = { wagered: round2(v!.wagered), net: round2(v!.net) }
    return {
      rounds: this.rounds,
      net: round2(this.net),
      wagered: round2(this.wagered),
      ev_per_round: this.rounds ? this.net / this.rounds : 0,
      edge: this.wagered ? this.net / this.wagered : 0,
      wins: this.wins,
      losses: this.losses,
      pushes: this.pushes,
      blackjacks: this.blackjacks,
      charlies: this.charlies,
      bankroll,
      side_bets: side,
      agreement_basic: this.agreeTotal ? this.agreeN / this.agreeTotal : null,
      max_drawdown: round2(this.maxDD),
      rounds_per_sec: this.rps,
    }
  }
}

interface SimRun {
  table: MockTable
  agent: MockAgent
  metrics: MetricsTracker
  total: number
  done: number
  started: number
  analytics: ShoeAnalytics | null
  t21p3: number
  stop: boolean
}

export type Listener = (msg: ServerMsg) => void

export class MockServer {
  private listeners = new Set<Listener>()
  cfg: GameConfig = structuredClone(DEFAULT_CONFIG)
  table: MockTable
  net = new MockNet()
  agents: Record<string, MockAgent>
  agentName = NN_NAME
  private queue: StateMsg[] = []
  private drainTimer: ReturnType<typeof setTimeout> | null = null
  private autoTimer: ReturnType<typeof setTimeout> | null = null
  autoplay = false
  delayMs = 700
  advice = false
  lastTrace: Trace | null = null
  metrics: MetricsTracker
  sim: SimRun | null = null
  private analyticsCache: { key: string; value: ShoeAnalytics } | null = null
  private seed = 20261025

  constructor() {
    this.table = this.makeTable(this.cfg.bankroll.start)
    this.metrics = new MetricsTracker(this.cfg.bankroll.start)
    this.agents = {
      [NN_NAME]: new NNMockAgent(NN_NAME, this.net),
      hilo: new HiLoAgent(),
      basic: new BasicAgent(),
    }
  }

  private makeTable(bankroll: number) {
    const t = new MockTable(this.cfg, this.seed++)
    t.balance = bankroll
    t.onSettle = (sum) => this.metrics.record(sum)
    t.onFrame = () => this.queue.push(this.snapshot())
    return t
  }

  // ───────────────────────── transport ─────────────────────────

  subscribe(fn: Listener) {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  private emit(msg: ServerMsg) {
    // deliver asynchronously and as plain JSON, like a socket would
    const copy = JSON.parse(JSON.stringify(msg)) as ServerMsg
    for (const l of this.listeners) l(copy)
  }

  start() {
    setTimeout(() => {
      this.emit(this.hello())
      this.emit(this.snapshot())
    }, 30)
  }

  close() {
    this.autoplay = false
    if (this.sim) this.sim.stop = true
    if (this.drainTimer) clearTimeout(this.drainTimer)
    if (this.autoTimer) clearTimeout(this.autoTimer)
    this.listeners.clear()
  }

  /** Non-protocol helper for tests/screenshots: force the next cards dealt (codes like "8_of_hearts"). */
  rig(codes: string[]) {
    this.table.rigged = codes.map(parseCode)
  }

  hello(): HelloMsg {
    return {
      type: 'hello',
      config: this.cfg,
      network: this.net.describe(NN_NAME),
      agents: Object.keys(this.agents),
    }
  }

  send(cmd: ClientCmd | { cmd: 'mock_rig'; cards: string[] }) {
    try {
      this.handle(cmd)
    } catch (e) {
      this.emit({ type: 'error', message: e instanceof Error ? e.message : String(e) })
      // re-sync the client with the authoritative state
      if (!this.busy) this.emit(this.snapshot())
    }
  }

  private get busy() {
    return this.queue.length > 0 || this.drainTimer !== null
  }

  private handle(cmd: ClientCmd | { cmd: 'mock_rig'; cards: string[] }) {
    switch (cmd.cmd) {
      case 'mock_rig':
        this.rig(cmd.cards)
        return
      case 'place_bets':
        this.guardIdle()
        this.table.placeBets(cmd.bets)
        break
      case 'action':
        this.guardIdle()
        this.table.act(cmd.action)
        break
      case 'ai_step':
        this.guardIdle()
        this.aiStep()
        break
      case 'autoplay':
        this.autoplay = cmd.on
        if (cmd.delay_ms !== undefined) this.delayMs = Math.max(0, cmd.delay_ms)
        if (!cmd.on && this.autoTimer) {
          clearTimeout(this.autoTimer)
          this.autoTimer = null
        }
        this.queue.push(this.snapshot())
        break
      case 'advice':
        this.advice = cmd.on
        this.queue.push(this.snapshot())
        break
      case 'simulate':
        this.startSim(cmd.rounds, cmd.agent ?? this.agentName)
        return
      case 'stop':
        this.autoplay = false
        if (this.autoTimer) clearTimeout(this.autoTimer)
        this.autoTimer = null
        if (this.sim) this.sim.stop = true
        this.queue.push(this.snapshot())
        break
      case 'set_agent':
        if (!this.agents[cmd.agent]) throw new Error(`Unknown agent ${cmd.agent}`)
        this.agentName = cmd.agent
        this.lastTrace = null
        this.queue.push(this.snapshot())
        break
      case 'new_shoe':
        if (this.table.phase !== 'betting' && this.table.phase !== 'settled') throw new Error('Finish the round first')
        this.table.newShoe()
        this.table.messages = [`New shoe #${this.table.shoeNumber} — shuffled and split into two halves`]
        this.queue.push(this.snapshot())
        break
      case 'reset': {
        this.autoplay = false
        const bank = cmd.bankroll ?? this.cfg.bankroll.start
        this.table = this.makeTable(bank)
        this.metrics = new MetricsTracker(bank)
        this.lastTrace = null
        this.queue = []
        this.queue.push(this.snapshot())
        break
      }
      case 'set_config': {
        deepMerge(this.cfg as unknown as Record<string, unknown>, cmd.patch)
        this.table.cfg = this.cfg
        this.emit(this.hello())
        this.queue.push(this.snapshot())
        break
      }
    }
    this.drain()
  }

  private guardIdle() {
    if (this.busy) throw new Error('Please wait — cards are still being dealt')
    if (this.sim && !this.sim.stop) throw new Error('A simulation is running — stop it first')
  }

  // ───────────────────────── pacing ─────────────────────────

  private frameDelay() {
    if (this.autoplay) return Math.max(30, Math.min(600, this.delayMs * 0.45))
    return 430
  }

  private drain() {
    if (this.drainTimer) return
    const next = this.queue.shift()
    if (!next) {
      this.drainTimer = null
      this.scheduleAuto()
      return
    }
    this.emit(next)
    if (this.queue.length === 0) {
      this.drainTimer = null
      this.scheduleAuto()
      return
    }
    this.drainTimer = setTimeout(() => {
      this.drainTimer = null
      this.drain()
    }, this.frameDelay())
  }

  private scheduleAuto() {
    if (!this.autoplay || this.autoTimer || (this.sim && !this.sim.stop)) return
    const settled = this.table.phase === 'settled' || this.table.phase === 'betting'
    const wait = settled ? this.delayMs * 1.6 + 150 : this.delayMs * 0.8 + 60
    this.autoTimer = setTimeout(() => {
      this.autoTimer = null
      if (!this.autoplay) return
      try {
        this.aiStep()
      } catch (e) {
        this.autoplay = false
        this.emit({ type: 'error', message: `Autoplay stopped: ${e instanceof Error ? e.message : String(e)}` })
        this.queue.push(this.snapshot())
      }
      this.drain()
    }, wait)
  }

  // ───────────────────────── AI ─────────────────────────

  private get agent() {
    return this.agents[this.agentName]
  }

  private aiStep() {
    const t = this.table
    const a = this.agent
    if (t.phase === 'betting' || t.phase === 'settled') {
      if (t.balance < t.cfg.limits.main_min) throw new Error('Bankroll exhausted — reset the session')
      const { bets, trace } = a.decideBets(t, this.analytics(), true)
      this.lastTrace = trace
      t.placeBets(bets)
      return
    }
    if (t.phase === 'insurance') {
      const { insure, trace } = a.decideInsurance(t, true)
      this.lastTrace = trace
      t.act(insure ? 'insure' : 'no_insurance')
      return
    }
    if (t.phase === 'player' && t.legalActions().length) {
      const { action, trace } = a.decideAction(t, true)
      this.lastTrace = trace
      if (trace?.reference?.basic_strategy) this.metrics.decision(trace.reference.basic_strategy === action)
      t.act(action)
      return
    }
    throw new Error('Nothing to decide right now')
  }

  private analytics(): ShoeAnalytics {
    const t = this.table
    const key = `${t.shoeNumber}:${t.half}:${t.pos}:${t.cardsUnseen}`
    if (this.analyticsCache?.key === key) return this.analyticsCache.value
    const value = computeAnalytics(t.unseen, this.cfg, t.trueCount('hilo'))
    for (const k of Object.keys(value) as (keyof ShoeAnalytics)[]) {
      const v = value[k]
      if (typeof v === 'number') (value as Record<string, unknown>)[k] = Math.round(v * 100000) / 100000
    }
    this.analyticsCache = { key, value }
    return value
  }

  private adviceTrace(): Trace | null {
    if (!this.advice) return null
    const t = this.table
    if (t.dealing) return null
    try {
      if (t.phase === 'betting' || t.phase === 'settled') return this.agent.decideBets(t, this.analytics(), true).trace
      if (t.phase === 'insurance') return this.agent.decideInsurance(t, true).trace
      if (t.phase === 'player' && t.legalActions().length) return this.agent.decideAction(t, true).trace
    } catch {
      return null
    }
    return null
  }

  snapshot(): StateMsg {
    const t = this.table
    const ai: AiState = {
      agent: this.agentName,
      advice: this.adviceTrace(),
      last_trace: this.lastTrace,
      autoplay: this.autoplay,
    }
    return { type: 'state', game: t.gameJson(), shoe: this.shoeJson(), ai, metrics: this.metrics.json() }
  }

  private shoeJson(): ShoeState {
    const t = this.table
    const running: ShoeState['running'] = {}
    const tc: ShoeState['true'] = {}
    for (const s of SYSTEMS) {
      running[s] = round2(t.running[s])
      tc[s] = round2(t.trueCount(s))
    }
    return {
      shoe_number: t.shoeNumber,
      half: t.half,
      half_size: t.halfSize,
      dealt_in_half: t.pos,
      cut_at: t.cutAt,
      burned: t.cfg.shoe.burn_cards,
      cards_unseen: t.cardsUnseen,
      decks_unseen: round2(t.cardsUnseen / 52),
      seen_counts: t.bucketRecord(t.seenBuckets),
      unseen_counts: t.bucketRecord(t.unseenBuckets()),
      running,
      true: tc,
      analytics: this.analytics(),
      history: t.history.slice(-400),
    }
  }

  // ───────────────────────── simulation ─────────────────────────

  private startSim(rounds: number, agentName: string) {
    if (this.sim && !this.sim.stop) throw new Error('A simulation is already running')
    const agent = this.agents[agentName]
    if (!agent) throw new Error(`Unknown agent ${agentName}`)
    this.autoplay = false
    const table = new MockTable(this.cfg, this.seed++)
    const start = this.cfg.bankroll.start
    table.balance = 1e12 // virtual unlimited bankroll; metrics report start + net
    table.sizingBankroll = start
    this.sim = {
      table,
      agent,
      metrics: new MetricsTracker(start),
      total: Math.max(1, Math.floor(rounds)),
      done: 0,
      started: performance.now(),
      analytics: null,
      t21p3: -0.037,
      stop: false,
    }
    this.queue.push(this.snapshot())
    this.drain()
    this.simTick()
  }

  private simTick() {
    const s = this.sim
    if (!s) return
    const t0 = performance.now()
    while (!s.stop && s.done < s.total && performance.now() - t0 < 28) {
      for (let i = 0; i < 25 && s.done < s.total; i++) {
        this.simRound(s)
        s.done++
      }
    }
    const elapsed = (performance.now() - s.started) / 1000
    s.metrics.rps = elapsed > 0 ? Math.round(s.done / elapsed) : null
    const finished = s.stop || s.done >= s.total
    this.emit({ type: 'sim_progress', done: s.done, total: s.total, metrics: s.metrics.json() })
    if (finished) {
      s.stop = true
      this.emit(this.snapshot())
      return
    }
    setTimeout(() => this.simTick(), 0)
  }

  private simRound(s: SimRun) {
    const t = s.table
    if (s.done % 10 === 0 || !s.analytics) {
      if (s.done % 200 === 0) s.t21p3 = twentyOnePlus3Ev(t.unseen, this.cfg)
      const p = computeAnalyticsNo21p3(t, this.cfg)
      p.twenty_one_plus_three = s.t21p3
      s.analytics = p
    }
    const { bets } = s.agent.decideBets(t, s.analytics, false)
    t.placeBets(bets)
    let guard = 0
    while (t.phase !== 'settled' && guard++ < 50) {
      if (t.phase === 'insurance') t.act(s.agent.decideInsurance(t, false).insure ? 'insure' : 'no_insurance')
      else if (t.phase === 'player') {
        const legal = t.legalActions()
        const { action } = s.agent.decideAction(t, false)
        const h = t.hands[t.active]
        const basic = basicAction(h.cards, t.dealer[0], legal.includes('double'), legal.includes('split'))
        s.metrics.decision(basic === action)
        t.act(action)
      } else break
    }
    if (t.lastSummary) s.metrics.record(t.lastSummary)
  }

}

/** Analytics without the expensive 21+3 enumeration (used inside the fast simulation loop). */
function computeAnalyticsNo21p3(t: MockTable, cfg: GameConfig): ShoeAnalytics {
  return computeAnalytics(t.unseen, cfg, t.trueCount('hilo'), true)
}

function deepMerge(target: Record<string, unknown>, patch: Record<string, unknown>) {
  for (const [k, v] of Object.entries(patch)) {
    if (v && typeof v === 'object' && !Array.isArray(v) && target[k] && typeof target[k] === 'object') {
      deepMerge(target[k] as Record<string, unknown>, v as Record<string, unknown>)
    } else target[k] = v
  }
}
