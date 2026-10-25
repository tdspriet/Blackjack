import { create } from 'zustand'
import {
  EMPTY_BETS,
  type BetKey,
  type Bets,
  type AiState,
  type ClientCmd,
  type CountSystem,
  type HelloMsg,
  type Metrics,
  type ServerMsg,
  type StateMsg,
} from './protocol'
import { connectLive, connectMock, type Transport } from './transport'

export type ConnMode = 'connecting' | 'live' | 'mock'
export type ConnStatus = 'connecting' | 'open' | 'reconnecting' | 'closed'

export interface SimState {
  done: number
  total: number
  metrics: Metrics
  running: boolean
  agent: string
  startedAt: number
}

export interface Toast {
  id: number
  kind: 'error' | 'info'
  message: string
}

/** Remembered final state of half 1 (the protocol only reports the current half). */
export interface HalfMemo {
  shoe: number
  dealt: number
  cutAt: number
  /** first round dealt from half 2 */
  switchRound: number
}

interface LabState {
  conn: { mode: ConnMode; status: ConnStatus; note?: string }
  hello: HelloMsg | null
  state: StateMsg | null
  prev: StateMsg | null
  sim: SimState | null
  toasts: Toast[]
  halfMemo: HalfMemo | null

  // ui
  countSystem: CountSystem
  chip: number
  pending: Bets
  lastBets: Bets | null
  autoplayDelay: number
  adviceOn: boolean
  simRounds: number
  traceSource: 'auto' | 'last' | 'advice'
  metricsSource: 'session' | 'sim'

  // actions
  connect(forceMock?: boolean): Promise<void>
  send(cmd: ClientCmd): void
  toast(message: string, kind?: Toast['kind']): void
  dismiss(id: number): void
  set<K extends keyof LabState>(k: K, v: LabState[K]): void
  addChip(spot: BetKey, amount?: number): void
  removeChip(spot: BetKey): void
  clearBets(): void
  rebet(): void
  doubleBets(): void
  deal(): void
}

let transport: Transport | null = null
let toastId = 1
let reconnectTimer: ReturnType<typeof setTimeout> | null = null

const wantMock = () => {
  const q = new URLSearchParams(location.search)
  return q.get('mock') === '1' || q.get('mock') === 'true'
}

const withDefaults = <T extends object>(d: T, v: Partial<T> | undefined | null): T => ({ ...d, ...(v ?? {}) })

/** Fill in defaults so a partially-implemented backend can't crash the UI. */
function normalizeState(m: StateMsg): StateMsg {
  const g = m.game
  const sh = m.shoe ?? ({} as StateMsg['shoe'])
  const emptyMetrics: Metrics = {
    rounds: 0, net: 0, wagered: 0, ev_per_round: 0, edge: 0, wins: 0, losses: 0, pushes: 0, blackjacks: 0, charlies: 0,
    bankroll: [], side_bets: {}, agreement_basic: null, max_drawdown: 0, rounds_per_sec: null,
  }
  return {
    ...m,
    game: {
      ...g,
      bets: withDefaults({ main: 0, perfect_pairs: 0, twenty_one_plus_three: 0, crazy_7: 0, bust_bonus: 0, insurance: 0 }, g.bets),
      dealer: g.dealer ?? { cards: [], total: 0, soft: false, blackjack: false, bust: false },
      hands: g.hands ?? [],
      crowd: g.crowd ?? { enabled: false, cards: [], hitters: 0, standers: 0 },
      legal_actions: g.legal_actions ?? [],
      side_bet_results: g.side_bet_results ?? [],
      messages: g.messages ?? [],
      round_net: g.round_net ?? null,
    },
    shoe: {
      ...sh,
      seen_counts: sh.seen_counts ?? ({} as StateMsg['shoe']['seen_counts']),
      unseen_counts: sh.unseen_counts ?? ({} as StateMsg['shoe']['unseen_counts']),
      running: sh.running ?? {},
      true: sh.true ?? {},
      analytics: withDefaults({ p_dealer_bust_by_upcard: {} } as StateMsg['shoe']['analytics'], sh.analytics),
      history: sh.history ?? [],
    },
    ai: withDefaults<AiState>({ agent: '', advice: null, last_trace: null, autoplay: false }, m.ai),
    metrics: withDefaults(emptyMetrics, m.metrics),
  }
}

export const totalOf = (b: Bets) => b.main + b.perfect_pairs + b.twenty_one_plus_three + b.crazy_7 + b.bust_bonus

export const useLab = create<LabState>((set, get) => {
  const onMessage = (msg: ServerMsg) => {
    switch (msg.type) {
      case 'hello': {
        const chips = msg.config.limits.chips
        const chip = chips.includes(get().chip) ? get().chip : chips[Math.min(3, chips.length - 1)]
        set({ hello: msg, chip })
        break
      }
      case 'state': {
        msg = normalizeState(msg)
        const prev = get().state
        let halfMemo = get().halfMemo
        if (prev && prev.shoe.shoe_number === msg.shoe.shoe_number && prev.shoe.half === 1 && msg.shoe.half === 2) {
          halfMemo = {
            shoe: msg.shoe.shoe_number,
            dealt: prev.shoe.dealt_in_half,
            cutAt: prev.shoe.cut_at,
            switchRound: msg.game.round_id,
          }
        }
        if (halfMemo && halfMemo.shoe !== msg.shoe.shoe_number) halfMemo = null
        set({ state: msg, prev, halfMemo })
        break
      }
      case 'sim_progress': {
        const cur = get().sim
        const running = msg.finished !== undefined ? !msg.finished : msg.done < msg.total
        set({
          sim: {
            done: msg.done,
            total: msg.total,
            metrics: msg.metrics,
            running,
            agent: msg.agent ?? cur?.agent ?? get().state?.ai.agent ?? '',
            startedAt: cur?.startedAt ?? Date.now(),
          },
          metricsSource: 'sim',
        })
        break
      }
      case 'error':
        get().toast(msg.message, 'error')
        break
    }
  }

  const scheduleReconnect = () => {
    if (reconnectTimer) return
    reconnectTimer = setTimeout(async () => {
      reconnectTimer = null
      try {
        transport = await connectLive({ onMessage, onClose: handleClose })
        set({ conn: { mode: 'live', status: 'open' } })
        get().toast('Reconnected to backend', 'info')
      } catch {
        scheduleReconnect()
      }
    }, 2000)
  }

  const handleClose = (reason: string) => {
    if (transport?.kind !== 'live') return
    set({ conn: { mode: 'live', status: 'reconnecting', note: reason } })
    scheduleReconnect()
  }

  return {
    conn: { mode: 'connecting', status: 'connecting' },
    hello: null,
    state: null,
    prev: null,
    sim: null,
    toasts: [],
    halfMemo: null,

    countSystem: 'hilo',
    chip: 10,
    pending: { ...EMPTY_BETS },
    lastBets: null,
    autoplayDelay: 700,
    adviceOn: false,
    simRounds: 10000,
    traceSource: 'auto',
    metricsSource: 'session',

    async connect(forceMock = false) {
      transport?.close()
      transport = null
      if (reconnectTimer) clearTimeout(reconnectTimer)
      reconnectTimer = null
      set({ conn: { mode: 'connecting', status: 'connecting' }, state: null, prev: null, hello: null, sim: null, halfMemo: null })
      if (!forceMock && !wantMock()) {
        try {
          transport = await connectLive({ onMessage, onClose: handleClose })
          set({ conn: { mode: 'live', status: 'open' } })
          return
        } catch {
          /* fall through to mock */
        }
      }
      transport = await connectMock({ onMessage })
      set({
        conn: {
          mode: 'mock',
          status: 'open',
          note: wantMock() || forceMock ? 'Mock mode requested' : 'Backend unreachable — running the in-browser mock',
        },
      })
    },

    send(cmd) {
      if (!transport) {
        get().toast('Not connected', 'error')
        return
      }
      transport.send(cmd)
    },

    toast(message, kind = 'info') {
      const id = toastId++
      set({ toasts: [...get().toasts.slice(-3), { id, kind, message }] })
      setTimeout(() => get().dismiss(id), kind === 'error' ? 6000 : 3500)
    },
    dismiss(id) {
      set({ toasts: get().toasts.filter((t) => t.id !== id) })
    },

    set(k, v) {
      set({ [k]: v } as Partial<LabState>)
    },

    addChip(spot, amount) {
      const s = get()
      const cfg = s.hello?.config
      const st = s.state
      if (!cfg || !st) return
      const add = amount ?? s.chip
      const L = cfg.limits
      const b = { ...s.pending }
      const max = spot === 'main' ? L.main_max : L.side_max
      if (spot !== 'main' && b.main <= 0) {
        s.toast('Place a main bet first — side bets require a main bet', 'error')
        return
      }
      const next = Math.min(max, b[spot] + add)
      if (next === b[spot]) {
        s.toast(`${spot === 'main' ? 'Main' : 'Side'} bet limit is €${max}`, 'error')
        return
      }
      const delta = next - b[spot]
      if (totalOf(b) + delta > st.game.balance + 1e-9) {
        s.toast('Insufficient balance', 'error')
        return
      }
      b[spot] = next
      set({ pending: b })
    },

    removeChip(spot) {
      const s = get()
      const b = { ...s.pending }
      if (b[spot] <= 0) return
      b[spot] = Math.max(0, b[spot] - s.chip)
      if (spot === 'main' && b.main === 0) {
        b.perfect_pairs = b.twenty_one_plus_three = b.crazy_7 = b.bust_bonus = 0
      }
      set({ pending: b })
    },

    clearBets() {
      set({ pending: { ...EMPTY_BETS } })
    },

    rebet() {
      const s = get()
      if (!s.lastBets) return
      if (s.state && totalOf(s.lastBets) > s.state.game.balance) {
        s.toast('Insufficient balance to rebet', 'error')
        return
      }
      set({ pending: { ...s.lastBets } })
    },

    doubleBets() {
      const s = get()
      const cfg = s.hello?.config
      if (!cfg || !s.state) return
      const L = cfg.limits
      const b = { ...s.pending }
      b.main = Math.min(L.main_max, b.main * 2)
      for (const k of ['perfect_pairs', 'twenty_one_plus_three', 'crazy_7', 'bust_bonus'] as const) b[k] = Math.min(L.side_max, b[k] * 2)
      if (totalOf(b) > s.state.game.balance) {
        s.toast('Insufficient balance to double', 'error')
        return
      }
      set({ pending: b })
    },

    deal() {
      const s = get()
      const cfg = s.hello?.config
      const st = s.state
      if (!cfg || !st || !st.game.legal_actions.includes('deal')) return
      let bets = s.pending
      if (bets.main <= 0 && s.lastBets) bets = s.lastBets
      if (bets.main < cfg.limits.main_min) {
        s.toast(`Place a main bet (min €${cfg.limits.main_min})`, 'error')
        return
      }
      if (totalOf(bets) > st.game.balance + 1e-9) {
        s.toast('Insufficient balance', 'error')
        return
      }
      s.send({ cmd: 'place_bets', bets })
      set({ lastBets: { ...bets }, pending: { ...EMPTY_BETS } })
    },
  }
})
