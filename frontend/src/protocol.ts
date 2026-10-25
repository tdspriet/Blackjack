/**
 * Wire protocol between the ONE Blackjack Lab backend and this frontend.
 * Mirrors docs/SPEC.md §5 (Trace, NNAgent.describe) and §6 (REST + WebSocket) exactly.
 *
 * Fields marked "(frontend assumption)" are not spelled out in SPEC; they are optional
 * so the UI degrades gracefully if the backend omits them.
 */

// ───────────────────────────── primitives ─────────────────────────────

export type Rank = 'A' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10' | 'J' | 'Q' | 'K'
export type Suit = 'hearts' | 'diamonds' | 'clubs' | 'spades'
export type Owner = 'player' | 'crowd' | 'dealer'

/** Rank buckets used for composition / counts: A, 2..9, T (all ten-valued). */
export type RankBucket = 'A' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | 'T'
export const RANK_BUCKETS: RankBucket[] = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T']

export interface Card {
  /** Absent on face-down cards. */
  rank?: Rank
  suit?: Suit
  /** e.g. "A_of_hearts"; "back" when face down. Image at /cards/<code>.png */
  code: string
  face_up: boolean
  owner?: Owner
}

export type Phase = 'betting' | 'insurance' | 'player' | 'crowd' | 'dealer' | 'settled'
export type PlayAction = 'hit' | 'stand' | 'double' | 'split' | 'cashout'
export type InsuranceAction = 'insure' | 'no_insurance'
export type Action = PlayAction | InsuranceAction
export type LegalAction = Action | 'deal'

export type MainSideBet = 'perfect_pairs' | 'twenty_one_plus_three' | 'crazy_7' | 'bust_bonus'
export const SIDE_BETS: MainSideBet[] = ['bust_bonus', 'twenty_one_plus_three', 'perfect_pairs', 'crazy_7']

export interface Bets {
  main: number
  perfect_pairs: number
  twenty_one_plus_three: number
  crazy_7: number
  bust_bonus: number
}
export type BetKey = keyof Bets

export interface GameBets extends Bets {
  insurance: number
}

// ───────────────────────────── config (§3) ─────────────────────────────

export interface GameConfig {
  shoe: { num_decks: number; split_halves: boolean; penetration: number; cut_jitter: number; burn_cards: number }
  rules: {
    blackjack_payout: number
    dealer_hits_soft_17: boolean
    peek_on_ace: boolean
    peek_on_ten: boolean
    late_dealer_bj_takes: 'all' | 'original'
    double_any_two: boolean
    double_after_split: boolean
    max_splits: number
    split_by_value: boolean
    split_aces_one_card: boolean
    charlie_cards: number
    charlie_beats_dealer_bj: boolean
    insurance: boolean
    insurance_payout: number
    surrender: boolean
    cashout?: boolean
    cashout_rtp?: number
  }
  limits: { main_min: number; main_max: number; side_min: number; side_max: number; chips: number[] }
  side_bets: {
    perfect_pairs: { mixed: number; coloured: number; perfect: number }
    twenty_one_plus_three: {
      flush: number
      straight: number
      three_of_a_kind: number
      straight_flush: number
      suited_trips: number
    }
    crazy_7: { one: number; pair_unsuited: number; pair_suited: number; three_unsuited: number; three_suited: number }
    bust_bonus: Record<string, number>
    bust_bonus_push_on_player_blackjack?: boolean
    /** older config key (pre official-rules update) */
    bust_bonus_if_dealer_not_playing?: 'lose' | 'push'
  }
  crowd: { enabled: boolean; size: number; min_hitters: number; mix: Record<string, number> }
  bankroll: { start: number }
}

// ───────────────────────────── NN (§5) ─────────────────────────────

export type TraceKind = 'play' | 'bet' | 'insurance'

export interface TraceFeature {
  name: string
  /** Normalised value fed to the network. */
  value: number
  /** Human-readable raw value (e.g. 16 for player_total). */
  raw?: number | string | boolean | null
}

export interface TraceLayer {
  name: string
  size: number
  activations: number[]
}

export interface TraceOutputs {
  labels: string[]
  values: (number | null)[]
  legal: boolean[]
  unit?: string
}

export interface Trace {
  kind: TraceKind
  features: TraceFeature[]
  layers: TraceLayer[]
  outputs: TraceOutputs
  chosen: string
  reference?: { basic_strategy?: string | null; hilo_deviation?: string | null; [k: string]: string | null | undefined }
  extra?: {
    predicted_ev?: Partial<Record<string, number>>
    bet_fraction?: number
    /** (frontend assumption) the bets that resulted from a bet trace. */
    bets?: Partial<Bets>
    [k: string]: unknown
  }
}

export interface NetworkLayer {
  name: string
  size: number
  labels?: string[]
}

/** (frontend assumption) SPEC only says `"heads": [...]`. */
export interface NetworkHead {
  name: string
  size: number
  labels?: string[]
  /** Name of the layer this head is attached to (defaults to the last trunk layer). */
  from?: string
}

export interface NetworkDescription {
  name: string
  checkpoint?: string | null
  layers: NetworkLayer[]
  heads: NetworkHead[]
  /** "<layerA>-><layerB>": top-K weights as [i, j, w] (i in A, j in B). */
  edges: Record<string, [number, number, number][]>
  training?: Record<string, unknown>
}

// ───────────────────────────── server → client (§6) ─────────────────────────────

export type HandStatus = 'active' | 'stood' | 'bust' | 'blackjack' | 'charlie' | 'cashed_out' | 'done'
export type HandResult = 'win' | 'lose' | 'push' | 'blackjack' | 'charlie' | 'cashout'

export interface DealerState {
  cards: Card[]
  total: number
  soft: boolean
  blackjack: boolean
  bust: boolean
}

export interface HandState {
  cards: Card[]
  bet: number
  doubled: boolean
  total: number
  soft: boolean
  status: HandStatus
  result: HandResult | null
  payout: number
  split_aces: boolean
  /** Current cash-out offer in € for this hand (main bet only); null when unavailable. */
  cashout_offer?: number | null
}

export interface CrowdState {
  enabled: boolean
  cards: Card[]
  hitters: number
  standers: number
}

export interface SideBetResult {
  bet: MainSideBet | 'insurance'
  outcome: string
  odds: number
  stake: number
  payout: number
}

export interface GameState {
  round_id: number
  phase: Phase
  balance: number
  total_bet: number
  bets: GameBets
  dealer: DealerState
  hands: HandState[]
  active_hand: number
  crowd: CrowdState
  legal_actions: LegalAction[]
  side_bet_results: SideBetResult[]
  round_net: number | null
  messages: string[]
}

export type CountSystem = 'hilo' | 'ko' | 'omega2' | 'wong_halves' | 'zen'
export const COUNT_SYSTEMS: CountSystem[] = ['hilo', 'ko', 'omega2', 'wong_halves', 'zen']

export interface ShoeAnalytics {
  perfect_pairs: number
  twenty_one_plus_three: number
  crazy_7: number
  bust_bonus: number
  insurance: number
  main_edge_estimate: number
  /** (frontend assumption on keys) rank bucket of the dealer up card → P(dealer busts). */
  p_dealer_bust_by_upcard: Partial<Record<RankBucket, number>>
  [k: string]: unknown
}

export interface CountHistoryPoint {
  round: number
  tc: number
  rc: number
  /** (frontend assumption) which shoe half the round was dealt from. */
  half?: number
}

export interface ShoeState {
  shoe_number: number
  half: number
  half_size: number
  /** Cards removed from the current half so far (frontend assumes this includes the burn card). */
  dealt_in_half: number
  cut_at: number
  burned: number
  cards_unseen: number
  decks_unseen: number
  seen_counts: Record<RankBucket, number>
  unseen_counts: Record<RankBucket, number>
  running: Partial<Record<CountSystem, number>>
  true: Partial<Record<CountSystem, number>>
  analytics: ShoeAnalytics
  history: CountHistoryPoint[]
}

export interface AiState {
  agent: string
  advice: Trace | null
  last_trace: Trace | null
  autoplay: boolean
}

export interface SideBetMetric {
  wagered: number
  net: number
}

export interface Metrics {
  rounds: number
  net: number
  wagered: number
  ev_per_round: number
  edge: number
  wins: number
  losses: number
  pushes: number
  blackjacks: number
  charlies: number
  bankroll: [number, number][]
  side_bets: Partial<Record<MainSideBet | 'insurance', SideBetMetric>>
  agreement_basic: number | null
  max_drawdown: number
  rounds_per_sec: number | null
}

export interface HelloMsg {
  type: 'hello'
  config: GameConfig
  network: NetworkDescription | null
  agents: string[]
}

export interface StateMsg {
  type: 'state'
  game: GameState
  shoe: ShoeState
  ai: AiState
  metrics: Metrics
}

export interface SimProgressMsg {
  type: 'sim_progress'
  done: number
  total: number
  metrics: Metrics
  /** True on the last message of a run (completed or stopped early); absent → done >= total. */
  finished?: boolean
  /** Agent being simulated. */
  agent?: string
}

export interface ErrorMsg {
  type: 'error'
  message: string
}

export type ServerMsg = HelloMsg | StateMsg | SimProgressMsg | ErrorMsg

// ───────────────────────────── client → server (§6) ─────────────────────────────

export type ClientCmd =
  | { cmd: 'place_bets'; bets: Bets }
  | { cmd: 'action'; action: Action }
  | { cmd: 'ai_step' }
  | { cmd: 'autoplay'; on: boolean; delay_ms?: number }
  | { cmd: 'advice'; on: boolean }
  | { cmd: 'simulate'; rounds: number; agent?: string }
  | { cmd: 'stop' }
  | { cmd: 'set_agent'; agent: string }
  | { cmd: 'new_shoe' }
  | { cmd: 'reset'; bankroll?: number }
  | { cmd: 'set_config'; patch: Record<string, unknown> }

// ───────────────────────────── REST (§6) ─────────────────────────────

export interface CheckpointInfo {
  name: string
  created: string | number | null
  metrics: Record<string, number | string | null> | null
}

export interface TrainingLogRow {
  step: number
  loss?: number | null
  eval_ev?: number | null
  [k: string]: number | string | null | undefined
}

// ───────────────────────────── helpers ─────────────────────────────

export const BET_LABELS: Record<BetKey | 'insurance', string> = {
  main: 'Main',
  perfect_pairs: 'Perfect Pairs',
  twenty_one_plus_three: '21+3',
  crazy_7: 'Crazy 7',
  bust_bonus: 'Bust Bonus',
  insurance: 'Insurance',
}

export const EMPTY_BETS: Bets = { main: 0, perfect_pairs: 0, twenty_one_plus_three: 0, crazy_7: 0, bust_bonus: 0 }

export const COUNT_LABELS: Record<CountSystem, string> = {
  hilo: 'Hi-Lo',
  ko: 'KO',
  omega2: 'Omega II',
  wong_halves: 'Wong Halves',
  zen: 'Zen',
}
