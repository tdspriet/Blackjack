import type {
  CountSystem,
  GameBets,
  GameConfig,
  GameState,
  HandResult,
  HandStatus,
  LegalAction,
  MainSideBet,
  Owner,
  PlayAction,
  RankBucket,
  SideBetResult,
  Action,
  Bets,
  CountHistoryPoint,
} from '../protocol'
import { BUCKETS, bucketOf, cardJson, handTotal, isRed, rankIdx, suitIdx, valueOf } from './cards'
import { classify21p3 } from './analytics'
import { actionEvs, basicAction } from './strategy'
import { Rng } from './rng'

/**
 * In-browser re-implementation of the ONE Blackjack rules (SPEC §1) — just enough to
 * drive the UI in mock mode. The Python engine remains the source of truth.
 */

export const COUNT_TAGS: Record<CountSystem, number[]> = {
  //          A    2    3    4    5    6    7    8    9    T
  hilo: [-1, 1, 1, 1, 1, 1, 0, 0, 0, -1],
  ko: [-1, 1, 1, 1, 1, 1, 1, 0, 0, -1],
  omega2: [0, 1, 1, 2, 2, 2, 1, 0, -1, -2],
  wong_halves: [-1, 0.5, 1, 1, 1.5, 1, 0.5, 0, -0.5, -1],
  zen: [-1, 1, 1, 2, 2, 2, 1, 0, 0, -2],
}
const SYSTEMS = Object.keys(COUNT_TAGS) as CountSystem[]

type CrowdType = 'basic' | 'cautious' | 'dealer_mimic' | 'reckless'
interface CrowdGroup {
  type: CrowdType
  size: number
  in: boolean
  live: boolean // stood/doubled with a non-busted hand (dealer must play)
}

interface MHand {
  cards: number[]
  owners: Owner[]
  bet: number
  doubled: boolean
  status: HandStatus
  result: HandResult | null
  payout: number
  splitAces: boolean
  fromSplit: boolean
  cashedOutFor: number | null
}

/** Full 8-deck composition (the casino prices cash-out offers composition-independently). */
const FULL_SHOE_P = [1, 1, 1, 1, 1, 1, 1, 1, 1, 4].map((x) => x / 13)

export interface RoundSummary {
  wagered: number
  net: number
  outcome: 'win' | 'lose' | 'push'
  blackjack: boolean
  charlies: number
  side: Partial<Record<MainSideBet | 'insurance', { wagered: number; net: number }>>
}

export class MockTable {
  cfg: GameConfig
  rng: Rng
  onFrame: () => void = () => {}
  onSettle: (s: RoundSummary) => void = () => {}

  // ── shoe ──
  shoe: number[] = []
  shoeNumber = 0
  half = 1
  pos = 0 // cards removed from the current half (incl. burn)
  cutAt = 156
  cutReached = false
  halfOneFinal: { dealt: number; cutAt: number } | null = null
  unseen: number[] = new Array(52).fill(8) // per card id
  seenBuckets: number[] = new Array(10).fill(0)
  running: Record<CountSystem, number> = { hilo: 0, ko: 0, omega2: 0, wong_halves: 0, zen: 0 }
  history: CountHistoryPoint[] = []
  rigged: number[] = []
  pendingMessages: string[] = []

  // ── crowd ──
  groups: CrowdGroup[] = []

  // ── round ──
  roundId = 0
  phase: GameState['phase'] = 'betting'
  dealing = false
  balance: number
  bets: GameBets = { main: 0, perfect_pairs: 0, twenty_one_plus_three: 0, crazy_7: 0, bust_bonus: 0, insurance: 0 }
  dealer: number[] = []
  holeUp = false
  dealerPlayed = false
  hands: MHand[] = []
  active = 0
  crowdCards: number[] = []
  sideResults: SideBetResult[] = []
  c7Pending = false
  roundNet: number | null = null
  messages: string[] = []
  lastSummary: RoundSummary | null = null
  /** Bankroll agents size bets against (simulation uses a virtual, unlimited balance). */
  sizingBankroll: number | null = null

  constructor(cfg: GameConfig, seed = 7) {
    this.cfg = cfg
    this.rng = new Rng(seed)
    this.balance = cfg.bankroll.start
    this.newShoe()
  }

  // ───────────────────────── shoe ─────────────────────────

  get halfSize() {
    return (this.cfg.shoe.num_decks * 52) / 2
  }
  get totalCards() {
    return this.cfg.shoe.num_decks * 52
  }
  get cardsUnseen() {
    return this.unseen.reduce((a, b) => a + b, 0)
  }
  trueCount(sys: CountSystem) {
    return this.running[sys] / Math.max(0.25, this.cardsUnseen / 52)
  }

  newShoe() {
    const n = this.cfg.shoe.num_decks
    this.shoe = []
    for (let d = 0; d < n; d++) for (let c = 0; c < 52; c++) this.shoe.push(c)
    this.rng.shuffle(this.shoe)
    this.shoeNumber++
    this.unseen = new Array(52).fill(n)
    this.seenBuckets = new Array(10).fill(0)
    for (const s of SYSTEMS) this.running[s] = 0
    this.history = []
    this.halfOneFinal = null
    this.startHalf(1)
    const mix = this.cfg.crowd.mix
    const size = this.cfg.crowd.size
    const types: CrowdType[] = ['basic', 'cautious', 'dealer_mimic', 'reckless']
    const counts: Record<string, number> = {}
    for (let i = 0; i < size; i++) {
      let u = this.rng.next()
      let t: CrowdType = 'basic'
      for (const ty of types) {
        u -= mix[ty] ?? 0
        if (u <= 0) {
          t = ty
          break
        }
      }
      counts[t] = (counts[t] ?? 0) + 1
    }
    this.groups = types.map((t) => ({ type: t, size: counts[t] ?? 0, in: true, live: false }))
  }

  private startHalf(h: number) {
    this.half = h
    this.pos = 0
    this.cutReached = false
    const j = this.cfg.shoe.cut_jitter
    this.cutAt = Math.round(this.halfSize * this.cfg.shoe.penetration) + this.rng.int(2 * j + 1) - j
    this.pos = this.cfg.shoe.burn_cards // burn (unseen)
  }

  private betweenRounds() {
    if (!this.cutReached) return
    if (this.half === 1 && this.cfg.shoe.split_halves) {
      this.halfOneFinal = { dealt: this.pos, cutAt: this.cutAt }
      this.startHalf(2)
      this.pendingMessages.push('Cut card reached — dealing continues from the second half (count carries over)')
    } else {
      this.newShoe()
      this.pendingMessages.push(`Cut card reached — full 8-deck reshuffle (shoe #${this.shoeNumber})`)
    }
  }

  private markSeen(c: number) {
    this.unseen[c]--
    const b = bucketOf(c)
    this.seenBuckets[b]++
    for (const s of SYSTEMS) this.running[s] += COUNT_TAGS[s][b]
  }

  private draw(faceUp = true): number {
    const base = (this.half - 1) * this.halfSize
    const idx = base + this.pos
    if (this.rigged.length) {
      const want = this.rigged.shift()!
      const end = this.half === 1 && this.cfg.shoe.split_halves ? base + this.halfSize : this.totalCards
      for (let k = idx; k < end; k++) {
        if (this.shoe[k] === want) {
          this.shoe[k] = this.shoe[idx]
          this.shoe[idx] = want
          break
        }
      }
    }
    const c = this.shoe[idx]
    this.pos++
    if (this.pos >= this.cutAt && !this.cutReached) {
      this.cutReached = true
      this.messages.push('Cut card is out — this is the last round of this half')
    }
    if (faceUp) this.markSeen(c)
    return c
  }

  // ───────────────────────── crowd ─────────────────────────

  private crowdWants(type: CrowdType, cards: number[], up: number): 'hit' | 'stand' | 'double' {
    const { total, soft } = handTotal(cards)
    if (total >= 21 || cards.length >= this.cfg.rules.charlie_cards) return 'stand'
    switch (type) {
      case 'basic': {
        const a = basicAction(cards, up, cards.length === 2, false)
        return a === 'double' ? 'double' : a === 'hit' ? 'hit' : 'stand'
      }
      case 'cautious':
        if (!soft) return total >= 12 ? 'stand' : 'hit'
        return basicAction(cards, up, false, false) === 'hit' ? 'hit' : 'stand'
      case 'dealer_mimic':
        return total < 17 ? 'hit' : 'stand'
      case 'reckless':
        return total < 19 ? 'hit' : 'stand'
    }
  }

  /** Crowd groups that are still in, before a card is added to the shared `cards`. */
  private crowdBeforeCard(cards: number[]) {
    const up = this.dealer[0]
    const decisions = this.groups.map((g) => (g.in && g.size > 0 ? this.crowdWants(g.type, cards, up) : null))
    return decisions
  }

  private crowdAfterCard(decisions: ('hit' | 'stand' | 'double' | null)[], prevCards: number[], cards: number[]) {
    const prevTotal = handTotal(prevCards).total
    const total = handTotal(cards).total
    this.groups.forEach((g, i) => {
      const d = decisions[i]
      if (!d) return
      if (d === 'stand') {
        g.in = false
        g.live = prevTotal <= 21
      } else if (d === 'double') {
        g.in = false
        g.live = total <= 21
      } else if (total > 21) {
        g.in = false
        g.live = false
      }
    })
  }

  crowdCounts(): { hitters: number; standers: number } {
    if (!this.cfg.crowd.enabled) return { hitters: 0, standers: 0 }
    const size = this.groups.reduce((a, g) => a + g.size, 0)
    if (this.phase !== 'player' && this.phase !== 'crowd') return { hitters: 0, standers: size }
    const shared = this.sharedCards()
    if (!shared.length || !this.dealer.length) return { hitters: 0, standers: size }
    const d = this.crowdBeforeCard(shared)
    let hitters = 0
    this.groups.forEach((g, i) => {
      if (d[i] === 'hit' || d[i] === 'double') hitters += g.size
    })
    return { hitters, standers: size - hitters }
  }

  /** The shared (unsplit) hand as the crowd sees it. */
  private sharedCards(): number[] {
    if (this.hands.length === 2) return [this.hands[0].cards[0], this.hands[1].cards[0], ...this.crowdCards]
    return [...(this.hands[0]?.cards ?? []), ...this.crowdCards]
  }

  private crowdPlay(base: number[]) {
    if (!this.cfg.crowd.enabled) return
    this.phase = 'crowd'
    let cards = [...base]
    for (;;) {
      const t = handTotal(cards)
      if (t.total >= 21 || cards.length >= this.cfg.rules.charlie_cards) break
      const d = this.crowdBeforeCard(cards)
      let hitters = 0
      this.groups.forEach((g, i) => {
        if (d[i] === 'hit' || d[i] === 'double') hitters += g.size
      })
      if (hitters < this.cfg.crowd.min_hitters) {
        this.groups.forEach((g, i) => {
          if (d[i] === 'stand') {
            g.in = false
            g.live = true
          }
        })
        break
      }
      const c = this.draw()
      this.crowdCards.push(c)
      if (this.hands.length === 1 && cards.length === 2) this.resolveCrazy7(c)
      const next = [...cards, c]
      this.crowdAfterCard(d, cards, next)
      cards = next
      this.frame()
    }
    // whoever is still in at the end stands (21 / charlie) or busted
    const final = handTotal(cards).total
    for (const g of this.groups) {
      if (g.in) {
        g.in = false
        g.live = final <= 21
      }
    }
  }

  // ───────────────────────── round ─────────────────────────

  private frame() {
    this.onFrame()
  }

  validateBets(b: Bets): string | null {
    const L = this.cfg.limits
    if (this.phase !== 'betting' && this.phase !== 'settled') return 'Bets can only be placed between rounds'
    if (!(b.main >= L.main_min && b.main <= L.main_max)) return `Main bet must be between €${L.main_min} and €${L.main_max}`
    for (const k of ['perfect_pairs', 'twenty_one_plus_three', 'crazy_7', 'bust_bonus'] as const) {
      const v = b[k]
      if (v && (v < L.side_min || v > L.side_max)) return `Side bets must be between €${L.side_min} and €${L.side_max}`
    }
    const total = b.main + b.perfect_pairs + b.twenty_one_plus_three + b.crazy_7 + b.bust_bonus
    if (total > this.balance + 1e-9) return 'Insufficient balance'
    return null
  }

  placeBets(b: Bets) {
    const err = this.validateBets(b)
    if (err) throw new Error(err)
    this.betweenRounds()
    this.roundId++
    this.messages = [...this.pendingMessages]
    this.pendingMessages = []
    this.bets = { ...b, insurance: 0 }
    const total = b.main + b.perfect_pairs + b.twenty_one_plus_three + b.crazy_7 + b.bust_bonus
    this.balance -= total
    this.hands = [this.newHand(b.main, [])]
    this.active = 0
    this.dealer = []
    this.holeUp = false
    this.dealerPlayed = false
    this.crowdCards = []
    this.sideResults = []
    this.c7Pending = false
    this.roundNet = null
    for (const g of this.groups) {
      g.in = true
      g.live = false
    }
    this.dealing = true
    this.phase = 'player'
    this.frame()

    const h = this.hands[0]
    h.cards.push(this.draw())
    h.owners.push('player')
    this.frame()
    this.dealer.push(this.draw())
    this.frame()
    h.cards.push(this.draw())
    h.owners.push('player')
    this.frame()
    this.dealer.push(this.draw(false))
    this.frame()

    this.settleInitialSideBets()
    if (valueOf(this.dealer[0]) === 1 && this.cfg.rules.insurance) {
      this.phase = 'insurance'
      this.dealing = false
      this.messages.push('Dealer shows an Ace — insurance?')
      this.frame()
      return
    }
    this.afterInsurance()
  }

  private newHand(bet: number, cards: number[], fromSplit = false): MHand {
    return {
      cards,
      owners: cards.map(() => 'player' as Owner),
      bet,
      doubled: false,
      status: 'active',
      result: null,
      payout: 0,
      splitAces: false,
      fromSplit,
      cashedOutFor: null,
    }
  }

  private settleInitialSideBets() {
    const [p1, p2] = this.hands[0].cards
    const up = this.dealer[0]
    const sb = this.cfg.side_bets
    const push = (bet: MainSideBet, outcome: string, odds: number, label?: string) => {
      const stake = this.bets[bet]
      if (!stake) return
      const payout = odds > 0 ? stake * (odds + 1) : 0
      this.sideResults.push({ bet, outcome, odds, stake, payout })
      this.balance += payout
      if (label) this.messages.push(label)
    }
    // Perfect pairs
    if (this.bets.perfect_pairs) {
      if (rankIdx(p1) === rankIdx(p2)) {
        if (suitIdx(p1) === suitIdx(p2)) push('perfect_pairs', 'perfect_pair', sb.perfect_pairs.perfect, `Perfect Pairs — Perfect pair ${sb.perfect_pairs.perfect}:1`)
        else if (isRed(p1) === isRed(p2)) push('perfect_pairs', 'coloured_pair', sb.perfect_pairs.coloured, `Perfect Pairs — Coloured pair ${sb.perfect_pairs.coloured}:1`)
        else push('perfect_pairs', 'mixed_pair', sb.perfect_pairs.mixed, `Perfect Pairs — Mixed pair ${sb.perfect_pairs.mixed}:1`)
      } else push('perfect_pairs', 'lose', 0)
    }
    // 21+3
    if (this.bets.twenty_one_plus_three) {
      const cls = classify21p3([rankIdx(p1), rankIdx(p2), rankIdx(up)], [suitIdx(p1), suitIdx(p2), suitIdx(up)])
      if (cls) {
        const odds = sb.twenty_one_plus_three[cls]
        push('twenty_one_plus_three', cls, odds, `21+3 — ${cls.replace(/_/g, ' ')} ${odds}:1`)
      } else push('twenty_one_plus_three', 'lose', 0)
    }
    // Crazy 7 — judged on the player's own first three cards (dealer card does not count)
    if (this.bets.crazy_7) {
      const t = sb.crazy_7
      if (rankIdx(p1) !== 6) push('crazy_7', 'lose', 0)
      else if (rankIdx(p2) !== 6) push('crazy_7', 'one_seven', t.one, `Crazy 7 — first card 7 ${t.one}:1`)
      else this.c7Pending = true // two 7s: wait for the third card of the shared hand
    }
  }

  /** Resolve a pending 7-7 Crazy 7 bet with the shared hand's third card (null = no third card drawn). */
  private resolveCrazy7(third: number | null) {
    if (!this.c7Pending) return
    this.c7Pending = false
    const [p1, p2] = this.hands.length === 2 ? [this.hands[0].cards[0], this.hands[1].cards[0]] : this.hands[0].cards
    const t = this.cfg.side_bets.crazy_7
    const stake = this.bets.crazy_7
    const pairSuited = suitIdx(p1) === suitIdx(p2)
    let outcome: string
    let odds: number
    if (third !== null && rankIdx(third) === 6) {
      const suited = pairSuited && suitIdx(third) === suitIdx(p1)
      outcome = suited ? 'three_suited' : 'three_unsuited'
      odds = suited ? t.three_suited : t.three_unsuited
    } else {
      outcome = pairSuited ? 'pair_suited' : 'pair_unsuited'
      odds = pairSuited ? t.pair_suited : t.pair_unsuited
    }
    const payout = stake * (odds + 1)
    this.sideResults.push({ bet: 'crazy_7', outcome, odds, stake, payout })
    this.balance += payout
    this.messages.push(`Crazy 7 — ${outcome.replace('_', ' ')} ${odds}:1`)
  }

  /** Cash-out offer (€) for a hand: rtp × fair value under composition-independent optimal play. */
  cashoutOffer(h: MHand | undefined): number | null {
    if (!h || !this.cfg.rules.cashout || this.phase !== 'player' || this.dealing) return null
    if (h.status !== 'active' || h.doubled || h.splitAces) return null
    const t = handTotal(h.cards).total
    if (t >= 21 || this.dealer.length < 1) return null
    const canDouble = h.cards.length === 2 && (!h.fromSplit || this.cfg.rules.double_after_split)
    const canSplit = this.hands.length === 1 && h.cards.length === 2 && valueOf(h.cards[0]) === valueOf(h.cards[1])
    const ev = actionEvs(h.cards, bucketOf(this.dealer[0]), FULL_SHOE_P, canDouble, canSplit)
    const best = Math.max(ev.stand, ev.hit, ev.double ?? -Infinity, ev.split ?? -Infinity)
    const rtp = this.cfg.rules.cashout_rtp ?? 0.98
    return Math.max(0, Math.round(rtp * (1 + best) * h.bet * 100) / 100)
  }

  private dealerHasBj() {
    return this.dealer.length === 2 && handTotal(this.dealer).total === 21
  }

  private revealHole() {
    if (!this.holeUp) {
      this.holeUp = true
      this.markSeen(this.dealer[1])
    }
  }

  private afterInsurance() {
    this.dealing = true
    const up = this.dealer[0]
    const h = this.hands[0]
    const playerBj = handTotal(h.cards).total === 21
    if (valueOf(up) === 1 && this.cfg.rules.peek_on_ace) {
      if (this.dealerHasBj()) {
        this.revealHole()
        this.messages.push('Dealer peeks — Blackjack!')
        this.phase = 'dealer'
        this.frame()
        if (playerBj) h.status = 'blackjack'
        this.settle(false)
        return
      }
      this.messages.push('Dealer peeks — no Blackjack')
    }
    if (playerBj) {
      h.status = 'blackjack'
      this.messages.push('Blackjack! Everyone shares it — dealer will not draw')
      this.phase = 'dealer'
      this.frame()
      this.revealHole()
      this.frame()
      this.settle(false)
      return
    }
    this.phase = 'player'
    this.dealing = false
    this.frame()
  }

  legalActions(): LegalAction[] {
    if (this.dealing) return []
    if (this.phase === 'betting' || this.phase === 'settled') return ['deal']
    if (this.phase === 'insurance') {
      return this.balance >= this.bets.main / 2 ? ['insure', 'no_insurance'] : ['no_insurance']
    }
    if (this.phase !== 'player') return []
    const h = this.hands[this.active]
    if (!h || h.status !== 'active') return []
    const out: LegalAction[] = []
    if (!h.splitAces) out.push('hit')
    out.push('stand')
    const canDouble =
      h.cards.length === 2 && !h.splitAces && this.balance >= h.bet && (!h.fromSplit || this.cfg.rules.double_after_split)
    if (canDouble) out.push('double')
    const canSplit =
      this.hands.length === 1 &&
      h.cards.length === 2 &&
      this.balance >= h.bet &&
      (this.cfg.rules.split_by_value ? valueOf(h.cards[0]) === valueOf(h.cards[1]) : rankIdx(h.cards[0]) === rankIdx(h.cards[1]))
    if (canSplit) out.push('split')
    if (this.cashoutOffer(h) !== null) out.push('cashout')
    return out
  }

  act(a: Action) {
    const legal = this.legalActions()
    if (!legal.includes(a)) throw new Error(`Illegal action "${a}" (legal: ${legal.join(', ') || 'none'})`)
    if (this.phase === 'insurance') {
      if (a === 'insure') {
        const cost = this.bets.main / 2
        this.bets.insurance = cost
        this.balance -= cost
        this.messages.push(`Insurance taken (€${cost.toFixed(2)})`)
      }
      if (this.bets.insurance) {
        if (this.dealerHasBj()) {
          const pay = this.bets.insurance * (1 + this.cfg.rules.insurance_payout)
          this.balance += pay
          this.sideResults.push({ bet: 'insurance', outcome: 'dealer_blackjack', odds: this.cfg.rules.insurance_payout, stake: this.bets.insurance, payout: pay })
        } else {
          this.sideResults.push({ bet: 'insurance', outcome: 'lose', odds: 0, stake: this.bets.insurance, payout: 0 })
        }
      }
      this.afterInsurance()
      return
    }
    const h = this.hands[this.active]
    const offerNow = a === 'cashout' ? this.cashoutOffer(h) : null
    this.dealing = true
    const shared = this.hands.length === 1
    switch (a as PlayAction) {
      case 'hit': {
        const prev = [...h.cards]
        const d = shared && this.cfg.crowd.enabled ? this.crowdBeforeCard(prev) : null
        h.cards.push(this.draw())
        h.owners.push('player')
        if (d) this.crowdAfterCard(d, prev, h.cards)
        if (shared && h.cards.length === 3) this.resolveCrazy7(h.cards[2])
        this.checkHand(h)
        this.frame()
        break
      }
      case 'stand':
        h.status = 'stood'
        break
      case 'cashout': {
        const offer = offerNow ?? 0
        h.cashedOutFor = offer
        h.status = 'cashed_out'
        this.messages.push(`Cashed out for €${offer.toFixed(2)}`)
        break
      }
      case 'double': {
        this.balance -= h.bet
        h.bet *= 2
        h.doubled = true
        const prev = [...h.cards]
        const d = shared && this.cfg.crowd.enabled ? this.crowdBeforeCard(prev) : null
        h.cards.push(this.draw())
        h.owners.push('player')
        if (d) this.crowdAfterCard(d, prev, h.cards)
        if (shared && h.cards.length === 3) this.resolveCrazy7(h.cards[2])
        const t = handTotal(h.cards).total
        h.status = t > 21 ? 'bust' : 'stood'
        this.messages.push('Doubled — one card only')
        this.frame()
        break
      }
      case 'split': {
        this.balance -= h.bet
        const [c0, c1] = h.cards
        const aces = valueOf(c0) === 1
        const h1 = this.newHand(h.bet, [c0], true)
        const h2 = this.newHand(h.bet, [c1], true)
        h1.splitAces = h2.splitAces = aces && this.cfg.rules.split_aces_one_card
        this.hands = [h1, h2]
        this.active = 0
        this.messages.push('Split — the crowd finishes the shared hand first')
        this.frame()
        this.crowdPlay([c0, c1])
        this.phase = 'player'
        h1.cards.push(this.draw())
        h1.owners.push('player')
        this.resolveCrazy7(h1.cards[1])
        this.checkHand(h1)
        this.frame()
        if (h1.splitAces) {
          h1.status = 'stood'
          h2.cards.push(this.draw())
          h2.owners.push('player')
          h2.status = 'stood'
          this.frame()
          this.dealerPlay()
          return
        }
        break
      }
    }
    this.advance()
  }

  private checkHand(h: MHand) {
    const t = handTotal(h.cards).total
    if (t > 21) h.status = 'bust'
    else if (h.cards.length >= this.cfg.rules.charlie_cards) {
      h.status = 'charlie'
      this.messages.push('Six Card Charlie!')
    } else if (t === 21) h.status = 'stood' // 21 auto-stands
  }

  private advance() {
    const h = this.hands[this.active]
    if (h.status === 'active') {
      this.dealing = false
      this.frame()
      return
    }
    if (this.hands.length === 1) {
      if (h.status !== 'bust') this.crowdPlay(h.cards)
      else {
        for (const g of this.groups) if (g.in) g.in = false
      }
      this.dealerPlay()
      return
    }
    if (this.active === 0) {
      this.active = 1
      const h2 = this.hands[1]
      this.phase = 'player'
      h2.cards.push(this.draw())
      h2.owners.push('player')
      this.checkHand(h2)
      this.frame()
      this.advance()
      return
    }
    this.dealerPlay()
  }

  private dealerPlay() {
    this.phase = 'dealer'
    this.dealing = true
    this.revealHole()
    this.frame()
    if (this.dealerHasBj()) {
      this.messages.push('Dealer has Blackjack')
      this.settle(false)
      return
    }
    const crowdLive = this.cfg.crowd.enabled && this.groups.some((g) => g.size > 0 && g.live)
    const playerLive = this.hands.some((h) => h.status !== 'bust' && h.status !== 'cashed_out')
    if (crowdLive || playerLive) {
      const soft17 = this.cfg.rules.dealer_hits_soft_17
      for (;;) {
        const t = handTotal(this.dealer)
        if (t.total > 17 || (t.total === 17 && !(soft17 && t.soft))) break
        this.dealer.push(this.draw())
        this.frame()
      }
      this.dealerPlayed = true
    }
    this.settle(this.dealerPlayed)
  }

  private settle(dealerPlayed: boolean) {
    this.resolveCrazy7(null)
    const d = handTotal(this.dealer)
    const dealerBj = this.dealerHasBj()
    const dealerBust = d.total > 21
    let returned = 0
    let mainWagered = 0
    let mainReturned = 0
    let charlies = 0
    const playerBj = this.hands.length === 1 && this.hands[0].status === 'blackjack'
    for (const h of this.hands) {
      const t = handTotal(h.cards).total
      mainWagered += h.bet
      if (h.status === 'bust') {
        h.result = 'lose'
        h.payout = 0
      } else if (h.status === 'cashed_out') {
        h.result = 'cashout'
        h.payout = h.cashedOutFor ?? 0
      } else if (h.status === 'charlie' && (!dealerBj || this.cfg.rules.charlie_beats_dealer_bj)) {
        h.result = 'charlie'
        h.payout = h.bet * 2
        charlies++
      } else if (h.status === 'blackjack') {
        h.result = dealerBj ? 'push' : 'blackjack'
        h.payout = dealerBj ? h.bet : h.bet * (1 + this.cfg.rules.blackjack_payout)
      } else if (dealerBj) {
        h.result = 'lose'
        const original = h.doubled ? h.bet / 2 : h.bet
        h.payout = this.cfg.rules.late_dealer_bj_takes === 'all' ? 0 : h.bet - original
      } else if (dealerBust || t > d.total) {
        h.result = 'win'
        h.payout = h.bet * 2
      } else if (t === d.total) {
        h.result = 'push'
        h.payout = h.bet
      } else {
        h.result = 'lose'
        h.payout = 0
      }
      if (h.status === 'active' || h.status === 'stood') h.status = 'done'
      mainReturned += h.payout
    }
    returned += mainReturned
    this.balance += mainReturned

    // Bust bonus
    if (this.bets.bust_bonus) {
      const stake = this.bets.bust_bonus
      const pushOnBj = this.cfg.side_bets.bust_bonus_push_on_player_blackjack ?? true
      if (playerBj && pushOnBj) {
        this.sideResults.push({ bet: 'bust_bonus', outcome: 'push', odds: 0, stake, payout: stake })
        this.balance += stake
        this.messages.push('Player Blackjack — Bust Bonus bet returned')
      } else if (dealerBust && dealerPlayed) {
        const n = this.dealer.length
        const odds = this.cfg.side_bets.bust_bonus[String(Math.min(8, n))] ?? 0
        const payout = stake * (odds + 1)
        this.sideResults.push({ bet: 'bust_bonus', outcome: n >= 8 ? 'bust_8_plus' : `bust_${n}`, odds, stake, payout })
        this.balance += payout
        this.messages.push(`Dealer busts with ${n} cards — Bust Bonus ${odds}:1`)
      } else {
        this.sideResults.push({ bet: 'bust_bonus', outcome: 'lose', odds: 0, stake, payout: 0 })
      }
    } else if (dealerBust) {
      this.messages.push(`Dealer busts with ${this.dealer.length} cards`)
    }

    const sideReturned = this.sideResults.reduce((a, r) => a + r.payout, 0)
    returned += sideReturned
    const wagered =
      mainWagered + this.bets.perfect_pairs + this.bets.twenty_one_plus_three + this.bets.crazy_7 + this.bets.bust_bonus + this.bets.insurance
    this.roundNet = returned - wagered
    const mainNet = mainReturned - mainWagered
    const side: RoundSummary['side'] = {}
    for (const k of ['perfect_pairs', 'twenty_one_plus_three', 'crazy_7', 'bust_bonus', 'insurance'] as const) {
      const st = this.bets[k]
      if (!st) continue
      const pay = this.sideResults.filter((r) => r.bet === k).reduce((a, r) => a + r.payout, 0)
      side[k] = { wagered: st, net: pay - st }
    }
    this.lastSummary = {
      wagered,
      net: this.roundNet,
      outcome: mainNet > 0 ? 'win' : mainNet < 0 ? 'lose' : 'push',
      blackjack: playerBj,
      charlies,
      side,
    }
    const sign = this.roundNet > 0 ? '+' : this.roundNet < 0 ? '−' : '±'
    this.messages.push(`Round ${this.roundId}: ${sign}€${Math.abs(this.roundNet).toFixed(2)}`)
    this.phase = 'settled'
    this.dealing = false
    this.history.push({ round: this.roundId, tc: round2(this.trueCount('hilo')), rc: this.running.hilo, half: this.half })
    this.onSettle(this.lastSummary)
    this.frame()
  }

  // ───────────────────────── serialisation ─────────────────────────

  gameJson(): GameState {
    const dealerCards = this.dealer.map((c, i) => cardJson(c, 'dealer', i !== 1 || this.holeUp))
    const visible = this.dealer.filter((_, i) => i !== 1 || this.holeUp)
    const dt = handTotal(visible)
    const crowd = this.crowdCounts()
    return {
      round_id: this.roundId,
      phase: this.phase,
      balance: round2(this.balance),
      total_bet: round2(
        this.hands.reduce((a, h) => a + h.bet, 0) +
          this.bets.perfect_pairs +
          this.bets.twenty_one_plus_three +
          this.bets.crazy_7 +
          this.bets.bust_bonus +
          this.bets.insurance,
      ),
      bets: { ...this.bets },
      dealer: {
        cards: dealerCards,
        total: dt.total,
        soft: dt.soft,
        blackjack: this.holeUp && this.dealerHasBj(),
        bust: dt.total > 21,
      },
      hands: this.hands.map((h) => {
        const t = handTotal(h.cards)
        return {
          cards: h.cards.map((c, i) => cardJson(c, h.owners[i])),
          bet: h.bet,
          doubled: h.doubled,
          total: t.total,
          soft: t.soft,
          status: h.status,
          result: h.result,
          payout: h.payout,
          split_aces: h.splitAces,
          cashout_offer: this.cashoutOffer(h),
        }
      }),
      active_hand: this.active,
      crowd: {
        enabled: this.cfg.crowd.enabled,
        cards: this.crowdCards.map((c) => cardJson(c, 'crowd')),
        hitters: crowd.hitters,
        standers: crowd.standers,
      },
      legal_actions: this.legalActions(),
      side_bet_results: [...this.sideResults],
      round_net: this.roundNet === null ? null : round2(this.roundNet),
      messages: [...this.messages],
    }
  }

  bucketRecord(arr: number[]): Record<RankBucket, number> {
    const r = {} as Record<RankBucket, number>
    BUCKETS.forEach((b, i) => (r[b] = arr[i]))
    return r
  }

  unseenBuckets(): number[] {
    const out = new Array(10).fill(0)
    for (let c = 0; c < 52; c++) out[bucketOf(c)] += this.unseen[c]
    return out
  }
}

export function round2(x: number) {
  return Math.round(x * 100) / 100
}
