"""ONE Blackjack table: round state machine (SPEC §1.2-1.5, §4).

``Table`` owns a persistent :class:`~onebj.shoe.NapoleonShoe`, a bankroll and the crowd. A round:

``place_bets`` → deal (P1, dealer up, P2, dealer hole face down) → side bets PP / 21+3 / Crazy 7
settled (a 7-7 Crazy 7 waits for the player's third card) → ``INSURANCE`` (ace up) → peek → ``PLAYER``
(hit/stand/double/split/cash out) → crowd continues the shared hand → dealer → settlement (incl. Bust
Bonus) → ``SETTLED``. ``CROWD`` and ``DEALER`` are transient: they run inside :meth:`Table.act` and
are visible only through ``events`` (``{"type": "phase", ...}``).

Between rounds (right after settlement) the shoe performs the half switch / reshuffle if the cut card
came out, so observations taken in ``SETTLED`` already describe the shoe the next round is dealt from.

Money: stakes are deducted from ``balance`` when placed (bets, double, split, insurance); at settlement
everything returned (stake + winnings) is credited. ``round_result.net`` = returned - wagered.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .cards import ACE, BUCKET_OF, BUCKETS, RANK_OF, TEN, card_json
from .config import GameConfig, load_config
from .counting import counts_from_seen
from .crowd import Crowd
from .hand import Hand
from .shoe import NapoleonShoe
from .sidebets import (SIDE_BETS, crazy_7_needs_third, payout, settle_21_3, settle_bust_bonus,
                       settle_crazy_7, settle_perfect_pairs)
from .strategy.basic import get_basic_strategy
from .strategy.cashout import get_cashout_table


class Phase(str, Enum):
    BETTING = "betting"
    INSURANCE = "insurance"
    PLAYER = "player"
    CROWD = "crowd"
    DEALER = "dealer"
    SETTLED = "settled"
    SHUFFLE = "shuffle"  # only used in events


class Action(str, Enum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    CASHOUT = "cashout"
    INSURE = "insure"
    NO_INSURANCE = "no_insurance"


PLAY_ACTIONS = (Action.STAND, Action.HIT, Action.DOUBLE, Action.SPLIT, Action.CASHOUT)


@dataclass
class Bets:
    main: float = 0
    perfect_pairs: float = 0
    twenty_one_plus_three: float = 0
    crazy_7: float = 0
    bust_bonus: float = 0

    @property
    def total(self) -> float:
        return self.main + self.perfect_pairs + self.twenty_one_plus_three + self.crazy_7 + self.bust_bonus

    def side(self, name: str) -> float:
        return getattr(self, name)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Bets":
        return cls(**{k: d.get(k, 0) or 0 for k in ("main", *SIDE_BETS)})


@dataclass
class SideBetResult:
    bet: str
    outcome: str | None  # None = lost
    odds: float
    stake: float
    payout: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HandResult:
    cards: list[int]
    bet: float  # final stake (doubled stakes included)
    doubled: bool
    total: int
    result: str  # "win" | "lose" | "push" | "blackjack" | "charlie" | "cashout"
    payout: float
    split_aces: bool = False


@dataclass
class RoundResult:
    round_id: int
    hands: list[HandResult]
    side_bets: list[SideBetResult]
    insurance_stake: float
    insurance_payout: float
    dealer_cards: list[int]
    dealer_total: int
    dealer_blackjack: bool
    dealer_bust: bool
    dealer_played: bool
    shared_blackjack: bool
    main_bet: float  # original main stake
    main_wagered: float  # incl. doubles/splits
    main_returned: float
    wagered: float  # everything incl. side bets and insurance
    returned: float
    net: float
    crowd_cards: list[int] = field(default_factory=list)

    @property
    def main_net(self) -> float:
        return self.main_returned - self.main_wagered

    def side_net(self, name: str) -> float:
        return sum(r.payout - r.stake for r in self.side_bets if r.bet == name)


@dataclass
class HandView:
    cards: tuple[int, ...]
    total: int
    soft: bool
    num_cards: int
    bet: float
    doubled: bool
    is_split: bool
    split_aces: bool
    status: str
    can_double: bool
    can_split: bool
    pair: int  # pair bucket or -1
    cashout_offer: float | None  # money offered for this hand now (active hand only), else None


@dataclass
class Observation:
    """Everything a player can legally know (the unrevealed hole card is never included)."""

    round_id: int
    phase: Phase
    legal_actions: list[Action]
    hands: list[HandView]
    active_hand: int
    dealer_up: int | None
    dealer_cards: list[int]  # visible dealer cards
    dealer_total: int  # visible total
    bets: Bets
    insurance_stake: float
    bankroll: float
    seen_counts: tuple[int, ...]  # buckets A, 2..9, T seen this shoe
    unseen_counts: tuple[int, ...]  # full - seen (includes burn cards, residue and the hole card)
    unseen_card_counts: tuple[int, ...]  # per card 0..51 (suits are visible too)
    cards_unseen: int
    decks_unseen: float
    half: int
    cards_left_in_half_to_cut: int
    shoe_number: int
    running_counts: dict[str, float]
    true_counts: dict[str, float]
    crowd_cards: list[int]
    num_decks: int

    @property
    def hand(self) -> HandView | None:
        return self.hands[self.active_hand] if self.hands and self.active_hand < len(self.hands) else None


class Table:
    """A ONE Blackjack table with persistent shoe, bankroll and crowd. Deterministic given ``seed``.

    ``record_events=False`` skips event/message bookkeeping (fast simulations). ``strict=True`` enforces
    the bankroll (bets/doubles/splits/insurance must be affordable); limits are always enforced.
    """

    def __init__(self, config: GameConfig | None = None, seed: int | None = None,
                 record_events: bool = True, strict: bool = False, bankroll: float | None = None) -> None:
        self.cfg = config if config is not None else load_config()
        self.seed = seed
        self.rng = random.Random(seed)
        self.record_events = record_events
        self.strict = strict
        r = self.cfg.rules
        self._s17 = not r.dealer_hits_soft_17
        self._charlie = r.charlie_cards
        self.strategy = get_basic_strategy(r, self.cfg.shoe.num_decks)
        self.cashout_table = get_cashout_table(r) if r.cashout else None
        self.shoe = NapoleonShoe(self.cfg.shoe, self.rng)
        self.crowd = Crowd(self.cfg.crowd, self.strategy, r.charlie_cards)
        self.crowd.resample(self.rng)
        self.balance = float(self.cfg.bankroll.start if bankroll is None else bankroll)
        self.round_id = 0
        self.phase = Phase.BETTING
        self.bets = Bets()
        self.hands: list[Hand] = []
        self.active = 0
        self.dealer = Hand()
        self.hole_revealed = False
        self.peeked = False
        self.insurance_stake = 0.0
        self.insurance_decided = False
        self.splits = 0
        self.shared: Hand | None = None  # the shared (crowd) hand while it is still the player's hand 0
        self.crowd_cards: list[int] = []
        self.side_results: list[SideBetResult] = []
        self.side_return = 0.0
        self._c7_pending = False
        self.wagered = 0.0
        self.events: list[dict[str, Any]] = []
        self.messages: list[str] = []
        self.round_result: RoundResult | None = None
        self.history: list[dict[str, Any]] = []
        self._cut_announced = False

    # ================================================================== public API
    def place_bets(self, bets: Bets) -> None:
        """Start a round with ``bets`` (valid in BETTING/SETTLED): deals and advances to the first decision."""
        if self.phase is not Phase.BETTING and self.phase is not Phase.SETTLED:
            raise RuntimeError(f"cannot place bets in phase {self.phase.value}")
        self._validate_bets(bets)
        self.round_id += 1
        self.events = []
        self.messages = []
        self.round_result = None
        self.bets = bets
        self.wagered = bets.total
        self.balance -= bets.total
        self.hands = [Hand(bets.main)]
        self.active = 0
        self.dealer = Hand()
        self.hole_revealed = False
        self.peeked = False
        self.insurance_stake = 0.0
        self.insurance_decided = False
        self.splits = 0
        self.crowd_cards = []
        self.side_results = []
        self.side_return = 0.0
        self._c7_pending = False
        self.crowd.start_round()
        self.shared = self.hands[0]
        self._cut_announced = self.shoe.cut_out
        rec = self.record_events
        if rec:
            self.events.append({"type": "round_start", "round_id": self.round_id, "bets": bets.to_dict()})
        hand = self.hands[0]
        shoe = self.shoe
        c1 = shoe.deal_seen()
        hand.add(c1)
        up = shoe.deal_seen()
        self.dealer.add(up)
        c2 = shoe.deal_seen()
        hand.add(c2)
        hole = shoe.draw()
        self.dealer.add(hole)
        if rec:
            ev = self.events
            ev.append({"type": "card", "card": c1, "owner": "player", "hand": 0, "face_up": True})
            ev.append({"type": "card", "card": up, "owner": "dealer", "hand": -1, "face_up": True})
            ev.append({"type": "card", "card": c2, "owner": "player", "hand": 0, "face_up": True})
            ev.append({"type": "card", "card": hole, "owner": "dealer", "hand": -1, "face_up": False})
            self._check_cut()
        if bets.perfect_pairs or bets.twenty_one_plus_three or bets.crazy_7:
            self._settle_initial_side_bets(c1, c2, up)
        if BUCKET_OF[up] == ACE and self.cfg.rules.insurance:
            self.phase = Phase.INSURANCE
            if rec:
                self.events.append({"type": "phase", "phase": "insurance"})
            return
        self._after_insurance()

    def legal_actions(self) -> list[Action]:
        ph = self.phase
        if ph is Phase.PLAYER:
            h = self.hands[self.active]
            acts = [Action.HIT, Action.STAND]
            if self._can_double(h):
                acts.append(Action.DOUBLE)
            if self._can_split(h):
                acts.append(Action.SPLIT)
            if self.cashout_table is not None:
                acts.append(Action.CASHOUT)
            return acts
        if ph is Phase.INSURANCE:
            if self.strict and self.balance < self.bets.main / 2:
                return [Action.NO_INSURANCE]
            return [Action.INSURE, Action.NO_INSURANCE]
        return []

    def act(self, action: Action | str) -> None:
        """Apply a player decision; the crowd, dealer and settlement run automatically when you're done."""
        a = Action(action)
        if self.phase is Phase.INSURANCE:
            if a is Action.INSURE:
                if Action.INSURE not in self.legal_actions():
                    raise ValueError("insurance not affordable")
                self.insurance_stake = self.bets.main / 2
                self.balance -= self.insurance_stake
                self.wagered += self.insurance_stake
            elif a is not Action.NO_INSURANCE:
                raise ValueError(f"illegal action {a.value} in insurance phase")
            self.insurance_decided = True
            if self.record_events:
                self.events.append({"type": "insurance", "taken": a is Action.INSURE,
                                    "stake": self.insurance_stake})
            self._after_insurance()
            return
        if self.phase is not Phase.PLAYER:
            raise RuntimeError(f"no decision pending in phase {self.phase.value}")
        h = self.hands[self.active]
        if a is Action.HIT:
            self._hit(h)
        elif a is Action.STAND:
            h.status = "stood"
            self._next_hand()
        elif a is Action.DOUBLE:
            if not self._can_double(h):
                raise ValueError("double not allowed")
            self._double(h)
        elif a is Action.SPLIT:
            if not self._can_split(h):
                raise ValueError("split not allowed")
            self._split()
        elif a is Action.CASHOUT:
            if self.cashout_table is None:
                raise ValueError("cash-out is disabled")
            h.payout = self.cashout_offer()
            h.status = "cashed_out"
            if self.record_events:
                self.events.append({"type": "cashout", "hand": self.active, "amount": h.payout})
                self.messages.append(f"Cashed out for {h.payout:.2f}")
            self._next_hand()
        else:
            raise ValueError(f"illegal action {a.value} in player phase")

    def new_shoe(self) -> None:
        """Force a full reshuffle (only between rounds)."""
        if self.phase not in (Phase.BETTING, Phase.SETTLED):
            raise RuntimeError("new_shoe only between rounds")
        self.shoe.shuffle()
        self.crowd.resample(self.rng)
        self.history = []
        if self.record_events:
            self.events.append({"type": "shuffle", "kind": "reshuffle", "shoe_number": self.shoe.shoe_number})

    def reset(self, bankroll: float | None = None) -> None:
        """New bankroll (default from config) and a fresh shoe."""
        self.balance = float(self.cfg.bankroll.start if bankroll is None else bankroll)
        self.phase = Phase.BETTING
        self.round_id = 0
        self.hands, self.dealer, self.round_result = [], Hand(), None
        self.bets = Bets()
        self.events, self.messages = [], []
        self.new_shoe()

    # ------------------------------------------------------------------ decisions helpers
    def _can_double(self, h: Hand, money: bool = True) -> bool:
        r = self.cfg.rules
        if len(h.cards) != 2 or h.split_aces or (h.is_split and not r.double_after_split):
            return False
        if not r.double_any_two and not (h.aces == 0 and 9 <= h.hard <= 11):
            return False
        return not (money and self.strict) or self.balance >= h.bet

    def _can_split(self, h: Hand, money: bool = True) -> bool:
        r = self.cfg.rules
        if len(h.cards) != 2 or self.splits >= r.max_splits:
            return False
        a, b = h.cards
        if r.split_by_value:
            if BUCKET_OF[a] != BUCKET_OF[b]:
                return False
        elif RANK_OF[a] != RANK_OF[b]:
            return False
        if h.split_aces:
            return False  # no re-splitting aces
        return not (money and self.strict) or self.balance >= h.bet

    def cashout_offer(self, index: int | None = None) -> float | None:
        """Current cash-out offer (money) for hand ``index`` (default: the active hand), or None when
        unavailable (disabled, not your turn, doubled/finished hand)."""
        i = self.active if index is None else index
        if self.cashout_table is None or self.phase is not Phase.PLAYER or i != self.active:
            return None
        h = self.hands[i]
        if h.status != "active" or h.doubled:
            return None
        can_split = self._can_split(h, money=False)  # the offer does not depend on your balance
        unit = self.cashout_table.offer(h.total, h.soft, len(h.cards), BUCKET_OF[self.dealer.cards[0]],
                                        self._can_double(h, money=False), BUCKET_OF[h.cards[0]] if can_split else -1,
                                        h.is_split)
        return round(unit * h.bet, 2)

    def decision_state(self) -> tuple[int, bool, int, int, bool, int, bool, bool]:
        """Cheap view of the pending play decision for fast agents:
        (total, soft, num_cards, up_bucket, can_double, pair_bucket|-1, can_split, is_split)."""
        h = self.hands[self.active]
        can_split = self._can_split(h)
        pair = BUCKET_OF[h.cards[0]] if can_split else -1
        return (h.total, h.soft, len(h.cards), BUCKET_OF[self.dealer.cards[0]], self._can_double(h),
                pair, can_split, h.is_split)

    # ================================================================== round internals
    def _validate_bets(self, b: Bets) -> None:
        lim = self.cfg.limits
        if not lim.main_min <= b.main <= lim.main_max:
            raise ValueError(f"main bet must be within [{lim.main_min}, {lim.main_max}]")
        for name in SIDE_BETS:
            v = getattr(b, name)
            if v and not lim.side_min <= v <= lim.side_max:
                raise ValueError(f"{name} must be 0 or within [{lim.side_min}, {lim.side_max}]")
            if v < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.strict and b.total > self.balance + 1e-9:
            raise ValueError("insufficient balance")

    def _check_cut(self) -> None:
        if not self._cut_announced and self.shoe.cut_out:
            self._cut_announced = True
            if self.record_events:
                nxt = "half 2" if self.shoe.pending == "half_switch" else "a new shoe"
                self.events.append({"type": "cut_card", "half": self.shoe.half, "next": self.shoe.pending})
                self.messages.append(f"Cut card is out — this is the last round of this half; next round "
                                     f"from {nxt}")

    def _deal(self, hand: Hand, owner: str, idx: int) -> int:
        c = self.shoe.deal_seen()
        hand.add(c)
        if self.record_events:
            self.events.append({"type": "card", "card": c, "owner": owner, "hand": idx, "face_up": True})
            self._check_cut()
        return c

    def _settle_initial_side_bets(self, c1: int, c2: int, up: int) -> None:
        """Perfect Pairs and 21+3 are decided by the deal; Crazy 7 too unless the first two cards are 7s
        (then it waits for the player's third card and is settled with the round)."""
        sb, b = self.cfg.side_bets, self.bets
        if b.perfect_pairs:
            self._record_side("perfect_pairs", b.perfect_pairs, *settle_perfect_pairs(c1, c2, sb.perfect_pairs))
        if b.twenty_one_plus_three:
            self._record_side("twenty_one_plus_three", b.twenty_one_plus_three,
                              *settle_21_3(c1, c2, up, sb.twenty_one_plus_three))
        if b.crazy_7:
            if crazy_7_needs_third(c1, c2):
                self._c7_pending = True
            else:
                self._record_side("crazy_7", b.crazy_7, *settle_crazy_7(c1, c2, None, sb.crazy_7))

    def _record_side(self, name: str, stake: float, outcome: str | None, odds: float) -> None:
        pay = payout(stake, outcome, odds)
        self.side_return += pay
        res = SideBetResult(name, outcome, odds, stake, pay)
        self.side_results.append(res)
        if self.record_events:
            self.events.append({"type": "side_bet", **res.to_dict()})
            if outcome and outcome != "push":
                self.messages.append(f"{_SIDE_LABEL[name]}: {outcome.replace('_', ' ')} pays {odds:g}:1")

    def _reveal_hole(self) -> None:
        if self.hole_revealed:
            return
        self.hole_revealed = True
        hole = self.dealer.cards[1]
        self.shoe.mark_seen(hole)
        if self.record_events:
            self.events.append({"type": "reveal", "card": hole, "owner": "dealer", "hand": -1})

    def _after_insurance(self) -> None:
        r = self.cfg.rules
        upb = BUCKET_OF[self.dealer.cards[0]]
        dealer_bj = self.dealer.blackjack
        player_bj = self.hands[0].blackjack
        if player_bj:
            self.hands[0].status = "blackjack"  # pushes against a peeked dealer blackjack
        if (upb == ACE and r.peek_on_ace) or (upb == TEN and r.peek_on_ten):
            self.peeked = True
            if self.record_events:
                self.events.append({"type": "peek", "blackjack": dealer_bj})
            if dealer_bj:
                self._reveal_hole()
                if self.record_events:
                    self.messages.append("Dealer has blackjack")
                self._settle(dealer_played=True)
                return
        if player_bj:
            self._reveal_hole()
            if self.record_events:
                self.messages.append("Blackjack! Everyone shares it — the dealer does not draw")
            self._settle(dealer_played=dealer_bj)
            return
        self.phase = Phase.PLAYER
        if self.record_events:
            self.events.append({"type": "phase", "phase": "player"})

    # ---------------------------------------------------------- player actions
    def _hit(self, h: Hand) -> None:
        shared = h is self.shared
        if shared and self.crowd.enabled:
            self.crowd.before_card(h.total, h.soft, len(h.cards), BUCKET_OF[self.dealer.cards[0]])
        self._deal(h, "player", self.active)
        if shared and self.crowd.enabled:
            self.crowd.after_card(h.total, len(h.cards))
        self._after_card(h)

    def _after_card(self, h: Hand) -> None:
        """Auto-transitions after a card: bust, Charlie, 21 (auto-stand)."""
        if h.hard > 21:
            h.status = "bust"
            self._next_hand()
        elif self._charlie and len(h.cards) >= self._charlie:
            h.status = "charlie"
            if self.record_events:
                self.messages.append("Six Card Charlie!")
            self._next_hand()
        elif h.total == 21:
            h.status = "stood"
            self._next_hand()

    def _double(self, h: Hand) -> None:
        self.balance -= h.bet
        self.wagered += h.bet
        h.bet *= 2
        h.doubled = True
        if self.record_events:
            self.events.append({"type": "double", "hand": self.active})
        shared = h is self.shared
        if shared and self.crowd.enabled:
            self.crowd.before_card(h.total, h.soft, 2, BUCKET_OF[self.dealer.cards[0]])
        self._deal(h, "player", self.active)
        if shared and self.crowd.enabled:
            self.crowd.after_card(h.total, len(h.cards))
        if h.hard > 21:
            h.status = "bust"
        elif self._charlie and len(h.cards) >= self._charlie:
            h.status = "charlie"
        else:
            h.status = "stood"
        self._next_hand()

    def _split(self) -> None:
        r = self.cfg.rules
        h = self.hands[self.active]
        first_split = h is self.shared
        if first_split:
            # the crowd finishes the unsplit shared hand first
            shared = Hand(0)
            shared.add(h.cards[0])
            shared.add(h.cards[1])
            self.shared = None
            self._crowd_phase(shared)
        self.splits += 1
        self.balance -= h.bet
        self.wagered += h.bet
        c1, c2 = h.cards
        aces = BUCKET_OF[c1] == ACE
        h1 = Hand(h.bet, is_split=True)
        h1.add(c1)
        h2 = Hand(h.bet, is_split=True)
        h2.add(c2)
        h1.split_aces = h2.split_aces = aces and r.split_aces_one_card
        self.hands[self.active:self.active + 1] = [h1, h2]
        if self.record_events:
            self.events.append({"type": "split", "hand": self.active})
            self.events.append({"type": "phase", "phase": "player"})
        self._start_hand()

    def _start_hand(self) -> None:
        """Deal the second card of a split hand (if needed) and resolve automatic stands."""
        h = self.hands[self.active]
        if len(h.cards) == 1:
            self._deal(h, "player", self.active)
            if h.split_aces:
                h.status = "stood"
                self._next_hand()
                return
            if h.total == 21:
                h.status = "stood"
                self._next_hand()

    def _next_hand(self) -> None:
        self.active += 1
        if self.active < len(self.hands):
            self._start_hand()
            return
        self.active = len(self.hands) - 1
        self._finish_player()

    def _crowd_phase(self, shared: Hand) -> None:
        """The crowd keeps drawing on the shared hand while >= min_hitters players want a card."""
        crowd = self.crowd
        if not crowd.enabled:
            return
        rec = self.record_events
        if rec:
            self.events.append({"type": "phase", "phase": "crowd"})
        up = BUCKET_OF[self.dealer.cards[0]]
        need = self.cfg.crowd.min_hitters
        while crowd.any_active():
            n = len(shared.cards)
            if shared.hard > 21 or (self._charlie and n >= self._charlie):
                break
            hitters = crowd.before_card(shared.total, shared.soft, n, up)
            if hitters < need:
                break
            c = self.shoe.deal_seen()
            shared.add(c)
            self.crowd_cards.append(c)
            if rec:
                self.events.append({"type": "card", "card": c, "owner": "crowd", "hand": -1, "face_up": True,
                                    "hitters": hitters})
                self._check_cut()
            crowd.after_card(shared.total, len(shared.cards))

    def _finish_player(self) -> None:
        if self.shared is not None:
            sh = self.shared
            cont = Hand(0)
            cont.cards = sh.cards[:]
            cont.hard, cont.aces = sh.hard, sh.aces
            self.shared = None
            self._crowd_phase(cont)
        self._dealer_phase()

    def _dealer_phase(self) -> None:
        live = any(h.status == "stood" for h in self.hands)
        plays = live or (self.crowd.enabled and self.crowd.needs_dealer()) or self.bets.bust_bonus > 0
        if self.record_events:
            self.events.append({"type": "phase", "phase": "dealer"})
        self._reveal_hole()
        d = self.dealer
        if d.blackjack:
            if self.record_events:
                self.messages.append("Dealer has blackjack")
            self._settle(dealer_played=True)
            return
        if plays:
            s17 = self._s17
            shoe = self.shoe
            rec = self.record_events
            while True:
                t = d.total
                if t > 17 or (t == 17 and (s17 or not d.soft)):
                    break
                c = shoe.deal_seen()
                d.add(c)
                if rec:
                    self.events.append({"type": "card", "card": c, "owner": "dealer", "hand": -1, "face_up": True})
                    self._check_cut()
        self._settle(dealer_played=plays)

    # ---------------------------------------------------------- settlement
    def _settle(self, dealer_played: bool) -> None:
        r = self.cfg.rules
        d = self.dealer
        dealer_bj = d.blackjack
        dt = d.total
        dealer_bust = d.hard > 21
        rec = self.record_events
        main_ret = 0.0
        hand_results: list[HandResult] = []
        late_original = r.late_dealer_bj_takes == "original"
        main_bet = self.bets.main
        first_live_lost = False
        for h in self.hands:
            st = h.status
            if st == "active":  # round ended before the player acted (dealer BJ)
                st = "done"
            if st == "cashed_out":
                res, pay = "cashout", h.payout
            elif st == "blackjack":
                if dealer_bj:
                    res, pay = "push", h.bet
                else:
                    res, pay = "blackjack", h.bet * (1 + r.blackjack_payout)
            elif st == "bust":
                res, pay = "lose", 0.0
            elif st == "charlie":
                if dealer_bj and not r.charlie_beats_dealer_bj:
                    res, pay = "lose", 0.0
                else:
                    res, pay = "charlie", 2 * h.bet
            elif dealer_bj:
                res, pay = "lose", 0.0
                if late_original and not self.peeked and len(h.cards) >= 2 and st != "done":
                    # only the original stake is lost: refund doubles and additional split hands
                    if not first_live_lost:
                        pay = h.bet - main_bet
                        first_live_lost = True
                    else:
                        res, pay = "push", h.bet
            else:
                t = h.total
                if dealer_bust or t > dt:
                    res, pay = "win", 2 * h.bet
                elif t == dt:
                    res, pay = "push", h.bet
                else:
                    res, pay = "lose", 0.0
            if st == "done" or st == "active":
                h.status = "done"
            h.result = res
            h.payout = pay
            main_ret += pay
            hand_results.append(HandResult(h.cards[:], h.bet, h.doubled, h.total, res, pay, h.split_aces))
        ins_pay = 0.0
        if self.insurance_stake:
            if dealer_bj:
                ins_pay = self.insurance_stake * (1 + r.insurance_payout)
            if rec:
                self.events.append({"type": "insurance_result", "won": dealer_bj, "payout": ins_pay})
                if dealer_bj:
                    self.messages.append(f"Insurance pays {r.insurance_payout:g}:1")
        shared_bj = bool(self.hands) and self.hands[0].status == "blackjack"
        sb = self.cfg.side_bets
        if self._c7_pending:
            self._c7_pending = False
            c3 = self.third_card()
            hand0 = self.round_first_cards
            outcome, odds = settle_crazy_7(hand0[0], hand0[1], c3, sb.crazy_7)
            self._record_side("crazy_7", self.bets.crazy_7, outcome, odds)
        if self.bets.bust_bonus:
            outcome, odds = settle_bust_bonus(len(d.cards), dealer_bust, shared_bj, sb.bust_bonus,
                                              sb.bust_bonus_push_on_player_blackjack)
            self._record_side("bust_bonus", self.bets.bust_bonus, outcome, odds)
        returned = main_ret + ins_pay + self.side_return
        self.balance += returned
        main_wagered = sum(h.bet for h in self.hands)
        self.round_result = RoundResult(
            self.round_id, hand_results, self.side_results, self.insurance_stake, ins_pay, d.cards[:], dt,
            dealer_bj, dealer_bust, dealer_played and not dealer_bj, shared_bj, main_bet, main_wagered,
            main_ret, self.wagered, returned, returned - self.wagered, self.crowd_cards[:])
        self.phase = Phase.SETTLED
        if rec:
            if dealer_bust:
                self.messages.append(f"Dealer busts with {len(d.cards)} cards")
            self.events.append({"type": "settle", "net": returned - self.wagered, "returned": returned,
                                "hands": [{"result": hr.result, "payout": hr.payout} for hr in hand_results]})
            self.events.append({"type": "phase", "phase": "settled"})
        self._end_round()

    @property
    def round_first_cards(self) -> tuple[int, int]:
        """The player's two initial cards of this round (also after a split)."""
        if self.splits:
            return self.hands[0].cards[0], self.hands[1].cards[0]
        c = self.hands[0].cards
        return c[0], c[1]

    def third_card(self) -> int | None:
        """The player's third card for Crazy 7: after splitting the initial pair, the next card of split
        hand 1; otherwise the 3rd card of the shared hand (own hit/double, else the crowd's first hit)."""
        if self.splits:
            c = self.hands[0].cards
            return c[1] if len(c) > 1 else None
        c = self.hands[0].cards
        if len(c) >= 3:
            return c[2]
        return self.crowd_cards[0] if self.crowd_cards else None

    def _end_round(self) -> None:
        shoe = self.shoe
        if self.record_events:
            rc, tc = counts_from_seen(shoe.seen_bucket, shoe.num_decks)
            self.history.append({"round": self.round_id, "rc": rc["hilo"], "tc": round(tc["hilo"], 3),
                                 "half": shoe.half})
        kind = shoe.end_round()
        if kind:
            if kind == "reshuffle":
                self.crowd.resample(self.rng)
                self.history = []
            if self.record_events:
                self.events.append({"type": "shuffle", "kind": kind, "shoe_number": shoe.shoe_number,
                                    "half": shoe.half})
                self.messages.append("Switching to the second half of the shoe" if kind == "half_switch"
                                     else "Shuffling a new 8-deck shoe")

    # ================================================================== views
    def observation(self) -> Observation:
        """Legal information only: the hole card stays hidden until revealed."""
        shoe = self.shoe
        hands = []
        playing = self.phase is Phase.PLAYER
        for i, h in enumerate(self.hands):
            cd = playing and i == self.active and self._can_double(h)
            cs = playing and i == self.active and self._can_split(h)
            hands.append(HandView(tuple(h.cards), h.total, h.soft, len(h.cards), h.bet, h.doubled,
                                  h.is_split, h.split_aces, h.status, cd, cs,
                                  BUCKET_OF[h.cards[0]] if cs else -1, self.cashout_offer(i)))
        dcards = self.dealer.cards if self.hole_revealed else self.dealer.cards[:1]
        if self.hole_revealed:
            dtotal = self.dealer.total
        elif dcards:
            dtotal = 11 if BUCKET_OF[dcards[0]] == ACE else BUCKET_OF[dcards[0]] + 1
        else:
            dtotal = 0
        running, true = counts_from_seen(shoe.seen_bucket, shoe.num_decks)
        unseen = shoe.cards_unseen
        return Observation(
            round_id=self.round_id, phase=self.phase, legal_actions=self.legal_actions(), hands=hands,
            active_hand=self.active, dealer_up=self.dealer.cards[0] if self.dealer.cards else None,
            dealer_cards=list(dcards), dealer_total=dtotal, bets=self.bets,
            insurance_stake=self.insurance_stake, bankroll=self.balance,
            seen_counts=tuple(shoe.seen_bucket), unseen_counts=tuple(shoe.unseen_buckets()),
            unseen_card_counts=tuple(shoe.unseen_cards()), cards_unseen=unseen, decks_unseen=unseen / 52,
            half=shoe.half, cards_left_in_half_to_cut=shoe.cards_left_to_cut, shoe_number=shoe.shoe_number,
            running_counts=running, true_counts=true, crowd_cards=self.crowd_cards[:],
            num_decks=shoe.num_decks)

    def snapshot(self, analytics: bool = True) -> dict[str, Any]:
        """JSON-able ``{"game": ..., "shoe": ...}`` exactly as in SPEC §6."""
        return {"game": self._game_json(), "shoe": self._shoe_json(analytics)}

    def _game_json(self) -> dict[str, Any]:
        d = self.dealer
        if self.hole_revealed:
            dcards = [card_json(c, "dealer") for c in d.cards]
            dtot, dsoft = d.total, d.soft
        else:
            dcards = [card_json(c, "dealer", i != 1) for i, c in enumerate(d.cards)]
            vis = d.cards[:1]
            dtot = (11 if BUCKET_OF[vis[0]] == ACE else BUCKET_OF[vis[0]] + 1) if vis else 0
            dsoft = bool(vis) and BUCKET_OF[vis[0]] == ACE
        hands = []
        for i, h in enumerate(self.hands):
            hands.append({"cards": [card_json(c, "player") for c in h.cards], "bet": h.bet, "doubled": h.doubled,
                          "total": h.total, "soft": h.soft, "status": h.status, "result": h.result,
                          "payout": h.payout if h.result is not None else 0, "split_aces": h.split_aces,
                          "cashout_offer": self.cashout_offer(i)})
        if self.phase in (Phase.BETTING, Phase.SETTLED):
            legal = ["deal"]
        else:
            legal = [a.value for a in self.legal_actions()]
        hitters = self.crowd.hitters if self.phase not in (Phase.BETTING,) and self.crowd.enabled else 0
        b = self.bets.to_dict()
        b["insurance"] = self.insurance_stake
        rr = self.round_result
        return {
            "round_id": self.round_id, "phase": self.phase.value, "balance": round(self.balance, 2),
            "total_bet": self.wagered if self.round_id else 0, "bets": b,
            "dealer": {"cards": dcards, "total": dtot, "soft": dsoft,
                       "blackjack": self.hole_revealed and d.blackjack, "bust": d.hard > 21},
            "hands": hands, "active_hand": self.active,
            "crowd": {"enabled": self.crowd.enabled, "cards": [card_json(c, "crowd") for c in self.crowd_cards],
                      "hitters": hitters, "standers": self.crowd.size - hitters},
            "legal_actions": legal,
            "side_bet_results": [{"bet": s.bet, "outcome": s.outcome, "odds": s.odds, "stake": s.stake,
                                  "payout": s.payout} for s in self.side_results],
            "round_net": None if rr is None else round(rr.net, 2),
            "messages": list(self.messages),
        }

    def _shoe_json(self, with_analytics: bool) -> dict[str, Any]:
        shoe = self.shoe
        running, true = counts_from_seen(shoe.seen_bucket, shoe.num_decks)
        unseen_b = shoe.unseen_buckets()
        out: dict[str, Any] = {
            "shoe_number": shoe.shoe_number, "half": shoe.half, "half_size": shoe.half_size,
            "dealt_in_half": shoe.dealt_in_half, "cut_at": shoe.cut_at, "burned": shoe.burned,
            "cards_unseen": shoe.cards_unseen, "decks_unseen": round(shoe.cards_unseen / 52, 3),
            "seen_counts": dict(zip(BUCKETS, shoe.seen_bucket)), "unseen_counts": dict(zip(BUCKETS, unseen_b)),
            "running": running, "true": {k: round(v, 3) for k, v in true.items()},
            "cut_card_out": shoe.cut_out, "pending_shuffle": shoe.pending,
            "history": list(self.history),
        }
        if with_analytics:
            from .analytics import table_analytics
            out["analytics"] = table_analytics(self)
        return out

    # ================================================================== cloning
    def clone(self, resample_unseen: bool = True, seed: int | None = None) -> "Table":
        """Deep copy. With ``resample_unseen`` every card the player hasn't seen (remaining shoe, both
        halves, burn/residue and an unrevealed hole card) is reshuffled, so rollouts cannot peek. A
        peeked hole card keeps its 'no blackjack' constraint. ``seed`` seeds the clone's RNG."""
        t = Table.__new__(Table)
        t.__dict__.update(self.__dict__)
        t.rng = random.Random(seed if seed is not None else self.rng.getrandbits(64))
        t.shoe = self.shoe.copy(t.rng)
        t.crowd = self.crowd.copy()
        t.hands = [h.copy() for h in self.hands]
        if self.shared is not None:
            t.shared = t.hands[self.hands.index(self.shared)]
        t.dealer = self.dealer.copy()
        t.bets = Bets(**asdict(self.bets))
        t.crowd_cards = self.crowd_cards[:]
        t.side_results = list(self.side_results)
        t.events = list(self.events)
        t.messages = list(self.messages)
        t.history = list(self.history)
        if resample_unseen:
            hidden = len(t.dealer.cards) >= 2 and not self.hole_revealed
            ok: Any = None
            if hidden and self.peeked:
                upb = BUCKET_OF[t.dealer.cards[0]]
                bad = TEN if upb == ACE else (ACE if upb == TEN else -1)

                def ok(c: int) -> bool:
                    return BUCKET_OF[c] != bad
            new = t.shoe.resample_unseen(t.rng, 1 if hidden else 0, ok)
            if hidden:
                up = t.dealer.cards[0]
                t.dealer = Hand()
                t.dealer.add(up)
                t.dealer.add(new[0])
        return t


_SIDE_LABEL = {"perfect_pairs": "Perfect Pairs", "twenty_one_plus_three": "21+3", "crazy_7": "Crazy 7",
               "bust_bonus": "Bust Bonus"}
