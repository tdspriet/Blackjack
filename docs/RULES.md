# ONE Blackjack — rules as implemented & calibration

This document describes exactly what `onebj` (the Python engine in `backend/onebj`) implements, how the
uncertain rule variants were resolved, and the evidence. Primary source: the official in-game help
(`docs/OFFICIAL_RULES_NL.txt`, Pragmatic Play client 1.66.401 at Napoleon Casino); secondary sources and
screenshots are listed in `docs/SPEC.md` §1. All numbers below were produced with this code base
(`uv run pytest`, `onebj.sim.simulate`, `onebj.strategy.solver`, `onebj.analytics`).

## 1. Rules as implemented (defaults in `config/default.json`)

### Shoe
* 8 standard decks (416 cards). Default model ("Napoleon procedure", `shoe.split_halves: true`): the 8 decks are
  shuffled together and split into two 4-deck halves of 208 cards. Each half: 1 unseen burn card, cut card after
  `round(0.75 × 208) = 156` cards ± uniform 8 (clamped so ≥ 40 cards stay behind it).
* When the cut card comes out the round is completed ("last hand"), then the next round is dealt from half 2;
  after half 2's cut card the whole 8-deck shoe is reshuffled. The official help only says "the shoe is shuffled
  once the cut card is out and the round finished"; `split_halves: false` gives that plain single 8-deck shoe with
  one cut card at 75 % (312 cards ± 8).
* Seen/unseen composition is tracked over the whole 8-deck shoe and only reset by a reshuffle (never at the half
  switch). Unseen = remaining cards of both halves + burn cards + half-1 residue + an unrevealed hole card.

### Deal, peek, insurance
* Deal order: player card 1, dealer up card, player card 2, dealer hole card (face down). Everyone shares the hand.
* Insurance is offered with an ace up (half the main bet, pays 2:1; insurance on a blackjack = even money). The
  dealer then peeks (ace up only). Dealer blackjack ends the round: every hand loses except a player blackjack,
  which pushes. With a 10-value up card there is **no peek**; a dealer blackjack is found when the hole card is
  turned after all decisions.
* Late dealer blackjack (10 up): `late_dealer_bj_takes: "all"` (default — see §3.2): every live hand loses its
  full stake, including doubles and the second split hand. Busted hands lose anyway; **Six Card Charlie hands still
  win**; cashed-out hands keep their cash-out. (`"original"`: only the original stake is lost.)
* Player blackjack (the shared initial hand) ends the round: the hole card is turned, the dealer does not draw;
  3:2, or push against a dealer blackjack.

### Player options
* Hit, Stand, Double (any first two cards, one card), Split (once, any two cards of equal value — K-Q splits),
  Cash out. No surrender, no re-split. Double after split allowed except after split aces; split aces get exactly
  one card each; A+10 after a split is 21, not blackjack. A hand reaching 21 is stood automatically (official:
  "you can draw until you stand or your hand reaches 21").
* **Six Card Charlie**: a non-busted 6-card hand is stood automatically and wins 1:1, even against a dealer
  blackjack.
* Dealer draws to 16 and stands on all 17s (S17). Blackjack 3:2, win 1:1, push returns the stake.
* **Cash out ("Uitbetalen")**, main bet only: offer = `cashout_rtp (0.98) × fair value × hand bet`, where fair
  value = expected return (stake included) per unit of that hand's bet under optimal play, vs that up card,
  for a full fresh shoe (composition independent; infinite-deck probabilities), including Six Card Charlie, no
  peek on tens/late-BJ rule, "no dealer BJ" after an ace peek, DAS and the one-split rule
  (`onebj/strategy/cashout.py`). Available for the active hand in the player phase; not after doubling or busting;
  each split hand gets its own offer. Taking it ends the hand; the amount is paid at settlement.
* Limits: main €1–€5000, side bets €1–€500 each (require a main bet).

### Side bets (odds X:1, stake returned on a win; only the highest combination pays)
| Bet | Cards | Paytable |
|---|---|---|
| Perfect Pairs | player's first two cards | perfect pair (same suit) 25, coloured pair (same colour) 12, mixed pair 6 |
| 21+3 | player's two cards + dealer up card | suited trips 100, straight flush 40, three of a kind 30, straight 10 (A-2-3 and Q-K-A, no K-A-2), flush 5 |
| Crazy 7 | player's first **three** cards | 7 as first card 5; first two 7s 25 / same suit 150; first three 7s 500 / all same suit 2000 |
| Bust Bonus | dealer's final hand | dealer busts with 3 cards 1, 4: 2, 5: 9, 6: 50, 7: 100, 8+: 250; **stake returned if the player has a blackjack** |

* Crazy 7 third card = the 3rd card of the shared hand (your own hit or double, or the crowd's hit if you stood
  on two cards), or — if you split 7s — the next card dealt to split hand 1. If the dealer peeks a blackjack no
  third card is drawn. A 7-7 start is settled at the end of the round; everything else at the deal.

### Crowd (shared hand)
250 virtual players share the hand (types sampled per shoe: basic 75 %, cautious 12 % = the official timeout
default "stand on 12+, hit on 11 or less", dealer-mimic 10 % = hit below 17, reckless 3 % = hit below 19). The
shared hand keeps receiving cards while at least one player wants one; cards after your stand are crowd cards
(visible, counted). If you split, the crowd first finishes the unsplit hand. The dealer plays out unless every
hand at the table is resolved (with the crowd: practically always, unless the shared hand is a blackjack);
without a crowd the dealer also plays out when you have a Bust Bonus bet.

## 2. Basic strategy for these rules

Derived with the solver (`onebj/strategy/solver.py`): 2-card decisions composition-weighted for 8 decks (both
player cards and the up card removed), multi-card decisions from the infinite-deck solver with the number of
cards in the state (Six Card Charlie). Test `test_standard_chart_reproduced` checks that the same machinery
reproduces the textbook 8-deck S17 DAS chart for standard rules. Differences from that chart: **11 v T hit** and
**8,8 v T hit** (a late 10-up blackjack takes doubled/split stakes), **A,4 v 4 hit**, plus the Charlie rows
below. (A,2 v 5 is a coin flip in 8 decks — EV difference < 0.001 — and comes out "hit".)

```
2-card hands
       2  3  4  5  6  7  8  9  T  A
H5     H  H  H  H  H  H  H  H  H  H
H6     H  H  H  H  H  H  H  H  H  H
H7     H  H  H  H  H  H  H  H  H  H
H8     H  H  H  H  H  H  H  H  H  H
H9     H  D  D  D  D  H  H  H  H  H
H10    D  D  D  D  D  D  D  D  H  H
H11    D  D  D  D  D  D  D  D  H  H
H12    H  H  S  S  S  H  H  H  H  H
H13    S  S  S  S  S  H  H  H  H  H
H14    S  S  S  S  S  H  H  H  H  H
H15    S  S  S  S  S  H  H  H  H  H
H16    S  S  S  S  S  H  H  H  H  H
H17    S  S  S  S  S  S  S  S  S  S
H18    S  S  S  S  S  S  S  S  S  S
H19    S  S  S  S  S  S  S  S  S  S
H20    S  S  S  S  S  S  S  S  S  S
H21    S  S  S  S  S  S  S  S  S  S
S13    H  H  H  H  D  H  H  H  H  H
S14    H  H  H  D  D  H  H  H  H  H
S15    H  H  H  D  D  H  H  H  H  H
S16    H  H  D  D  D  H  H  H  H  H
S17    H  D  D  D  D  H  H  H  H  H
S18    S Ds Ds Ds Ds  S  S  H  H  H
S19    S  S  S  S  S  S  S  S  S  S
S20    S  S  S  S  S  S  S  S  S  S
S21    S  S  S  S  S  S  S  S  S  S
P22    P  P  P  P  P  P  -  -  -  -
P33    P  P  P  P  P  P  -  -  -  -
P44    -  -  -  P  P  -  -  -  -  -
P55    -  -  -  -  -  -  -  -  -  -
P66    P  P  P  P  P  -  -  -  -  -
P77    P  P  P  P  P  P  -  -  -  -
P88    P  P  P  P  P  P  P  P  -  P
P99    P  P  P  P  P  -  P  P  -  -
PTT    -  -  -  -  -  -  -  -  -  -
PAA    P  P  P  P  P  P  P  P  P  P
```

```
4-card hands (only rows that differ from the 2-card chart without doubling)
       2  3  4  5  6  7  8  9  T  A
H12    H  H  H  H  H  H  H  H  H  H
H13    H  H  S  S  S  H  H  H  H  H
S18    H  H  H  H  H  S  H  H  H  H
S19    S  S  S  S  S  S  S  S  H  S
5-card hands (only rows that differ from the 2-card chart without doubling)
       2  3  4  5  6  7  8  9  T  A
H12    H  H  H  H  H  H  H  H  H  H
H13    H  H  H  H  H  H  H  H  H  H
H14    H  H  H  H  H  H  H  H  H  H
H15    H  H  H  H  H  H  H  H  H  H
H16    H  H  S  S  S  H  H  H  H  H
H17    S  S  S  S  S  S  S  H  H  H
S18    H  H  H  H  H  H  H  H  H  H
S19    H  H  H  H  H  H  H  H  H  H
S20    H  H  H  H  H  H  H  H  H  H
```
3-card hands: identical to the 2-card chart without doubling.

**Hi-Lo index plays** (`ILLUSTRIOUS_18` in `strategy/basic.py`), recomputed with `solver.hilo_index` (composition
shifted along the Hi-Lo direction; the method reproduces the published Illustrious 18 within ±0.6 for standard
rules): insurance ≥ +3; 16vT stand ≥ 0; 15vT stand ≥ +4; TTv5 split ≥ +5; TTv6 split ≥ +4; **11vT double ≥ +3**
(new: basic is hit here); 12v3 stand ≥ +1; 12v2 stand ≥ +3; 11vA double ≥ +2; 9v2 double ≥ +1; 10vA double ≥ +4;
9v7 double ≥ +3; 16v9 stand ≥ +5; 13v2 hit < −1; 12v4 hit < 0; 12v5 hit < −2; 12v6 hit < −1; 13v3 hit < −2.
"10vT double" (I18 +4) never becomes correct under the late-BJ rule (solver: no crossing up to +12) and is dropped.

## 3. Calibration

### 3.1 Side bets at a fresh 8-deck shoe
Exact values (combinatorics without replacement from a full 416-card shoe; `onebj.analytics` and the exact
Bust Bonus enumeration in §3.1.2) and simulation (50 M rounds per row, `basic_sides` agent: basic strategy +
€1 on every side bet every round).

| Bet | Official | Exact (fresh 8-deck shoe) | Analytics estimate (fresh shoe) |
|---|---|---|---|
| Perfect Pairs | 95.90 % | **95.904 %** | 95.904 % (exact) |
| 21+3 | 96.30 % | **96.296 %** | 96.296 % (exact) |
| Crazy 7 | 94.77 % | **94.772 %** (3rd card drawn unless the dealer peeks a BJ) | 94.772 % |
| Bust Bonus | 94.12 % | **94.117 %** (250:1 for 8+, stake returned on player BJ) | 94.109 % |

#### 3.1.1 Crazy 7 — which cards count
| Definition | RTP |
|---|---|
| **Official: player's first three cards; no third card after a peeked dealer BJ** | **94.772 %** |
| Player's first three cards, third card always drawn | 95.288 % |
| Old spec guess: player 2 cards + dealer up card | 95.288 % (same as above: the 3rd card is just a random card) |
| Never a third card (only the 1- and 2-card tiers) | 73.86 % |
| Official definition with 6 decks | 92.97 % |

The official 94.77 % is reproduced to the second decimal only if the 7-7-7 tiers are impossible when the dealer
peeks a blackjack (ace up + ten in the hole), exactly as the help states — which also implies that a third card
is (practically) always drawn otherwise, i.e. the crowd's hit counts as the third card (§5 item 4).

#### 3.1.2 Bust Bonus variants (exact enumeration, 8 decks without replacement; player's 2 cards removed)
| 8+ cards pays | Player BJ → stake returned | Player BJ → bet lost | Dealer always plays out |
|---|---|---|---|
| **250:1** | **94.117 %** | 89.371 % | 93.823 % |
| 200:1 | 94.074 % | 89.329 % | 93.779 % |

Official 94.12 % = 250:1 with the stake returned on a player blackjack (both confirmed by the help). The
third-party 93.82 % figure is the same paytable with the dealer playing out regardless (no push).

#### 3.1.3 Simulation (basic strategy + all side bets)

| Crowd | Shoe | Perfect Pairs | 21+3 | Crazy 7 | Bust Bonus | Main game |
|---|---|---|---|---|---|---|
| on | two halves | 95.91 ± 0.06 | 96.32 ± 0.06 | 94.87 ± 0.20 | 94.21 ± 0.05 | 99.569 ± 0.016 |
| on | single 8-deck | 95.89 ± 0.06 | 96.27 ± 0.06 | 94.83 ± 0.20 | 94.19 ± 0.05 | 99.568 ± 0.016 |
| off | two halves | 95.93 ± 0.06 | 96.28 ± 0.06 | 94.68 ± 0.20 | 94.17 ± 0.05 | 99.564 ± 0.016 |
| off | single 8-deck | 95.93 ± 0.06 | 96.25 ± 0.06 | 95.20 ± 0.20 | 94.16 ± 0.05 | 99.574 ± 0.016 |

All side bets agree with the exact values within noise, with the crowd on or off and for both shoe procedures,
except a small but significant Bust Bonus excess in play: 94.18 % ± 0.02 (average of the four rows) vs 94.12 %
exact-for-the-deal. The exact figure (like the official one) ignores the cards players draw before the dealer;
in play hitters keep drawing while small cards come and stop on big ones, so the dealer draws from a slightly
ten-richer remainder and busts a bit more often. Crazy 7 is the same with or without the crowd for a basic-strategy
player because 7-7 is always split or hit (a third card is always drawn unless the dealer peeks a blackjack); a
player who stands on 7-7 without a crowd would get only the two-card tiers.

Main game over these 4 × 50 M rounds (different seeds from §3.2): 99.569 % ± 0.008; together with §3.2's default
row: **99.571 % ± 0.008 for flat basic strategy**. The official 99.59 % is labelled *optimal*; a composition-
dependent optimal strategy typically adds 0.02–0.04 % in an 8-deck shoe, consistent with our basic-strategy
figure. Every alternative rule set is ≥ 0.06 % higher still and would need an *even lower* optimal-vs-basic gap.

### 3.2 Main game (basic strategy, flat bet, RTP on the initial bet)
20 M rounds per variant (10 processes, common random numbers: seed 12345 for every variant, so differences
between rows are more precise than the per-row standard error), crowd on unless stated, `basic` agent
(basic strategy derived for each variant's rules, flat bet, never insures, never cashes out). "Solver" = exact
infinite-deck optimal total-dependent RTP for the same rules. Official target: **99.59 %**.

| Variant | Sim RTP (± 1 SE) | Solver (∞ deck) | Sim − solver | Charlie hands / round |
|---|---|---|---|---|
| default (no peek T, late all, charlie, DAS) | 99.591 % ± 0.025 | 99.511 % | +0.080 | 0.263 % |
| peek_on_ten | 99.654 % ± 0.026 | 99.591 % | +0.063 | 0.248 % |
| late original | 99.674 % ± 0.026 | 99.609 % | +0.065 | 0.258 % |
| no charlie | 99.420 % ± 0.025 | 99.326 % | +0.094 | 0.000 % |
| no charlie + peek_on_ten | 99.506 % ± 0.026 | 99.430 % | +0.076 | 0.000 % |
| no charlie + late original | 99.507 % ± 0.026 | 99.427 % | +0.080 | 0.000 % |
| no DAS | 99.476 % ± 0.025 | 99.395 % | +0.082 | 0.266 % |
| no DAS + no charlie | 99.311 % ± 0.025 | 99.208 % | +0.103 | 0.000 % |
| default, crowd off | 99.584 % ± 0.025 | 99.511 % | +0.073 | 0.263 % |

**Chosen: no peek on tens, late dealer blackjack takes all (doubles and splits), Six Card Charlie (also beats a
late dealer blackjack), DAS** — the official configuration (help: DAS allowed, peek only with an ace, 6-card
Charlie) lands at **99.591 % ± 0.025 %**, i.e. exactly the published 99.59 %. The alternatives miss it by 2.5–3.3
standard errors (and by more in the common-random-numbers comparison): late BJ "original only" 99.674 %, peek on
tens 99.654 %. Variants without Charlie or without DAS (all ≤ 99.51 %) contradict the official rules anyway.
(The third-party figure 99.28 % quoted earlier corresponds to an older rule set: no DAS, no Charlie credit;
our "no DAS + no Charlie" row gives 99.31 %.)

Other observations:
* The crowd does not change the main-game EV (99.591 % vs 99.584 % — different card sequences, within noise):
  crowd cards are random cards drawn between your hand and the dealer's.
* Six Card Charlie is worth +0.17 % with the Charlie-aware strategy; 0.26 % of rounds end in a Charlie hand.
* An 8-deck shoe dealt to 75 % beats the infinite deck by a consistent +0.07–0.10 %; `analytics.FINITE_DECK_CORRECTION
  = +0.0008` adds this to the solver base of the main-edge estimator.
* Basic strategy never insures; the HiLo agent insures at TC ≥ +3 (solver: break-even at +3.4 for an infinite
  deck shifted along the Hi-Lo direction).

Baseline counter (`HiLoAgent` without side bets: basic + the indices of §2, insurance at TC ≥ +3, 1–12 unit
spread by floor(TC): ≥2→2, ≥3→4, ≥4→6, ≥5→8, ≥6→12), 20 M rounds: **+0.0035 units per round** (± 0.0005;
+0.22 % of the amount wagered) with the two-halves shoe, +0.0040 with a plain 8-deck shoe at 75 %.

### 3.3 Cash-out fair values
Offer = 0.98 × fair value by construction, so the option's RTP is 98.00 % relative to optimal continuation.
Validation of the fair values against the engine: at the first decision of 9.3 M rounds (basic strategy, crowd
on) the mean fair value was 0.94559 per unit and the realised return of playing on was 0.94641 per unit
(+0.0008 ± 0.0003 — the same 8-deck-vs-infinite-deck bonus seen in §3.2); per up card all within ±0.0035.
Repeated with `late_dealer_bj_takes: "original"`: +0.0006 ± 0.0004. Since offer ≤ fair value for every state
(test `test_cashout_never_beats_fresh_shoe_value`), the baseline agents never cash out; a counter (or the NN)
can profit from cash-outs only when the live composition makes the hand worse than its fresh-shoe value.
Sample offers per €1: 16 v T (2 cards) €0.42, 12 v 6 €0.83, T-T v 9 €1.72.

## 4. Analytics model (`onebj/analytics.py`)
* Perfect Pairs and 21+3 EVs are exact for the unseen composition (per card: rank and suit). Crazy 7 is exact
  under "a third card is drawn unless the dealer peeks a blackjack" (true for basic strategy / with the crowd).
* Bust Bonus: exact dealer bust-by-card-count distribution drawing without replacement from the unseen pool,
  times P(no shared blackjack), plus the returned stake on a player blackjack (fresh shoe: 94.109 % vs exact
  94.117 %). Dealer outcome distribution per up card: S17 recursion on the current composition.
* Insurance EV: `3·P(ten) − 1` with P(ten) = unseen tens / unseen cards.
* Main-game edge estimate: `base + FINITE_DECK_CORRECTION + Σ EoR_r × (expected_r − unseen_r) / decks_unseen`.
  EoRs computed by the solver for the configured rules (removing one card of a rank from a 52-card deck; by the
  envelope theorem equal to the fixed-basic-strategy EoR to first order). Default rules (EV change in %, per card
  removed): A -0.64, 2 +0.32, 3 +0.42, 4 +0.59, 5 +0.78, 6 +0.46, 7 +0.29, 8 +0.00, 9 -0.17, T -0.44. Compare Griffin's classic values (A −0.61, 2 +0.38, 3 +0.44, 4 +0.55, 5 +0.69, 6 +0.46,
  7 +0.28, 8 0.00, 9 −0.18, T −0.51): the late-BJ and Charlie rules shift them slightly.
* `table_analytics()` takes ≈ 2 ms on a new composition (memoised exact Bust Bonus recursion ≈ 1 ms; results cached per composition).

## 5. Please verify in the in-game help / at the table
1. **10-up dealer blackjack vs doubles/splits** — the help is silent. Simulation picks "takes all"
   (99.59 % exactly as published); "original only" would give ≈ 99.67 %. Verify on a hand where the dealer shows a
   ten, you double/split and the dealer turns a blackjack.
2. **Is there really no peek with a ten up?** The help only describes the ace peek; "peek on ten" would give
   ≈ 99.65 %. Watch for a 10-up round that ends immediately on a dealer blackjack.
3. **Cash-out formula** — we model `0.98 × fresh-shoe fair value`. Compare a few real offers (e.g. hard 16 v T on
   two cards: our offer = €0.42 per €1; 12 v 6 two cards: €0.83; 20 v 9: €1.72 (pair of tens)) with the button amount.
   Also check whether the offer appears before insurance is resolved and for not-yet-played split hands (we show it
   for the active hand only).
4. **Crazy 7 third card for a player who stands on two cards** — the help says "the extra card you received by
   hitting or splitting". We count the crowd's first hit on the shared hand as your third card (the shared-hand
   card *is* the third card of the hand, and the published 94.77 % is reproduced exactly when a third card is
   always drawn unless the dealer peeks a blackjack). If the real game only counts a card you took yourself, a
   7-7 hand you stand on can only win the two-card tiers.
5. **Six Card Charlie vs dealer blackjack** (help: "every hand of 21 or less with six cards wins") — implemented as
   winning even against a late 10-up dealer blackjack.
6. **21 auto-stand** also for soft 21 (help: you can draw "until your hand reaches 21").
7. **Napoleon two-halves procedure, cut-card position (75 % ± 8) and one burn card per half** are table-side
   observations, not in the help; `split_halves: false` gives the plain 8-deck shoe the help describes.
8. **Split by value** (10-J-Q-K pairs): help says "same value" — implemented; re-split not allowed (help: once).
