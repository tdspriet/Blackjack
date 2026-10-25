# ONE Blackjack Lab — Specification

A faithful simulator of **Pragmatic Play's ONE Blackjack** (as played at Napoleon Casino, Belgium),
a neural-network agent that plays it (main bet + 4 side bets + bet sizing + insurance), and a web
app that visualises everything:

```
┌──────────────── NN panel ───────────────┬──────────── Game table ────────────┬──────── Counting panel ────────┐
│ network graph w/ live activations       │ dealer, shared hand, crowd cards,  │ running/true counts (systems), │
│ input features, Q-values / EV outputs   │ bet spots (Bust Bonus, 21+3, MAIN, │ shoe: halves + cut card,       │
│ AI on/off, autoplay, speed, sim N hands │ Perfect Pairs, Crazy 7), actions,  │ remaining composition per rank,│
│ metrics: bankroll curve, EV/hand, W/L/P,│ SALDO / TOTALE INZET, messages     │ side-bet EV given composition, │
│ side-bet P&L, agreement w/ basic strat  │                                    │ count history over the shoe    │
└─────────────────────────────────────────┴────────────────────────────────────┴────────────────────────────────┘
```

Stack: **Python 3.12 (uv) + PyTorch + FastAPI/WebSocket** backend (`backend/`), **Vite + React + TypeScript**
frontend (`frontend/`). The Python engine is the single source of truth for rules; training uses the same engine.

---

## 1. Game rules (defaults — every item is a config field, see §3)

Sources: in-game felt text ("Blackjack pays 3 to 2 · 6 player cards win", "Dealer must draw at 16 and stand on
all 17's"), screenshots in `screens/`, blackjackinfo.com, casinobloke.com, livecasinocomparer.com, livecasinos.com.

### 1.1 Shoe ("Napoleon procedure")
* 8 decks (416 cards) are shuffled together, then **split into two halves of 4 decks (208 cards)**, each placed in
  its own container (two shoes visible left/right on the table).
* The dealer deals from one half. Each half has a **cut card**: default penetration `0.75` of the half
  (i.e. after 156 of 208 cards), with uniform jitter `±8` cards (human placement). One **burn card** (unseen) at the
  start of each half.
* When the cut card comes out, the current round is completed; the next round starts from the **second half**.
  After the second half's cut card → full 8-deck reshuffle (new shoe, new split).
* Cards behind the cut card of half 1 are never seen. **Key counting insight:** the two halves come from the same
  8-deck shuffle, so the count must NOT reset at the half switch: the composition of everything unseen
  (= 416 − seen) is what matters; true count = running count / (unseen cards / 52). The counting panel must
  explain/visualise this (two containers, cards seen per half, unseen residue of half 1).

### 1.2 Deal & shared hand
* ONE Blackjack is an unlimited-player game: **everyone shares the same initial hand**; each player makes their own
  decisions.
* Deal order: player card 1, dealer up card, player card 2, dealer hole card (face down).
* Dealer **peeks only when the up card is an Ace** (after insurance). With a 10-value up card, there is no peek: a
  dealer blackjack is discovered at the end. Config `peek_on_ten: false`.
* When the dealer has blackjack discovered late (10 up, no peek): config `late_dealer_bj_takes: "all" | "original"`,
  default `"all"` (doubles/splits lost too). **Six Card Charlie still wins vs. dealer blackjack.**
* If the (shared) player hand is a blackjack, everyone has blackjack → the round ends; the dealer turns the hole
  card but **does not draw**. Player BJ vs dealer BJ = push.

### 1.3 Player options
* Hit / Stand / Double / Split / Cash out. No surrender.
* **AUTHORITATIVE SOURCE: `docs/OFFICIAL_RULES_NL.txt`** (official in-game help, Dutch). It overrides anything below.
* Double on any first two cards (any total), one card only. Double after split allowed, **except after split aces**.
* **Cash out ("Uitbetalen")**: after the deal, each of your main hands has a cash-out offer (main bet only, not side
  bets) that is constantly recalculated from hand value, dealer up card and your decisions. Taking it ends that hand
  and pays the offer at settlement. Disabled after doubling or busting; each split hand has its own offer. RTP of the
  cash-out option is 98.00% → offer = `0.98 × fair value` where fair value = EV (incl. returned stake, per unit of the
  hand's bet) of the hand under optimal composition-independent play vs. that up card, computed for a full 8-deck
  shoe (the casino does NOT use the live composition — that is exactly what makes it interesting for a counter/NN).
  Precompute the table (hand state × up card → EV) once. Config `rules.cashout: true`, `rules.cashout_rtp: 0.98`.
  Action name `cashout`.
* Split once (no re-split), any two cards of equal value (10/J/Q/K splittable together — config
  `split_by_value: true`). Split aces receive exactly one card each, cannot hit/double; A+10 after split = 21, not BJ.
* **Six Card Charlie**: a non-busted hand of 6 cards wins automatically at 1:1 (it is auto-stood when it reaches 6
  cards) — even against dealer blackjack.
* Insurance: offered when dealer up card is Ace, costs half the main bet, pays 2:1. (Even money = insurance on BJ.)
* Blackjack pays 3:2. Wins 1:1. Ties push.
* Bet limits: main €1–€5000, each side bet €1–€500 (config). Side bets require a main bet. Chip values
  1, 2, 5, 10, 25, 100, 500, 1000.

### 1.4 Crowd (shared-hand simulation)
* `crowd.enabled: true`. A virtual crowd of `crowd.size` (default 250) players share the hand. Each crowd player
  has a type sampled at shoe start: `basic` (p=0.75, plays basic strategy), `cautious` (p=0.12, never hits hard 12+),
  `dealer_mimic` (p=0.10, hits below 17), `reckless` (p=0.03, hits below 19). (`cautious` = the official timeout default:
  stand on 12+, hit on 11 or less.)
* The shared hand keeps receiving cards **as long as at least `crowd.min_hitters` (default 1) players (crowd or
  you) want a card**. Your own hand is the prefix of the shared hand up to where you stood; cards after that are
  "crowd cards" (shown dimmed, tagged `owner:"crowd"`, still counted). A crowd player who doubles takes exactly one
  card. Crowd players never split.
* If **you** split: the crowd first finishes the unsplit shared hand, then your split hands are dealt
  (hand 1 then hand 2), then the dealer. If you double you get one card (shared with crowd hitters).
* The dealer plays out unless every active hand at the table (yours + crowd stop states) is resolved
  without needing the dealer (all busted or blackjack). With the crowd enabled the dealer practically always plays
  unless the shared hand is a blackjack.
* With `crowd.enabled: false` it is a normal single-player table.

### 1.5 Side bets (paytables in config; odds are "X to 1", stake returned on win)
| Bet | Cards used | Paytable |
|---|---|---|
| **Perfect Pairs** | player's first 2 cards | mixed pair (same rank, different colour) 6:1, coloured pair (same colour, different suit) 12:1, perfect pair (same suit) 25:1 |
| **21+3** | player's 2 cards + dealer up card (3-card poker) | flush 5:1, straight 10:1 (A low or high, no wrap), three of a kind 30:1, straight flush 40:1, suited trips 100:1 — highest only |
| **Crazy 7** | the first THREE cards of the player's (shared) hand: 2 initial cards + the 3rd card drawn by hitting (on the shared hand — the crowd's hit counts too) or, after splitting 7s, the next card dealt to the first split hand. NOT the dealer card. | first card is a 7 → 5:1; first two cards 7-7 different suits 25:1, same suit 150:1; first three cards 7-7-7 different suits 500:1, all same suit 2000:1. If the dealer peeks a blackjack, no third card is drawn. If nobody draws a 3rd card, only the 1- and 2-card tiers can pay. |
| **Bust Bonus** | dealer final hand | dealer busts with 3 cards 1:1, 4: 2:1, 5: 9:1, 6: 50:1, 7: 100:1, 8+: 250:1. Loses if dealer doesn't bust. **If the player has blackjack with the initial hand, the Bust Bonus bet is returned (push).** |

Official RTPs (in-game help) to calibrate against: main game **99.59%** ("optimal", on the initial bet);
21+3 **96.30%**, Bust Bonus **94.12%**, Perfect Pairs **95.90%**, Crazy 7 **94.77%**, cash-out option **98.00%**.
Remaining uncertain item: what happens to double/split stakes when a 10-up dealer turns out to have blackjack (no peek
on tens) — resolve by simulation against the 99.59% and document in `docs/RULES.md` with the evidence. Crazy 7's RTP
depends on how often a third card is drawn (crowd model) — report it with crowd on/off.

---

## 2. Repository layout

```
pyproject.toml                # uv project (python 3.12): torch, numpy, fastapi, uvicorn[standard], pydantic, pytest
config/default.json           # GameConfig defaults (rules, paytables, shoe, crowd, limits)
backend/onebj/
  __init__.py
  config.py                   # dataclasses + load_config(path=None) / to_dict
  cards.py                    # Card (int 0..51), rank/suit helpers, rank_bucket (A,2..9,T => 0..9), json
  shoe.py                     # NapoleonShoe: 8 decks, halves, burn, cut card, events, seen/unseen tracking
  hand.py                     # hand totals, soft, blackjack, charlie
  sidebets.py                 # settle_perfect_pairs/twenty_one_plus_three/crazy7/bust_bonus
  crowd.py                    # crowd model
  engine.py                   # Table: round state machine (see §4)
  strategy/basic.py           # basic strategy table for these rules (+ Illustrious-18 / Fab-4 style deviations)
  counting.py                 # count systems + true count + composition helpers
  analytics.py                # composition-dependent probabilities: side-bet EVs, insurance EV, dealer outcome
                              # distribution, approximate main-game edge from composition
  agents/base.py              # Agent protocol (§5)
  agents/basic.py             # BasicStrategyAgent (flat bet), HiLoAgent (basic + deviations + bet spread)
  ai/features.py              # observation → tensor, with feature names
  ai/model.py                 # PyTorch networks
  ai/agent.py                 # NNAgent: decisions + activation traces
  ai/train.py                 # CLI: uv run python -m onebj.ai.train ...
  ai/evaluate.py              # CLI: compare agents over N rounds
  sim.py                      # fast batched simulation utilities (multiprocessing)
  server/app.py               # FastAPI app: REST + WebSocket (§6), serves /cards
  server/session.py           # per-connection session: table, agent, autoplay loop, metrics
backend/tests/                # pytest
checkpoints/                  # trained models (*.pt + *.json metadata + training log)
frontend/                     # Vite React TS app
docs/SPEC.md, docs/RULES.md, README.md
cards/                        # existing PNGs (A_of_hearts.png ... back.png) — served at /cards/<code>.png
```

Run: `uv run uvicorn onebj.server.app:app --reload --port 8000` (backend, `backend/` on PYTHONPATH via
pyproject `[tool.uv] package` / src layout) and `npm run dev` in `frontend/` (Vite on 5173, proxies `/api`, `/ws`,
`/cards` to 8000). `make dev` / `./dev.sh` starts both.

---

## 3. Config (`config/default.json`)

```json
{
  "shoe": {"num_decks": 8, "split_halves": true, "penetration": 0.75, "cut_jitter": 8, "burn_cards": 1},
  "rules": {"blackjack_payout": 1.5, "dealer_hits_soft_17": false, "peek_on_ace": true, "peek_on_ten": false,
            "late_dealer_bj_takes": "all", "double_any_two": true, "double_after_split": true, "max_splits": 1,
            "split_by_value": true, "split_aces_one_card": true, "charlie_cards": 6, "charlie_beats_dealer_bj": true,
            "insurance": true, "insurance_payout": 2.0, "surrender": false, "cashout": true, "cashout_rtp": 0.98},
  "limits": {"main_min": 1, "main_max": 5000, "side_min": 1, "side_max": 500,
             "chips": [1, 2, 5, 10, 25, 100, 500, 1000]},
  "side_bets": {
    "perfect_pairs": {"mixed": 6, "coloured": 12, "perfect": 25},
    "twenty_one_plus_three": {"flush": 5, "straight": 10, "three_of_a_kind": 30, "straight_flush": 40, "suited_trips": 100},
    "crazy_7": {"one": 5, "pair_unsuited": 25, "pair_suited": 150, "three_unsuited": 500, "three_suited": 2000},
    "bust_bonus": {"3": 1, "4": 2, "5": 9, "6": 50, "7": 100, "8": 250}, "bust_bonus_push_on_player_blackjack": true
  },
  "crowd": {"enabled": true, "size": 250, "min_hitters": 1,
            "mix": {"basic": 0.75, "cautious": 0.12, "dealer_mimic": 0.10, "reckless": 0.03}},
  "bankroll": {"start": 1000}
}
```

---

## 4. Engine (Python, `onebj.engine.Table`)

A `Table` owns a persistent shoe (state survives across rounds) and a bankroll.

```python
from onebj.config import load_config
from onebj.engine import Table, Bets, Action, Phase

t = Table(load_config(), seed=42)
t.place_bets(Bets(main=10, perfect_pairs=1, twenty_one_plus_three=0, crazy_7=0, bust_bonus=1))  # deals
t.phase            # Phase.INSURANCE | Phase.PLAYER | Phase.SETTLED (BJ etc.)
t.legal_actions()  # [Action.HIT, Action.STAND, Action.DOUBLE, Action.SPLIT] or [INSURE, NO_INSURANCE]
t.act(Action.HIT)  # advances; when the player is done → crowd → dealer → settle automatically
t.round_result     # RoundResult (per-hand results, side-bet results, net, insurance)
t.observation()    # Observation: everything a player could legally know (NO hidden info)
t.snapshot()       # JSON-able dict = "game" + "shoe" part of the WebSocket state (§6)
t.clone(resample_unseen=True)  # deep copy; with resample_unseen the unseen cards are reshuffled so
                               # rollouts can't peek at the true shoe order (needed for training targets)
t.events           # list of events of the current round (card dealt w/ owner, phase changes, settlements)
```

* Phases: `BETTING → (deal) → INSURANCE? → PLAYER → CROWD → DEALER → SETTLED`. `t.place_bets` is only valid in
  `BETTING`/`SETTLED`; after settlement the next `place_bets` starts a new round (and handles half switch/reshuffle
  between rounds). `Phase.SHUFFLE` is reported as an event when a half switch or reshuffle happens.
* `Observation` (dataclass) includes: player hands (cards, totals, soft, num cards, flags can_double/can_split,
  is_split, split_aces), active hand index, dealer up card, phase, legal actions, bets, bankroll,
  `seen_counts[10]` (rank buckets A,2..9,T seen this 8-deck shoe), `unseen_counts[10]` (= full − seen, includes
  burn/residue/hole card), `cards_unseen`, `half` (1/2), `cards_left_in_half_to_cut`, running counts.
* Must be fast: ≥ 20k rounds/s/core for a flat-bet basic-strategy sim is the target (avoid per-card object churn;
  ints for cards; precomputed tables).
* Fully deterministic given `seed`.

### 4.1 Engine API as implemented (authoritative details for the NN / server engineers)

```python
Table(config: GameConfig | None = None, seed: int | None = None, record_events: bool = True,
      strict: bool = False, bankroll: float | None = None)
  .place_bets(bets: Bets) -> None            # BETTING/SETTLED only; limits always validated (ValueError)
  .legal_actions() -> list[Action]           # PLAYER: HIT, STAND, [DOUBLE], [SPLIT], [CASHOUT]; INSURANCE: [INSURE], NO_INSURANCE
  .act(action: Action | str) -> None         # runs crowd → dealer → settlement automatically
  .cashout_offer(index: int | None = None) -> float | None   # money; active hand only
  .observation() -> Observation              # no hidden info
  .snapshot(analytics: bool = True) -> dict  # {"game": ..., "shoe": ...} per §6
  .clone(resample_unseen: bool = True, seed: int | None = None) -> Table
  .decision_state() -> (total, soft, n_cards, up_bucket, can_double, pair_bucket|-1, can_split, is_split)
  .new_shoe() / .reset(bankroll=None)
  .phase, .round_id, .balance, .bets, .hands, .active, .dealer, .events, .messages, .round_result, .history,
  .shoe (NapoleonShoe), .crowd (Crowd), .strategy (BasicStrategy), .cashout_table (CashoutTable | None)
Action: HIT STAND DOUBLE SPLIT CASHOUT INSURE NO_INSURANCE   (str Enum; values = protocol strings)
Phase:  BETTING INSURANCE PLAYER CROWD DEALER SETTLED SHUFFLE (str Enum; CROWD/DEALER/SHUFFLE only in events)
Bets(main, perfect_pairs, twenty_one_plus_three, crazy_7, bust_bonus)   .total .to_dict() Bets.from_dict(d)
```

* **Shoe transition timing**: the half switch / reshuffle is executed right after settlement (a `shuffle` event
  closes the round's event list), so `observation()`/`snapshot()` taken in `SETTLED` already describe the shoe the
  next round is dealt from (betting decisions see the right composition). `shoe.pending_shuffle` /
  `shoe.cut_card_out` are extra snapshot fields (`"half_switch" | "reshuffle" | null`).
* **Observation** fields: `round_id, phase, legal_actions, hands: list[HandView], active_hand, dealer_up (card int or
  None), dealer_cards (visible), dealer_total (visible), bets, insurance_stake, bankroll, seen_counts[10],
  unseen_counts[10], unseen_card_counts[52], cards_unseen, decks_unseen, half, cards_left_in_half_to_cut,
  shoe_number, running_counts{system}, true_counts{system}, crowd_cards, num_decks`; `.hand` = active HandView.
  `HandView = (cards, total, soft, num_cards, bet, doubled, is_split, split_aces, status, can_double, can_split,
  pair, cashout_offer)`.
* **RoundResult**: `hands: list[HandResult(cards, bet, doubled, total, result, payout, split_aces)]`,
  `side_bets: list[SideBetResult(bet, outcome, odds, stake, payout)]`, `insurance_stake, insurance_payout,
  dealer_cards, dealer_total, dealer_blackjack, dealer_bust, dealer_played, shared_blackjack, main_bet (original),
  main_wagered, main_returned, wagered, returned, net, crowd_cards`, `.main_net`, `.side_net(name)`.
* **Events** (`record_events=True`): `round_start`, `card {card, owner, hand, face_up[, hitters]}`, `reveal {card}`
  (hole card turned), `phase {phase}`, `peek {blackjack}`, `insurance {taken, stake}`, `insurance_result`,
  `double`, `split`, `cashout {hand, amount}`, `side_bet {bet, outcome, odds, stake, payout}`, `cut_card`,
  `settle {net, returned, hands}`, `shuffle {kind: "half_switch"|"reshuffle", shoe_number, half}`.
  `onebj.counting.CountTracker.apply_events` reproduces the engine's counts from this stream.
* **Rules details**: 21 is auto-stood (official: you can hit "until your hand reaches 21"). Your own hit always
  gets a card; `crowd.min_hitters` applies to the crowd phase. Crazy 7 with a 7-7 start is settled at the end of the
  round (the third card may come from you, the crowd, or split hand 1). Without a crowd the dealer still plays out
  if you have a Bust Bonus bet. `late_dealer_bj_takes: "original"`: busted hands lose, the first live hand loses its
  original stake only (double refunded), further split hands are refunded; Charlie hands win. Cash-out offers are
  shown for the active hand only and paid even if the dealer later shows a blackjack.
* KO is reported with IRC `4 − 4·decks`; its `true` value is balanced-equivalent
  `(rc − IRC − 4·cards_seen/52) / decks_unseen`. `history` entries are `{round, rc, tc, half}` (Hi-Lo), reset at a
  reshuffle only. `analytics` additionally has `dealer_outcomes_by_upcard`.
* Simulation: `onebj.sim.simulate(config=None, agent="basic"|"basic_sides"|"hilo"|"hilo_nosides"|factory,
  rounds=100_000, seed=0, workers=None) -> dict` (multiprocessing); `run_rounds(table, agent, rounds, metrics=None,
  progress=None, progress_every=5000, bankroll_every=0, should_stop=None) -> Metrics` (in-process, for the server);
  `play_round(table, agent) -> RoundResult`. Agents may implement `fast_bets(table)`, `fast_insurance(table)`,
  `fast_action(table, decision_state)` to skip Observation building.
* Strategy / analytics helpers: `onebj.strategy.basic.get_basic_strategy(rules, num_decks)` →
  `BasicStrategy.action(total, soft, n_cards, up_bucket, can_double, pair, can_split) -> "hit"|"stand"|"double"|"split"`;
  `hilo_action(...)`, `should_insure(tc)`; `onebj.strategy.cashout.get_cashout_table(rules).fair_value(...)`;
  `onebj.analytics.side_bet_evs(unseen_card_counts, cfg)`, `insurance_ev`, `main_edge_estimate`,
  `dealer_distribution`, `table_analytics(table)`.

---

## 5. Agents (`onebj.agents.base.Agent`)

```python
class Agent(Protocol):
    name: str
    def decide_bets(self, obs: Observation) -> tuple[Bets, Trace | None]: ...
    def decide_insurance(self, obs: Observation) -> tuple[bool, Trace | None]: ...
    def decide_action(self, obs: Observation) -> tuple[Action, Trace | None]: ...
```

`Trace` (JSON-able dict) = what the NN panel renders:
```json
{"kind": "play" | "bet" | "insurance",
 "features": [{"name": "player_total", "value": 0.62, "raw": 16}, ...],
 "layers": [{"name": "input", "size": 48, "activations": [...]},
            {"name": "trunk.0", "size": 128, "activations": [...]}, ...],
 "outputs": {"labels": ["stand","hit","double","split","cashout"], "values": [-0.21, -0.18, -0.55, null, -0.24], "legal": [true,true,true,false,true],
             "unit": "EV per unit bet"},
 "chosen": "hit",
 "reference": {"basic_strategy": "hit", "hilo_deviation": null},
 "extra": {"predicted_ev": {"main": -0.004, "perfect_pairs": -0.04, ...}, "bet_fraction": 0.0}}
```

Network topology (for the static graph) is exposed via `NNAgent.describe()`:
`{"name", "checkpoint", "layers": [{"name","size","labels"?}], "heads": [...], "edges": {"<layerA>-><layerB>": top-K
weights as [[i,j,w],...]}, "training": {...metadata}}`.

Baselines: `BasicStrategyAgent` (flat min bet, no sides, never insures), `HiLoAgent` (basic + Illustrious 18 index
plays, insurance at TC ≥ +3, 1–12 spread by true count, side bets when analytic EV > 0).

### 5.1 NNAgent as implemented (`onebj/ai/`, authoritative details — see docs/NN.md for the ML side)

```python
from onebj.ai.agent import NNAgent, BetConfig, list_checkpoints, load_training_log
list_checkpoints(dir="checkpoints") -> list[dict]      # [{name, path (.pt, absolute), created (ISO), metrics}], newest first
load_training_log(path_or_name, dir="checkpoints") -> list[dict]   # = checkpoints/<name>.json["training_log"]
NNAgent.load(path_or_name, config: GameConfig | None = None, *, trace=True, bet: BetConfig | dict | None = None,
             dir="checkpoints") -> NNAgent              # name in dir, or a path to <name>.pt
agent.name == "nn:<name>";  agent.trace: bool;  agent.set_trace(on)   # trace=False → decide_* return (decision, None)
agent.decide_bets(obs) / decide_insurance(obs) / decide_action(obs)    # Agent protocol, (decision, Trace | None)
agent.fast_bets(table) / fast_insurance(table) / fast_action(table, decision_state) -> str   # onebj.sim fast path
agent.describe() -> dict
```

* Relative `dir`s resolve against the CWD first, then the repository root. A checkpoint is `<name>.pt` (torch
  `state_dict` + architecture + feature spec, loaded with `weights_only=True`) plus `<name>.json` (metadata: `name,
  created, config_hash, feature_names, feature_spec, arch, bet_config, train_config, gen_config, game_config, data,
  timing, metrics, training_log` and, after `evaluate --save-metrics`, `evaluation`).
* **Network**: `input` (61 features) → `trunk.0` (128, ReLU) → `trunk.1` (64, ReLU) → heads `play` (5:
  `["stand","hit","double","split","cashout"]`) and `bet` (6: `["main","perfect_pairs","twenty_one_plus_three",
  "crazy_7","bust_bonus","insurance"]`), both `from: "trunk.1"`. The `cashout` output is wired to the input
  `cashout_value` (= offer / bet − 1, exact) and has no edges.
* **describe()**: `layers` = input (with `labels` = feature names), `trunk.0`, `trunk.1`; `heads` as above (with
  `labels`, `from`); `edges` = `{"input->trunk.0", "trunk.0->trunk.1", "trunk.1->play", "trunk.1->bet"}`, each the
  top ≤ 400 weights by |w| as `[i, j, w]` (bet-head weights are multiplied by the head's output scale, i.e. in EV
  units); `training` = `{created, config_hash, metrics, data, timing, train_config, gen_config, description,
  cashout_output, bet_config}`.
* **Traces** (all JSON-able, floats rounded): `features` = all 61 `{name, value (fed to the net), raw (human value or
  null)}`; `layers` = `input`, `trunk.0`, `trunk.1` activations (post-ReLU) and the active head (`play`, or `bet` for
  bet and insurance decisions).
  * `play`: `outputs` = Q per label in EV per unit of the hand's original bet, `null` for illegal actions, `legal` =
    mask; `chosen` = argmax over legal; `reference = {basic_strategy, hilo_deviation}` (the Illustrious-18 entry that
    applies at the current TC, else null); `extra = {cashout_offer (money), q_margin, true_count}`.
  * `insurance`: `outputs = {labels: ["insure","no_insurance"], values: [predicted EV per unit insurance stake, 0]}`;
    `reference.basic_strategy = "no_insurance"`, `hilo_deviation = "insurance"` at TC ≥ +3; `extra = {predicted_ev:
    {insurance}, analytic_ev: {insurance}, true_count}`.
  * `bet`: `outputs` = the 6 bet-head EVs per unit stake (`legal` all true); `chosen` = main stake as text;
    `reference.hilo_bet_units` = what the Hi-Lo ramp would bet (string); `extra = {predicted_ev{6}, bet_fraction (total
    stake / bankroll), bets (Bets dict), kelly {edge, kelly_stake, bankroll, units, side_kelly_stakes}, analytic_ev
    (exact side-bet / insurance EVs and the EoR main-edge estimate for the composition), true_count}`.
* **Bet sizing** (`BetConfig`, stored in the checkpoint, overridable with `bet=`): main = `kelly_fraction (0.5) ×
  predicted edge / main_variance (1.33) × bankroll`, rounded to whole euros, clamped to `[unit, unit × max_units]`
  (defaults €1 and 12); `bankroll_mode "fixed"` uses `bankroll` (default `config.bankroll.start`), `"live"` the
  current balance. Side bet placed when predicted EV > `side_threshold` (0.02), stake = side-bet Kelly stake clamped
  to `[side_min, side_min × side_max_units]` (= the €1 minimum in practice). Insure when the predicted insurance EV >
  `insurance_threshold` (0).
* Other evaluator agents (`onebj.ai.evaluate.build_agent`): `"<ckpt>:flat"` (NN play only, flat minimum, no side
  bets/insurance), `"<ckpt>:nosides"`, `"solver"` (`onebj.ai.oracle.SolverAgent`: composition-dependent DP + cash-out,
  flat, slow reference), `"solver_nocash"`.

---

## 6. Server protocol (FastAPI)

REST
* `GET /api/config` → config JSON. `GET /api/checkpoints` → list `{name, created, metrics}`.
* `GET /api/training/{name}` → training log (`[{step, loss, eval_ev, ...}]`). `GET /cards/{code}.png` static.

WebSocket `GET /ws` — server → client messages:

```jsonc
{"type": "hello", "config": {...}, "network": {/* NNAgent.describe() */}, "agents": ["nn:<ckpt>", "hilo", "basic"]}

{"type": "state",
 "game": {
   "round_id": 17, "phase": "betting|insurance|player|crowd|dealer|settled",
   "balance": 1012.5, "total_bet": 12,
   "bets": {"main": 10, "perfect_pairs": 1, "twenty_one_plus_three": 0, "crazy_7": 0, "bust_bonus": 1, "insurance": 0},
   "dealer": {"cards": [Card...], "total": 10, "soft": false, "blackjack": false, "bust": false},
   "hands": [{"cards": [Card...], "bet": 10, "doubled": false, "total": 16, "soft": false,
              "status": "active|stood|bust|blackjack|charlie|cashed_out|done",
              "result": null|"win|lose|push|blackjack|charlie|cashout", "payout": 0, "split_aces": false,
              "cashout_offer": 9.12 /* null when unavailable */}],
   "active_hand": 0,
   "crowd": {"enabled": true, "cards": [Card...], "hitters": 12, "standers": 238},
   "legal_actions": ["hit","stand","double","split","cashout"] /* or ["insure","no_insurance"] or ["deal"] */,
   "side_bet_results": [{"bet": "perfect_pairs", "outcome": "mixed_pair", "odds": 6, "stake": 1, "payout": 7}],
   "round_net": null, "messages": ["Dealer busts with 4 cards — Bust Bonus 2:1"]
 },
 "shoe": {
   "shoe_number": 3, "half": 1, "half_size": 208, "dealt_in_half": 58, "cut_at": 153, "burned": 1,
   "cards_unseen": 358, "decks_unseen": 6.88,
   "seen_counts": {"A": 4, "2": 5, ..., "T": 20}, "unseen_counts": {"A": 28, ..., "T": 108},
   "running": {"hilo": 5, "ko": 9, "omega2": 7, "wong_halves": 4.5, "zen": 6},
   "true": {"hilo": 0.73, ...},
   "analytics": {"perfect_pairs": -0.041, "twenty_one_plus_three": -0.037, "crazy_7": -0.06, "bust_bonus": -0.062,
                 "insurance": -0.071, "main_edge_estimate": -0.004, "p_dealer_bust_by_upcard": {...}},
   "history": [{"round": 1, "tc": 0.2, "rc": 1}, ...]
 },
 "ai": {"agent": "nn:v1", "advice": Trace|null, "last_trace": Trace|null, "autoplay": false},
 "metrics": {"rounds": 1234, "net": 52.0, "wagered": 13100, "ev_per_round": 0.042, "edge": 0.004,
             "wins": 520, "losses": 610, "pushes": 104, "blackjacks": 58, "charlies": 3,
             "bankroll": [[round, balance], ...], "side_bets": {"perfect_pairs": {"wagered": 40, "net": -3}, ...},
             "agreement_basic": 0.97, "max_drawdown": 210, "rounds_per_sec": null}}

{"type": "sim_progress", "done": 5000, "total": 100000, "metrics": {...}, "finished": false, "agent": "hilo"}   // headless simulation
{"type": "error", "message": "..."}
```

`Card` = `{"rank": "A|2..10|J|Q|K", "suit": "hearts|diamonds|clubs|spades", "code": "A_of_hearts",
"face_up": true, "owner": "player|crowd|dealer"}` (face-down cards: `{"face_up": false, "code": "back"}`).

Client → server:
```jsonc
{"cmd": "place_bets", "bets": {"main": 10, "perfect_pairs": 0, "twenty_one_plus_three": 0, "crazy_7": 0, "bust_bonus": 0}}
{"cmd": "action", "action": "hit|stand|double|split|cashout|insure|no_insurance"}
{"cmd": "ai_step"}                                  // AI makes the next decision (bets or action)
{"cmd": "autoplay", "on": true, "delay_ms": 700}     // AI plays continuously, state streamed with pacing
{"cmd": "advice", "on": true}                        // show AI recommendation while human plays
{"cmd": "simulate", "rounds": 100000, "agent": "nn:v1"}  // headless, fast; emits sim_progress
{"cmd": "stop"}
{"cmd": "set_agent", "agent": "nn:v1|hilo|basic"}
{"cmd": "new_shoe"} {"cmd": "reset", "bankroll": 1000}
{"cmd": "set_config", "patch": {"crowd": {"enabled": false}}}
```

Dealer/crowd cards are streamed one by one (a `state` per card) with `delay_ms` pacing for animation in
interactive/autoplay modes; zero pacing in `simulate`.

### 6.1 Server as implemented (`onebj/server/`, authoritative details)

* **Session** = one WebSocket connection: own seeded table (persistent shoe, bankroll enforced — `strict=True`),
  agent, metrics. `?seed=N` on `/ws` makes it reproducible. Default agent: newest NN checkpoint, else `hilo`.
* **hello**: `network` = `describe()` of the selected NN agent, else of the newest checkpoint (graph without
  activations), `null` when no checkpoint exists. `agents` = `nn:<name>` per checkpoint (newest first), `hilo`,
  `basic`. A new `hello` is sent after every `set_config` (new config) and after `set_agent` when the network changes.
* **Streaming**: every decision produces the intermediate frames (round start with empty hands, each card of the deal,
  each crowd / dealer / split-hand card, the hole-card reveal, the split itself) and then the final state. Intermediate
  frames have `legal_actions: []`, `cashout_offer: null`, `round_net: null`, `ai.advice: null`, the transient `phase`
  (`player` while dealing, `crowd`, `dealer`) and the metrics from before the decision. Pacing: 0.38 s per card when a
  human plays, `clamp(0.45·delay_ms, 30, 600)` ms in autoplay; autoplay waits `0.8·delay_ms + 60` ms before a play
  decision and `1.6·delay_ms + 150` ms before the next round. While frames are pending, `place_bets`, `action`,
  `ai_step`, `new_shoe`, `set_config` are refused (`error` "Please wait — cards are still being dealt").
* **game.side_bet_results**: lost side bets have `outcome: "lose"` (and `odds: 0`); a settled insurance bet is listed
  as `{"bet": "insurance", "outcome": "dealer_blackjack" | "lose", "odds", "stake", "payout"}`.
* **ai**: `last_trace` = trace of the agent's last decision (`ai_step` / autoplay; human moves don't change it; cleared
  by `set_agent` / `reset`). `advice` (when on, not in autoplay) = the agent's trace for the pending decision, including
  bet decisions (`extra.bets` = the recommended bets, scaled to the bankroll). Baseline traces: play `outputs` = per-action
  EV per unit bet from the fresh-shoe solver (cash-out = offer/bet − 1), bet `outputs` = analytic EV per unit stake of
  `main` (edge estimate), each side bet and insurance for the current composition; `reference.basic_strategy` always set.
* **metrics** (session and simulation, `onebj.server.metrics.LabMetrics`, a superset of `onebj.sim.Metrics.to_dict`):
  `edge` = main-game net / original main stakes (the official RTP definition); `ev_per_round` = total net per round
  incl. side bets and insurance; `wins/losses/pushes/blackjacks/charlies` count main hands; `side_bets[name]` =
  `{wagered, net, rtp, bets, wins, rtp_se}` plus an `insurance` row `{wagered, net, rtp}`; `agreement_basic` = share of
  play decisions (human and AI in a session) equal to basic strategy; `bankroll` ≤ ~1000 points (stride doubles);
  extra fields `std_per_round, main_rtp, edge_se, main_std_per_unit, doubles, splits, dealer_bust_rate, cashouts`;
  a cashed-out hand counts as a win / loss / push by whether the offer was above / below / equal to its stake.
* **simulate** runs on fresh tables with the session's config and an unlimited virtual bankroll (curve starts at
  `bankroll.start`), split over up to 8 spawned worker processes (threads with `ONEBJ_SIM_PROCESSES=0`); the
  session's table is not touched and stays playable. Messages: an immediate `sim_progress` with `done: 0`, then about
  every 0.5 s, then a final one with `"finished": true` (also after `stop`, with `done < total`), then a `state`.
  `sim_progress` also carries `agent`. Multi-worker bankroll curves interleave the workers (point k = sum of every
  worker's k-th sample); `max_drawdown` is measured on that curve.
* **Errors**: invalid JSON / unknown command / bad arguments / illegal moves → `{"type": "error", "message"}` followed
  by a fresh `state` (re-sync); the socket is never closed. `set_config` applies `onebj.config.apply_patch` between
  rounds: `crowd` / `limits` / `side_bets` / `bankroll` changes keep the shoe, `shoe` / `rules` changes rebuild the
  table (bankroll and round number kept). Agents' bets are scaled down to the bankroll (side bets dropped first).
* **Debug** (`ONEBJ_DEBUG=1`): `{"cmd": "debug_rig", "cards": ["8_of_hearts", "6s", ...]}` makes the next cards drawn
  be these (swapped in from the unseen part of the shoe, so the composition is unchanged).
* **REST** extras: `GET /api/health` → `{status, version, checkpoints, agents, debug}`. `GET /api/training/{name}` reads
  `checkpoints/<name>.json` (`training_log` key, or the file itself if it is a list). With `frontend/dist` built, the app
  serves it at `/` (SPA fallback) next to `/api`, `/ws`, `/cards`.

---

## 7. Neural network

Goals: play the full game (bets incl. side bets & sizing, insurance, hit/stand/double/split) using everything a
card counter can see, and be inspectable.

* **Features** (`onebj/ai/features.py`, named): hand total (normalised + one-hot 4..21), soft, pair rank one-hot,
  num cards (Charlie!), can_double, can_split, is_split, dealer up card one-hot (10), composition: unseen fraction per
  rank bucket minus expected (10), decks unseen, Hi-Lo true count, half indicator, cards-to-cut.
* **Play network**: MLP trunk → Q-head over [stand, hit, double, split, cashout] (EV per unit of original bet), masked by legality.
  (Cashout value is known exactly = offer/bet − 1, so the net only has to learn when holding beats it.)
  *As implemented:* the targets are the action EVs of a cash-aware dynamic-programming solver at the current unseen
  composition (`onebj/ai/oracle.py`), iterated over generations with the network's own behaviour policy; all-action
  Monte-Carlo rollouts on `clone(resample_unseen=True)` are implemented as an alternative/blend (`targets: "mc" |
  "blend"`) and were measured to be worse at equal budget — see docs/NN.md §2.1.
* **Bet network**: composition features → predicted EV per unit for main bet (under the current play policy) and each side
  bet + insurance; bet sizing = fractional Kelly on the predicted edge, clamped to limits; side bet placed when predicted
  EV > threshold. *As implemented:* one network with a shared trunk and two heads (`play`, `bet`); bet targets are
  realised engine returns (main, Bust Bonus) and a deal Monte Carlo settled with the engine's `settle_*` (PP, 21+3,
  Crazy 7, insurance).
* Extra modules: `ai/datagen.py` (multiprocess data generation), `ai/policy.py` (fast table policies), `ai/oracle.py`
  (`CashSolver`, `SolverAgent`), `ai/analysis.py` (solver regret, strategy charts, index scans, bet head vs exact),
  `ai/sidebet_study.py` (exact side-bet +EV frequencies), `notebooks/nn_playground.ipynb`.
* Checkpoints in `checkpoints/<name>.pt` + `<name>.json` (config hash, feature names, metrics, training log).
* Evaluation report vs baselines (basic flat, Hi-Lo spread): EV per round, edge, std, agreement with basic strategy,
  side-bet results.
