### Full game, 2,000,000 rounds per agent (seed 2026)

| agent | EV/round (units) | ± SE | main edge (on initial bet) | main edge on wagered | std/round (units) | avg bet (units) | agreement w/ basic | cash-outs /1000 | insured / offered | side bets (n, RTP) |
|---|---|---|---|---|---|---|---|---|---|---|
| basic | -0.00324 | 0.00080 | -0.324% | -0.292% | 1.126 | 1.00 | 100.00% | 0.00 | 0 / 153,667 | — |
| nn_v1:flat_nocash | -0.00209 | 0.00080 | -0.209% | -0.188% | 1.131 | 1.00 | 95.96% | 0.00 | 0 / 153,609 | — |
| nn_v1:flat | +0.00199 | 0.00072 | +0.199% | +0.182% | 1.023 | 1.00 | 80.95% | 210.58 | 0 / 153,419 | — |
| nn_v1:nosides | +0.03234 | 0.00222 | +1.540% | +1.395% | 3.138 | 2.02 | 80.91% | 210.64 | 15,747 / 153,426 | — |
| nn_v1 | +0.11671 | 0.00801 | +1.540% | +1.395% | 11.334 | 2.02 | 80.91% | 210.64 | 15,747 / 153,426 | PP 8,334 @ 101.3%; 21+3 11,133 @ 101.1%; C7 627,286 @ 121.5%; BB 136,774 @ 108.8% |
| hilo_nosides | +0.00479 | 0.00157 | +0.285% | +0.256% | 2.221 | 1.38 | 98.28% | 0.00 | 9,168 / 153,683 | — |
| hilo | +0.06840 | 0.00723 | +0.285% | +0.256% | 10.223 | 1.38 | 98.28% | 0.00 | 9,168 / 153,683 | PP 41,858 @ 102.6%; 21+3 55,440 @ 102.4%; C7 698,405 @ 116.3%; BB 208,425 @ 105.1% |
| solver_nocash | -0.00206 | 0.00080 | -0.206% | -0.185% | 1.132 | 1.00 | 96.00% | 0.00 | 0 / 153,559 | — |
| solver | +0.00181 | 0.00072 | +0.181% | +0.166% | 1.024 | 1.00 | 80.98% | 210.23 | 0 / 153,413 | — |

### Cash-out diagnostic

EV per unit of the hand's original bet in the states where the agent cashed out: the cash-out EV (exact) vs the best *hold* EV from the composition solver and vs one realised counterfactual hold round per cash-out (resampled unseen cards). Cash-out rate = share of play decisions, by Hi-Lo TC.

| agent | cash-outs / 1000 rounds | mean cash-out EV | mean hold EV (solver) | mean hold return (MC) | gain per cash-out vs solver hold | TC -4 | TC -3 | TC -2 | TC -1 | TC +0 | TC +1 | TC +2 | TC +3 | TC +4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_v1:flat | 210.6 | -0.1936 | -0.2139 | -0.2188 | +0.0203 | 24.5% | 21.6% | 18.2% | 13.1% | 9.5% | 16.4% | 24.1% | 28.6% | 32.8% |
| nn_v1:nosides | 210.6 | -0.1950 | -0.2152 | -0.2215 | +0.0202 | 24.5% | 21.6% | 18.2% | 13.1% | 9.5% | 16.4% | 24.2% | 28.6% | 32.7% |
| nn_v1 | 210.6 | -0.1950 | -0.2152 | -0.2215 | +0.0202 | 24.5% | 21.6% | 18.2% | 13.1% | 9.5% | 16.4% | 24.2% | 28.6% | 32.7% |
| solver | 210.2 | -0.1921 | -0.2133 | -0.2201 | +0.0212 | 24.3% | 21.6% | 18.2% | 13.0% | 9.3% | 16.6% | 24.4% | 28.7% | 32.9% |

By hand class — nn_v1:flat (overall cash-out rate 16.4% of play decisions, mean Hi-Lo TC when cashing out +0.14):

| hand class | decisions /1000 rounds | cash-out rate | mean TC at cash-out | gain per cash-out vs solver hold |
|---|---|---|---|---|
| 3+ cards | 318.0 | 15.3% | +0.40 | +0.0154 |
| hard 12-16 v 2-6 | 151.3 | 19.9% | -1.54 | +0.0211 |
| hard 12-16 v 7-A | 220.9 | 26.0% | +1.47 | +0.0194 |
| hard 17-21 | 157.3 | 15.5% | -0.25 | +0.0172 |
| hard <=11 | 196.0 | 13.1% | -1.32 | +0.0351 |
| pair | 141.8 | 8.5% | +0.11 | +0.0199 |
| soft 2-card | 95.1 | 12.8% | +0.84 | +0.0170 |

By hand class — nn_v1:nosides (overall cash-out rate 16.5% of play decisions, mean Hi-Lo TC when cashing out +0.14):

| hand class | decisions /1000 rounds | cash-out rate | mean TC at cash-out | gain per cash-out vs solver hold |
|---|---|---|---|---|
| 3+ cards | 318.1 | 15.3% | +0.40 | +0.0152 |
| hard 12-16 v 2-6 | 151.3 | 20.0% | -1.53 | +0.0211 |
| hard 12-16 v 7-A | 221.0 | 25.8% | +1.47 | +0.0194 |
| hard 17-21 | 157.3 | 15.9% | -0.20 | +0.0173 |
| hard <=11 | 196.0 | 13.1% | -1.32 | +0.0351 |
| pair | 141.7 | 8.3% | +0.06 | +0.0199 |
| soft 2-card | 95.1 | 12.7% | +0.83 | +0.0166 |

By hand class — nn_v1 (overall cash-out rate 16.5% of play decisions, mean Hi-Lo TC when cashing out +0.14):

| hand class | decisions /1000 rounds | cash-out rate | mean TC at cash-out | gain per cash-out vs solver hold |
|---|---|---|---|---|
| 3+ cards | 318.1 | 15.3% | +0.40 | +0.0152 |
| hard 12-16 v 2-6 | 151.3 | 20.0% | -1.53 | +0.0211 |
| hard 12-16 v 7-A | 221.0 | 25.8% | +1.46 | +0.0194 |
| hard 17-21 | 157.3 | 15.9% | -0.20 | +0.0173 |
| hard <=11 | 196.0 | 13.1% | -1.32 | +0.0351 |
| pair | 141.7 | 8.3% | +0.06 | +0.0199 |
| soft 2-card | 95.1 | 12.7% | +0.83 | +0.0166 |

By hand class — solver (overall cash-out rate 16.4% of play decisions, mean Hi-Lo TC when cashing out +0.15):

| hand class | decisions /1000 rounds | cash-out rate | mean TC at cash-out | gain per cash-out vs solver hold |
|---|---|---|---|---|
| 3+ cards | 317.7 | 14.9% | +0.39 | +0.0173 |
| hard 12-16 v 2-6 | 151.1 | 20.1% | -1.52 | +0.0213 |
| hard 12-16 v 7-A | 220.9 | 25.9% | +1.47 | +0.0197 |
| hard 17-21 | 157.1 | 15.9% | -0.22 | +0.0180 |
| hard <=11 | 195.7 | 12.8% | -1.30 | +0.0371 |
| pair | 141.7 | 8.8% | +0.09 | +0.0210 |
| soft 2-card | 95.1 | 13.3% | +0.90 | +0.0179 |

### Paired play comparison vs basic strategy (flat 1 unit, no side bets/insurance), 2,000,000 rounds

| agent | EV/round | basic EV/round (same cards) | gain vs basic | ± SE | rounds that differ | agreement |
|---|---|---|---|---|---|---|
| nn_v1 (play only) | +0.00179 | -0.00444 | +0.00623 | 0.00038 | 22.41% | 80.96% |
| nn_v1 (play only, never cashes out) | -0.00311 | -0.00449 | +0.00139 | 0.00019 | 2.50% | 95.97% |
| hilo_nosides (play only) | -0.00397 | -0.00443 | +0.00046 | 0.00012 | 0.86% | 98.29% |
| solver (play only) | +0.00195 | -0.00450 | +0.00645 | 0.00038 | 22.41% | 80.97% |
| solver_nocash (play only) | -0.00310 | -0.00465 | +0.00154 | 0.00019 | 2.50% | 96.02% |

### Learned strategy — nn_v1

Solver regret on 60,000 natural decision states (basic strategy + 10% exploration; regret = EV lost per decision vs the composition-dependent solver, cash-out exact):

| policy | regret per decision | agreement with solver-optimal | cash-out rate |
|---|---|---|---|
| basic | 0.00568 | 77.86% | 0.00% |
| net | 0.00032 | 95.32% | 17.89% |

Q-value RMSE vs solver EVs: 0.0075 (bias +0.0007).


Index plays (3 decks unseen, Hi-Lo-shifted composition): true count where the network switches

| play | Hi-Lo index | NN index |
|---|---|---|
| insurance | +3 | +4.0 |
| 16 v T stand | +0 | +0.1 |
| 15 v T stand | +4 | +5.4 |
| TT v 5 split | +5 | +5.2 |
| TT v 6 split | +4 | +4.3 |
| 11 v T double | +3 | never |
| 12 v 3 stand | +1 | +1.2 |
| 12 v 2 stand | +3 | +2.7 |
| 11 v A double | +2 | +1.9 |
| 9 v 2 double | +1 | +1.1 |
| 10 v A double | +4 | +4.3 |
| 9 v 7 double | +3 | +4.0 |
| 16 v 9 stand | +5 | +4.7 |
| 13 v 2 hit | -1 | -1.1 |
| 12 v 4 hit | +0 | +0.0 |
| 12 v 5 hit | -2 | -1.3 |
| 12 v 6 hit | -1 | -1.0 |
| 13 v 3 hit | -2 | -2.2 |

Strategy chart at TC -3 (3 decks unseen; `*` = differs from basic strategy, C = cash out)

```
         2   3   4   5   6   7   8   9   T   A
H5      H   H   H   H   H   H   H   H   H   H 
H6      H   H   H   H   H   H   H   H   H   H 
H7      H   H   H   H   H   H   H   H   H   H 
H8      H   C*  C*  C*  C*  H   H   H   H   H 
H9      H   C*  C*  C*  C*  H   H   H   H   H 
H10     C*  C*  C*  C*  C*  C*  C*  C*  H   H 
H11     C*  C*  C*  C*  C*  C*  C*  C*  H   C*
H12     H   H   H*  C*  H*  H   H   H   H   H 
H13     H*  C*  C*  C*  C*  H   H   H   H   H 
H14     C*  C*  C*  C*  C*  H   H   H   H   H 
H15     C*  C*  C*  C*  C*  H   H   H   H   H 
H16     C*  C*  C*  C*  C*  H   H   H   H   H 
H17     C*  C*  C*  C*  C*  S   S   S   S   C*
H18     S   S   S   C*  C*  C*  S   S   S   S 
H19     S   S   S   S   S   S   S   S   S   S 
H20     S   S   S   S   S   S   S   S   S   S 
S13     H   H   H   H   H*  H   H   H   H   H 
S14     H   H   H   H*  C*  H   H   H   H   H 
S15     H   H   H   C*  C*  H   H   H   H   H 
S16     H   H   H*  C*  C*  H   H   H   H   H 
S17     H   C*  C*  C*  C*  H   H   H   H   H 
S18     S   C*  C*  C*  C*  C*  S   H   H   H 
S19     S   S   S   S   S   S   S   S   S   S 
S20     S   S   S   S   S   S   S   S   S   S 
P22     P   P   P   P   P   P   H   H   H   H 
P33     H*  P   P   C*  P   P   H   H   H   H 
P44     H   C*  H   C*  C*  H   H   H   H   H 
P55     C*  C*  C*  C*  C*  C*  C*  C*  H   C*
P66     P   C*  C*  C*  C*  H   H   H   H   H 
P77     C*  C*  C*  C*  C*  P   H   H   H   H 
P88     C*  C*  C*  C*  C*  P   P   P   H   P 
P99     C*  C*  C*  C*  C*  C*  C*  P   S   S 
PTT     S   S   S   S   S   S   S   S   S   S 
PAA     C*  C*  C*  C*  C*  C*  C*  C*  C*  C*
```

Strategy chart at TC +0 (3 decks unseen; `*` = differs from basic strategy, C = cash out)

```
         2   3   4   5   6   7   8   9   T   A
H5      H   H   H   H   H   H   H   H   H   H 
H6      H   H   H   H   H   H   H   H   H   H 
H7      H   H   H   H   H   H   H   H   H   H 
H8      H   H   H   H   H   H   H   H   H   H 
H9      H   D   D   D   D   H   H   H   H   H 
H10     D   D   D   D   D   D   D   D   H   H 
H11     D   D   D   D   D   D   D   D   H   H 
H12     H   H   H*  S   S   H   H   H   H   H 
H13     S   S   S   S   S   H   H   H   H   H 
H14     S   S   S   S   S   H   H   H   H   H 
H15     S   S   S   S   S   H   H   H   H   H 
H16     S   S   S   S   S   H   H   H   H   H 
H17     S   S   S   S   S   S   S   S   S   S 
H18     S   S   S   S   S   S   S   S   S   S 
H19     S   S   S   S   S   S   S   S   S   S 
H20     S   S   S   S   S   S   S   S   S   S 
S13     H   H   H   H   D   H   H   H   H   H 
S14     H   H   H   D   D   H   H   H   H   H 
S15     H   H   H   D   D   H   H   H   H   H 
S16     H   H   D   D   D   H   H   H   H   H 
S17     H   D   D   D   D   H   H   H   H   H 
S18     S   D   D   D   D   S   S   H   H   H 
S19     S   S   S   S   S   S   S   S   S   S 
S20     S   S   S   S   S   S   S   S   S   S 
P22     P   P   P   P   P   P   H   H   H   H 
P33     P   P   P   P   P   P   H   H   H   H 
P44     H   H   H   P   P   H   H   H   H   H 
P55     D   D   D   D   D   D   D   D   H   H 
P66     P   P   P   P   P   H   H   H   H   H 
P77     P   P   P   P   P   P   H   H   H   H 
P88     P   P   P   P   P   P   P   P   S*  P 
P99     P   P   P   P   P   S   P   P   S   S 
PTT     S   S   S   S   S   S   S   S   S   S 
PAA     P   P   P   P   P   P   P   P   P   P 
```

Strategy chart at TC +3 (3 decks unseen; `*` = differs from basic strategy, C = cash out)

```
         2   3   4   5   6   7   8   9   T   A
H5      H   H   H   H   H   H   H   H   C*  C*
H6      H   H   H   H   H   C*  C*  H   C*  C*
H7      H   H   H   H   H   H   H   H   H   H 
H8      H   H   H   H   D*  H   H   H   H   H 
H9      D*  D   D   D   D   H   H   H   H   H 
H10     D   D   D   D   D   D   D   D   H   H 
H11     D   D   D   D   D   D   D   D   H   D*
H12     S*  S*  S   S   S   C*  H   C*  C*  C*
H13     S   S   S   S   S   C*  C*  C*  C*  C*
H14     S   S   S   S   S   C*  C*  C*  C*  C*
H15     S   S   S   S   S   C*  C*  C*  C*  C*
H16     S   S   S   S   S   C*  C*  C*  C*  C*
H17     S   S   S   S   S   S   C*  C*  C*  S 
H18     S   S   S   S   S   S   S   C*  C*  S 
H19     S   S   S   S   S   S   S   S   C*  S 
H20     S   S   S   S   S   S   S   S   S   S 
S13     H   H   H   D*  D   H   C*  C*  C*  H 
S14     H   H   D*  D   D   C*  C*  C*  C*  H 
S15     H   H   D*  D   D   C*  C*  C*  C*  C*
S16     H   H   D   D   D   C*  C*  C*  C*  C*
S17     D*  D   D   D   D   H   C*  C*  C*  H 
S18     D*  D   D   D   D   S   S   H   H   H 
S19     S   S   S   D*  D*  S   S   S   C*  S 
S20     S   S   S   S   S   S   S   S   S   S 
P22     P   P   P   P   P   P   H   H   H   H 
P33     P   P   P   P   P   P   P*  H   H   H 
P44     H   H   H   P   P   H   H   H   H   H 
P55     D   D   D   D   D   D   D   D   H   H 
P66     P   P   P   P   P   C*  C*  C*  C*  C*
P77     P   P   P   P   P   P   C*  C*  C*  C*
P88     P   P   P   P   P   P   P   P   C*  P 
P99     P   P   P   P   P   P*  P   P   C*  P*
PTT     S   S   S   S   S   S   S   S   S   S 
PAA     P   P   P   P   P   P   P   P   P   P 
```

Bet head vs exact composition EVs (20,000 pre-deal states, natural shoes; `main` vs the EoR estimate for basic strategy)

| output | mean predicted | mean exact | corr | MAE | P(pred > 0) | P(exact > 0) |
|---|---|---|---|---|---|---|
| main | +0.0003 | -0.0038 | 0.864 | 0.0045 | 38.37% | 29.78% |
| perfect_pairs | -0.0410 | -0.0402 | 0.675 | 0.0072 | 1.29% | 2.33% |
| twenty_one_plus_three | -0.0365 | -0.0371 | 0.721 | 0.0071 | 1.69% | 2.57% |
| crazy_7 | -0.0559 | -0.0575 | 0.993 | 0.0179 | 33.93% | 33.68% |
| bust_bonus | -0.0567 | -0.0564 | 0.981 | 0.0078 | 12.88% | 12.57% |
| insurance | -0.0760 | -0.0760 | 0.999 | 0.0019 | 9.47% | 8.81% |

#### Most frequent disagreements with basic strategy — nn_v1:flat_nocash

| decision cell: NN action (basic) | per 1000 rounds | mean TC |
|---|---|---|
| H16vT/3c: stand (basic hit) | 7.49 | +0.60 |
| H16vT: stand (basic hit) | 6.94 | +0.77 |
| H12v4: hit (basic stand) | 3.15 | -1.23 |
| H13v2: hit (basic stand) | 2.28 | -1.26 |
| H12v6: hit (basic stand) | 2.23 | -1.41 |
| H11vT: double (basic hit) | 2.05 | +1.01 |
| H12v3: stand (basic hit) | 1.86 | +1.90 |
| H12v5: hit (basic stand) | 1.60 | -2.03 |
| H13v3: hit (basic stand) | 1.30 | -1.93 |
| H16vT/4c: stand (basic hit) | 1.25 | +1.00 |
| H9v2: double (basic hit) | 1.00 | +2.00 |
| H12v2: stand (basic hit) | 1.00 | +2.42 |
| H15vT: stand (basic hit) | 0.92 | +3.33 |
| P88vT: stand (basic hit) | 0.75 | +0.74 |
| H9v3: hit (basic double) | 0.73 | -1.91 |

#### Most frequent disagreements with basic strategy — nn_v1:flat

| decision cell: NN action (basic) | per 1000 rounds | mean TC |
|---|---|---|
| H15vT: cashout (basic hit) | 6.94 | +1.67 |
| H13vT: cashout (basic hit) | 6.33 | +1.39 |
| H12vT: cashout (basic hit) | 5.49 | +1.48 |
| H14vT: cashout (basic hit) | 5.19 | +1.59 |
| H16vT: stand (basic hit) | 4.30 | -0.02 |
| H16vT/3c: stand (basic hit) | 4.06 | -0.34 |
| H16vT: cashout (basic hit) | 3.61 | +1.91 |
| H12v4: hit (basic stand) | 2.85 | -1.22 |
| PTTvT: cashout (basic stand) | 2.85 | +2.14 |
| H15vT/3c: cashout (basic hit) | 2.83 | +1.43 |
| H14vT/3c: cashout (basic hit) | 2.73 | +0.96 |
| H13v7: cashout (basic hit) | 2.37 | +1.18 |
| H16vT/3c: cashout (basic hit) | 2.32 | +1.39 |
| H18vT: cashout (basic stand) | 2.25 | +2.19 |
| H19vT: cashout (basic stand) | 2.20 | +2.36 |

#### Most frequent disagreements with basic strategy — nn_v1:nosides

| decision cell: NN action (basic) | per 1000 rounds | mean TC |
|---|---|---|
| H15vT: cashout (basic hit) | 6.85 | +1.67 |
| H13vT: cashout (basic hit) | 6.33 | +1.39 |
| H12vT: cashout (basic hit) | 5.51 | +1.49 |
| H14vT: cashout (basic hit) | 5.40 | +1.59 |
| H16vT: stand (basic hit) | 4.62 | +0.11 |
| H16vT/3c: stand (basic hit) | 4.19 | -0.28 |
| H16vT: cashout (basic hit) | 3.22 | +1.91 |
| H12v4: hit (basic stand) | 2.85 | -1.22 |
| H15vT/3c: cashout (basic hit) | 2.80 | +1.43 |
| H17vT: cashout (basic stand) | 2.65 | +2.36 |
| H14vT/3c: cashout (basic hit) | 2.65 | +0.93 |
| H18vT: cashout (basic stand) | 2.53 | +2.08 |
| PTTvT: cashout (basic stand) | 2.51 | +2.20 |
| H13v7: cashout (basic hit) | 2.32 | +1.16 |
| H19vT: cashout (basic stand) | 2.29 | +2.33 |

#### Most frequent disagreements with basic strategy — nn_v1

| decision cell: NN action (basic) | per 1000 rounds | mean TC |
|---|---|---|
| H15vT: cashout (basic hit) | 6.85 | +1.67 |
| H13vT: cashout (basic hit) | 6.33 | +1.39 |
| H12vT: cashout (basic hit) | 5.51 | +1.49 |
| H14vT: cashout (basic hit) | 5.40 | +1.59 |
| H16vT: stand (basic hit) | 4.62 | +0.11 |
| H16vT/3c: stand (basic hit) | 4.19 | -0.28 |
| H16vT: cashout (basic hit) | 3.22 | +1.91 |
| H12v4: hit (basic stand) | 2.85 | -1.22 |
| H15vT/3c: cashout (basic hit) | 2.80 | +1.43 |
| H17vT: cashout (basic stand) | 2.65 | +2.36 |
| H14vT/3c: cashout (basic hit) | 2.65 | +0.93 |
| H18vT: cashout (basic stand) | 2.53 | +2.08 |
| PTTvT: cashout (basic stand) | 2.51 | +2.20 |
| H13v7: cashout (basic hit) | 2.32 | +1.16 |
| H19vT: cashout (basic stand) | 2.29 | +2.33 |
