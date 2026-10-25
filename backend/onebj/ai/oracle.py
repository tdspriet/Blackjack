"""Composition-dependent "oracle" play: dynamic programming on the current unseen composition.

:class:`CashSolver` extends the engine's infinite-deck solver (:mod:`onebj.strategy.solver`, the machinery
that derives basic strategy, the Hi-Lo indices and the cash-out fair values) with the **cash-out option
at every later decision**: after a hit the hand may be cashed out for ``cashout_rtp × fresh-shoe fair
value`` (the offer ignores the composition), and so may each split hand at its first decision. The
plain solver assumes you never cash out later, which undervalues hitting/splitting whenever the
composition makes future cash-outs attractive.

:class:`SolverAgent` plays argmax over {stand, hit, double, split} EVs from that solver at the *current*
composition and the exact current cash-out EV. It is not a neural network — it is a strong, slow
(~0.1-1 ms per decision) reference player used

* in :mod:`onebj.ai.evaluate` as agents ``solver`` (cash-aware, flat bet), ``solver_nocash`` (never
  cashes out) to show what composition-dependent play and the cash-out option are worth;
* as the optional *teacher* of the play head (``targets: "solver"`` / ``"blend"`` in :mod:`onebj.ai.datagen`);
* in :mod:`onebj.ai.analysis` for the regret benchmark.

Only public information is used (seen cards, the hand, the up card, the cash-out offer).
"""

from __future__ import annotations

from functools import lru_cache

from ..config import GameConfig, load_config
from ..engine import Bets, Table
from ..strategy.cashout import CashoutTable, get_cashout_table
from ..strategy.solver import Solver, SolverRules, _Values

_CODES = (("S", "stand"), ("H", "hit"), ("D", "double"), ("P", "split"))


class _CashValues(_Values):
    """Hit/stand values where every later decision may also cash out (except at 21: auto-stand)."""

    def __init__(self, solver: CashSolver, up: int, loss1: float, loss2: float, split_hand: bool) -> None:
        super().__init__(solver, up, loss1, loss2)
        self.up = up
        self.split_hand = split_hand
        self.cash = solver.cash_table

    def best(self, t: int, soft: bool, n: int) -> float:
        key = (t, soft, n if self.s.rules.charlie else 0)
        v = self.memo.get(key)
        if v is None:
            if t == 21:
                v = self.stand[t]
            else:
                c = self.cash.offer(t, soft, n, self.up, False, -1, self.split_hand) - 1.0
                v = max(self.stand[t], self.hit(t, soft, n), c)
            self.memo[key] = v
        return v


class CashSolver(Solver):
    """:class:`~onebj.strategy.solver.Solver` with cash-out options at later decisions (see module doc)."""

    def __init__(self, rules: SolverRules, p, cash_table: CashoutTable | None) -> None:
        super().__init__(rules, p)
        self.cash_table = cash_table

    def _prepare(self, u: int) -> None:
        super()._prepare(u)
        if self.cash_table is not None:
            late_all = self.rules.late_all
            self._main[u] = _CashValues(self, u, 1.0, 2.0 if late_all else 1.0, False)
            self._split[u] = _CashValues(self, u, 1.0 if late_all else 0.5, 2.0 if late_all else 0.5, True)

    def split_ev(self, x: int, u: int) -> float:
        V = self.values(u, True)
        p = self.p
        r = self.rules
        ct = self.cash_table
        ev = 0.0
        for c in range(10):
            hard = x + c + 2
            ace = x == 0 or c == 0
            soft = ace and hard <= 11
            t = hard + 10 if soft else hard
            if (x == 0 and r.split_aces_one_card) or t == 21:
                v = V.stand[t]
            else:
                v = max(V.stand[t], V.hit(t, soft, 2))
                can_d = r.das and self.can_double(t, soft)
                if can_d:
                    v = max(v, V.double(t, soft))
                if ct is not None:
                    v = max(v, ct.offer(t, soft, 2, u, can_d, -1, True) - 1.0)
            ev += p[c] * v
        return 2 * ev


@lru_cache(maxsize=2048)
def composition_solver(rules: SolverRules, buckets: tuple[int, ...], cash_rtp: float | None) -> Solver:
    """Solver for an unseen bucket composition (``cash_rtp`` None → plain solver without cash-outs)."""
    tot = sum(buckets)
    p = tuple(x / tot for x in buckets)
    if cash_rtp is None:
        return Solver(rules, p)
    return CashSolver(rules, p, _cash_table(rules, cash_rtp))


@lru_cache(maxsize=8)
def _cash_table(rules: SolverRules, rtp: float) -> CashoutTable:
    return CashoutTable(rules, rtp)


def state_evs(solver: Solver, state: tuple) -> dict[str, float]:
    """EVs of the legal non-cash-out actions for a ``Table.decision_state()`` tuple."""
    total, soft, n, up, can_double, pair, can_split, is_split = state
    e = solver.evs(total, soft, n, up, pair if can_split else -1, can_double, can_split, is_split)
    return {name: e[c] for c, name in _CODES if c in e}


class SolverAgent:
    """Flat minimum bet, no side bets, no insurance; play = argmax of composition-dependent solver EVs and
    the exact cash-out EV. ``cashout=False``: plain solver, never cashes out."""

    def __init__(self, config: GameConfig | None = None, cashout: bool = True, name: str = "solver") -> None:
        self.cfg = config or load_config()
        self.rules = SolverRules.from_config(self.cfg.rules)
        self.cashout = bool(cashout and self.cfg.rules.cashout)
        self.rtp = self.cfg.rules.cashout_rtp if self.cashout else None
        self.name = name
        self.unit = self.cfg.limits.main_min
        get_cashout_table(self.cfg.rules)

    def fast_bets(self, table: Table) -> Bets:
        return Bets(main=self.unit)

    def fast_insurance(self, table: Table) -> bool:
        return False

    def action_evs(self, table: Table, state: tuple) -> dict[str, float]:
        s = composition_solver(self.rules, tuple(table.shoe.unseen_buckets()), self.rtp)
        out = state_evs(s, state)
        if self.cashout:
            off = table.cashout_offer()
            if off is not None:
                out["cashout"] = off / table.hands[table.active].bet - 1.0
        return out

    def fast_action(self, table: Table, state: tuple) -> str:
        e = self.action_evs(table, state)
        return max(e, key=e.__getitem__)

    def action(self, table: Table):  # Policy protocol (datagen)
        from ..engine import Action
        return Action(self.fast_action(table, table.decision_state()))

    def decide_bets(self, obs):  # pragma: no cover - evaluators use the fast path
        return Bets(main=self.unit), None

    def decide_insurance(self, obs):  # pragma: no cover
        return False, None
