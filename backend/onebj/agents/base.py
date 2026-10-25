"""Agent protocol and the JSON-able decision ``Trace`` rendered by the NN panel (SPEC §5)."""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from ..engine import Action, Bets, Observation


class TraceFeature(TypedDict, total=False):
    name: str
    value: float
    raw: Any


class TraceLayer(TypedDict):
    name: str
    size: int
    activations: list[float]


class TraceOutputs(TypedDict, total=False):
    labels: list[str]
    values: list[float | None]
    legal: list[bool]
    unit: str


class TraceReference(TypedDict, total=False):
    basic_strategy: str | None
    hilo_deviation: str | None


class Trace(TypedDict, total=False):
    kind: Literal["play", "bet", "insurance"]
    features: list[TraceFeature]
    layers: list[TraceLayer]
    outputs: TraceOutputs
    chosen: str
    reference: TraceReference
    extra: dict[str, Any]


@runtime_checkable
class Agent(Protocol):
    """Anything that can play a round. Return ``(decision, trace_or_None)`` from each method."""

    name: str

    def decide_bets(self, obs: Observation) -> tuple[Bets, Trace | None]: ...

    def decide_insurance(self, obs: Observation) -> tuple[bool, Trace | None]: ...

    def decide_action(self, obs: Observation) -> tuple[Action, Trace | None]: ...
