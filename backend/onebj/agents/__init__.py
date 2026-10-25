"""Agents that play a :class:`onebj.engine.Table` (SPEC §5)."""

from .base import Agent, Trace
from .basic import BasicStrategyAgent, HiLoAgent

__all__ = ["Agent", "Trace", "BasicStrategyAgent", "HiLoAgent"]
