"""
Sequential setup (finite state machine) for bar-by-bar trading logic
===================================================================

Use this when strategy rules are **ordered steps** with optional **invalidation**
(reset to the initial state). Example: multi-step long entries, staged filters.

**Per bar, evaluation order**

1. **Invalidators** (in registration order): if a rule’s predicate is true **and**
   the current state is allowed (see below), the machine **resets** to ``initial_state``
   and ``step()`` returns with ``invalidated=True``. No transitions run that bar.
2. **Transitions** for the current state (in registration order): first matching
   ``condition(context)`` wins; state updates and optional ``meta`` is attached.

**Invalidator scope**

- ``from_states=None``: predicate may fire from **any** state (including initial).
- ``from_states=some frozenset``: only reset when ``current_state in from_states``.

Typical ``context`` is a small dataclass or namespace passed each bar with OHLC,
indicators, and anything predicates need (keep it explicit and cheap to build).

See ``research/mach4_ema_band_ep1/strategies/ema_band_ep1_strategy.py`` for usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Collection, Generic, Hashable, Optional, TypeVar

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class StepResult(Generic[T]):
    """Result of :meth:`SequentialSetup.step`."""

    state: T
    """State after this bar (may be unchanged)."""

    invalidated: bool = False
    """True if an invalidator fired and the machine reset to ``initial_state``."""

    transitioned: bool = False
    """True if a transition moved to a *different* state (not invalidation)."""

    meta: Any = None
    """Optional payload from the matched transition (e.g. ``\"long_entry\"``)."""


class SequentialSetup(Generic[T]):
    """
    Small explicit FSM for sequential trading setups.

    States must be hashable (``Enum``, ``str``, ``int``, …). Conditions are
    ``Callable[[Any], bool]`` receiving your per-bar ``context`` object.
    """

    def __init__(self, initial_state: T) -> None:
        self._initial = initial_state
        self._state = initial_state
        self._transitions: dict[T, list[tuple[Callable[[Any], bool], T, Any]]] = {}
        self._invalidators: list[tuple[Callable[[Any], bool], Optional[frozenset[T]]]] = []

    @property
    def state(self) -> T:
        return self._state

    @property
    def initial_state(self) -> T:
        return self._initial

    def reset(self) -> None:
        """Return to the initial state (e.g. when flat or after a fill)."""
        self._state = self._initial

    def add_transition(
        self,
        from_state: T,
        condition: Callable[[Any], bool],
        to_state: T,
        *,
        meta: Any = None,
    ) -> SequentialSetup[T]:
        """
        When ``current == from_state`` and ``condition(context)`` is true, move to
        ``to_state``. First registered transition for that ``from_state`` wins.

        ``meta`` is returned on :class:`StepResult` for the strategy to map to
        orders or signals (e.g. entry tag).
        """
        self._transitions.setdefault(from_state, []).append((condition, to_state, meta))
        return self

    def add_invalidator(
        self,
        condition: Callable[[Any], bool],
        from_states: Optional[Collection[T]] = None,
    ) -> SequentialSetup[T]:
        """
        When ``condition(context)`` is true and the current state is allowed, reset
        to ``initial_state``.

        If ``from_states`` is ``None``, any current state may be reset (including
        initial — usually harmless). Otherwise only states in the collection.
        """
        fs: Optional[frozenset[T]] = None
        if from_states is not None:
            fs = frozenset(from_states)
        self._invalidators.append((condition, fs))
        return self

    def step(self, context: Any) -> StepResult[T]:
        """
        Apply invalidators, then transitions for the current state.

        Returns a :class:`StepResult` describing what happened this bar.
        """
        # 1) Invalidators first
        for pred, allowed in self._invalidators:
            try:
                if not pred(context):
                    continue
            except Exception:
                continue
            if allowed is None or self._state in allowed:
                self._state = self._initial
                return StepResult(self._state, invalidated=True, transitioned=False)

        # 2) Transitions from current state (first match wins)
        prev = self._state
        for cond, to_st, meta in self._transitions.get(self._state, ()):
            try:
                if not cond(context):
                    continue
            except Exception:
                continue
            self._state = to_st
            return StepResult(
                self._state,
                invalidated=False,
                transitioned=prev != to_st,
                meta=meta,
            )

        return StepResult(self._state, invalidated=False, transitioned=False)
