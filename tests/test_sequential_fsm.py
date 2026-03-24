"""Tests for framework.sequential_fsm."""

from __future__ import annotations

import unittest
from enum import Enum, auto

from framework.sequential_fsm import SequentialSetup, StepResult


class S(Enum):
    A = auto()
    B = auto()
    C = auto()


class SequentialFsmTests(unittest.TestCase):
    def test_invalidator_resets_when_in_from_states(self) -> None:
        fsm: SequentialSetup[S] = SequentialSetup(S.A)
        fsm.add_transition(S.A, lambda _: True, S.B)
        fsm.add_invalidator(lambda _: True, from_states={S.B})

        r1 = fsm.step({})
        self.assertEqual(r1.state, S.B)
        self.assertTrue(r1.transitioned)
        self.assertFalse(r1.invalidated)

        r2 = fsm.step({})
        self.assertEqual(r2.state, S.A)
        self.assertTrue(r2.invalidated)
        self.assertFalse(r2.transitioned)

    def test_invalidator_skips_when_state_not_in_from_states(self) -> None:
        fsm: SequentialSetup[S] = SequentialSetup(S.A)
        fsm.add_transition(S.A, lambda _: True, S.B)
        fsm.add_invalidator(lambda _: True, from_states={S.C})

        fsm.step({})
        r = fsm.step({})
        self.assertEqual(r.state, S.B)
        self.assertFalse(r.invalidated)

    def test_invalidators_run_before_transitions(self) -> None:
        fsm: SequentialSetup[S] = SequentialSetup(S.A)
        fsm.add_invalidator(lambda _: True, from_states={S.A})
        fsm.add_transition(S.A, lambda _: True, S.B)

        r = fsm.step({})
        self.assertEqual(r.state, S.A)
        self.assertTrue(r.invalidated)

    def test_first_registered_transition_wins(self) -> None:
        fsm: SequentialSetup[S] = SequentialSetup(S.A)
        fsm.add_transition(S.A, lambda _: True, S.B)
        fsm.add_transition(S.A, lambda _: True, S.C)

        r = fsm.step({})
        self.assertEqual(r.state, S.B)
        self.assertTrue(r.transitioned)

    def test_reset_returns_to_initial(self) -> None:
        fsm: SequentialSetup[S] = SequentialSetup(S.A)
        fsm.add_transition(S.A, lambda _: True, S.B)
        fsm.step({})
        self.assertEqual(fsm.state, S.B)
        fsm.reset()
        self.assertEqual(fsm.state, S.A)

    def test_meta_on_transition(self) -> None:
        fsm: SequentialSetup[S] = SequentialSetup(S.A)
        fsm.add_transition(S.A, lambda _: True, S.B, meta="tag")

        r = fsm.step({})
        self.assertEqual(r.meta, "tag")
        self.assertIsInstance(r, StepResult)


if __name__ == "__main__":
    unittest.main()
