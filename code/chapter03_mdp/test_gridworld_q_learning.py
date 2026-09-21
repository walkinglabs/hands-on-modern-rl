"""Regression tests for the GridWorld terminal-reward convention."""

import unittest

import gridworld_q_learning as gridworld


class TerminalRewardContractTest(unittest.TestCase):
    """Keep terminal rewards on entry and prevent a second bootstrap reward."""

    def test_entering_goal_returns_reward_and_ends_episode(self):
        next_state, reward, done = gridworld.transition((3, 2), 3)

        self.assertEqual(next_state, gridworld.GOAL)
        self.assertEqual(reward, gridworld.GOAL_REWARD)
        self.assertTrue(done)

    def test_entering_trap_returns_penalty_and_ends_episode(self):
        next_state, reward, done = gridworld.transition((1, 0), 3)

        self.assertEqual(next_state, gridworld.TRAP)
        self.assertEqual(reward, gridworld.TRAP_REWARD)
        self.assertTrue(done)

    def test_terminal_states_have_no_successor_reward(self):
        for terminal in gridworld.TERMINALS:
            for action in range(len(gridworld.ACTIONS)):
                with self.subTest(terminal=terminal, action=action):
                    self.assertEqual(
                        gridworld.transition(terminal, action),
                        (terminal, 0.0, True),
                    )

    def test_action_value_does_not_bootstrap_after_termination(self):
        values = {state: 0.0 for state in gridworld.all_states()}
        values[gridworld.GOAL] = 999.0
        values[gridworld.TRAP] = -999.0

        self.assertEqual(gridworld.action_value(values, (3, 2), 3), 1.0)
        self.assertEqual(gridworld.action_value(values, (1, 0), 3), -1.0)

    def test_value_iteration_keeps_terminal_values_at_zero(self):
        values, history = gridworld.value_iteration()

        self.assertEqual(values[gridworld.GOAL], 0.0)
        self.assertEqual(values[gridworld.TRAP], 0.0)
        self.assertEqual(values[(3, 2)], gridworld.GOAL_REWARD)
        self.assertAlmostEqual(values[gridworld.START], 0.728537125)
        self.assertEqual(len(history) - 1, 7)


if __name__ == "__main__":
    unittest.main()
