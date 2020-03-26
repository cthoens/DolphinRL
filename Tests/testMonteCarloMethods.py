import unittest
import numpy as np
from Environments.CleanBotEnv import CleanBotEnv
from Models.TableModel import TableModel
from Methods.MonteCarloMethods import AveragingMC, ConstAlphaMC
from Methods.Policies import EpsilonGreedyPolicy, GreedyPolicy

SOUTH = CleanBotEnv.BotActions.SOUTH.value
EAST = CleanBotEnv.BotActions.EAST.value
WEST = CleanBotEnv.BotActions.WEST.value
CLEAN = CleanBotEnv.BotActions.CLEAN.value


class MockPolicy(EpsilonGreedyPolicy):
    def __init__(self, actions):
        super().__init__(None, 0)
        self.actions = actions
        self.current = 0

    def choose_action(self, observation):
        action = self.actions[self.current]
        self.current += 1
        return action


class MockEnv(CleanBotEnv):
    def reset(self):
        super().reset()
        self.bot_x = 0
        self.bot_y = 0
        self.dirty_count = 1
        self.state = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int)
        return self._get_obs()


class TestAveragingMonteCarlo(unittest.TestCase):

    def test_policy_update(self):
        np.random.seed(643674)
        initial_state = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int)
        initial_obs = np.array([[2, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int)
        env = MockEnv(3)
        model = TableModel(env)
        greedy_policy = GreedyPolicy(model)

        policy = MockPolicy([EAST, EAST, SOUTH, WEST, WEST, CLEAN])
        mc = AveragingMC(env, model, policy)
        mc.run_episode()

        self.assertEqual(mc.stats.first_time_visited, 6)
        self.assertEqual(mc.stats.fifth_time_visited, 0)
        self.assertEqual(mc.stats.episode_reward, 194)
        self.assertEqual(mc.stats.max_action_value_delta, 0)
        self.assertEqual(greedy_policy.choose_action(initial_obs), EAST)

        mock_policy = MockPolicy([SOUTH, CLEAN])
        mc.policy = mock_policy
        mc.run_episode()

        self.assertEqual(mc.stats.first_time_visited, 7)
        self.assertEqual(mc.stats.fifth_time_visited, 0)
        self.assertEqual(mc.stats.episode_reward, 198)
        self.assertEqual(mc.stats.max_action_value_delta, 2.0)
        self.assertEqual(greedy_policy.choose_action(initial_obs), SOUTH)

    def test_smoke(self):
        np.random.seed(643674)
        env = CleanBotEnv(3)
        model = TableModel(env)
        policy = EpsilonGreedyPolicy(model, 0.1)
        mc = AveragingMC(env, model, policy)

        policy.exploration = 0.1
        episode_count = 100
        for i in range(episode_count):
            mc.run_episode()


class TestConstAlphaMonteCarlo(unittest.TestCase):

    def test_policy_update(self):
        np.random.seed(643674)
        initial_state = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int)
        initial_obs = np.array([[2, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int)
        env = MockEnv(3)
        model = TableModel(env)
        greedy_policy = GreedyPolicy(model)

        policy = MockPolicy([EAST, EAST, SOUTH, WEST, WEST, CLEAN])
        mc = ConstAlphaMC(env, model, policy)
        mc.alpha = 0.005
        mc.run_episode()

        self.assertEqual(mc.stats.episode_reward, 194)
        self.assertAlmostEqual(mc.stats.max_action_value_delta, 0.97)
        self.assertEqual(greedy_policy.choose_action(initial_obs), EAST)

        policy = MockPolicy([EAST, EAST, SOUTH, WEST, WEST, CLEAN])
        mc = ConstAlphaMC(env, model, policy)
        mc.run_episode()

        self.assertEqual(mc.stats.episode_reward, 194)
        self.assertAlmostEqual(mc.stats.max_action_value_delta, 0.9651499998569488)
        self.assertEqual(greedy_policy.choose_action(initial_obs), EAST)

        mock_policy = MockPolicy([SOUTH, CLEAN])
        mc.policy = mock_policy
        mc.run_episode()

        self.assertEqual(mc.stats.episode_reward, 198)
        self.assertEqual(mc.stats.max_action_value_delta, 0.99)
        self.assertEqual(greedy_policy.choose_action(initial_obs), EAST)

        mock_policy = MockPolicy([SOUTH, CLEAN])
        mc.policy = mock_policy
        mc.run_episode()

        self.assertEqual(mc.stats.episode_reward, 198)
        self.assertEqual(mc.stats.max_action_value_delta, 0.9850499999523163)
        self.assertEqual(greedy_policy.choose_action(initial_obs), SOUTH)


if __name__ == "__main__":
    unittest.main()
