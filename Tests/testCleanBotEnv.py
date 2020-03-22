import numpy as np
from Environments.CleanBotEnv import CleanBotEnv
import unittest


class TestCleanBotEnv(unittest.TestCase):

    def test_interaction(self):
        """Performs a series of interaction with the environment"""
        # Setup
        def check_state(bot_pos, dirty=False, reward=0, done=False):
            _state, _reward, _done, _ = response
            self.assertEqual(DIRTY if dirty else BOT, _state[bot_pos])
            self.assertEqual(_reward, reward)
            self.assertEqual(_done, done)

        BOT = CleanBotEnv.TileState.BOT.value
        DIRTY = CleanBotEnv.TileState.DIRTY.value
        np.random.seed(34567876)
        env = CleanBotEnv(5, dirty_rate=0.1)
        env.state = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        env.dirty_count = 2
        response = env.step(CleanBotEnv.BotActions.EAST.value)
        check_state((0, 1))
        response = env.step(CleanBotEnv.BotActions.EAST.value)
        check_state((0, 2))
        response = env.step(CleanBotEnv.BotActions.SOUTH.value)
        check_state((1, 2), dirty=True)
        response = env.step(CleanBotEnv.BotActions.CLEAN.value)
        check_state((1, 2), reward=196)

        response = env.step(CleanBotEnv.BotActions.EAST.value)
        check_state((1, 3))
        response = env.step(CleanBotEnv.BotActions.EAST.value)
        check_state((1,4))
        response = env.step(CleanBotEnv.BotActions.SOUTH.value)
        check_state((2, 4), dirty=True)
        response = env.step(CleanBotEnv.BotActions.CLEAN.value)
        check_state((2, 4), reward=192, done=True)


if __name__ == '__main__':
    unittest.main()
