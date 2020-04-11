import numpy as np
from CleanBotEnv import CleanBotEnv
from Policies import EpsilonGreedyPolicy


class MockPolicy(EpsilonGreedyPolicy):
    """
    Policy that chooses a given static sequence of actions
    """
    def __init__(self, actions):
        super().__init__(None, 0)
        self.actions = actions
        self.current = 0

    def choose_action(self, observation):
        action = self.actions[self.current]
        self.current += 1
        return action


class MockEnv(CleanBotEnv):
    """
    Environment that always starts in the same state
    """
    def reset(self):
        super().reset()
        self.bot_x = 0
        self.bot_y = 0
        self.dirty_count = 1
        self.state = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int)
        return self._get_obs()
