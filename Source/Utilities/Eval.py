"""
Utilities.Env
============

Utility functions for evaluating methods
"""

import numpy as np
from gym import Env
from Policies import Policy


class StatsLogger:
    """
    Logs performance statistics by collecting the values of all attributes of a given instance. Returns as numpy
    array that can be used efficiently with mathplotlib. Buffers have a maximum length. When this length is reached
    the oldest values get rotated out of the buffer when now values are added.
    """
    def __init__(self, stats, max_length=100000):

        self.count = 0
        """The number of elements in any array returned by data. Between 0 and max_length"""

        self.lower_bound = 0
        """
        The index of the fist element returned by data 

        Example for max_length = 10:
            Append called 5 times => 0
            Append called 10 times => 0
            Append called 15 times => 5
            Append called 20 times => 10
            Append called 25 times => 15
        """
        self.upper_bound = -1
        """The index of the last element returned by data. One less than the number of times append was called."""
        self.max_length = max_length
        self.data = {key: [] for key in stats.__dict__.keys()}
        "Dict: each value in stat -> numpy array of values that have been collected"
        self.max = {key: float("-inf") for key in stats.__dict__.keys()}
        "Dict: each value in stat -> maximum value ever collected"
        self.min = {key: float("inf") for key in stats.__dict__.keys()}
        "Dict: each value in stat -> minimum value ever collected"

        self._rollover_count = 0
        self._next_index = 0
        """Next index in data to write to"""
        self._data = {key: np.zeros(2 * max_length) for key in stats.__dict__.keys()}

    def append(self, stats):
        if self._next_index == 2 * self.max_length:
            self._next_index = 0
            self._rollover_count += 1
            for stat in self._data.values():
                np.roll(stat, self.max_length)

        self.count = min(self.max_length, self.count + 1)
        self.upper_bound += 1
        for key, value in stats.__dict__.items():
            self.min[key] = min(self.min[key], value)
            self.max[key] = max(self.max[key], value)
            self._data[key][self._next_index] = value
            start_index = max(0, self._next_index + 1 - self.max_length)
            self.data[key] = self._data[key][start_index: start_index + self.count]
            self.lower_bound = max(0, self.upper_bound + 1 - self.max_length)
        self._next_index += 1


class TestingStats:
    def __init__(self):
        self.avg_reward = 0.0


def test_policy(env: Env, policy: Policy, episode_count=1000, random_seed=52346, max_steps=10000) -> float:
    """
    Return the average reward received after evaluating the policy episode_count times.

    Preserves the state of the random number generator.
    """
    random_number_generator_state = np.random.get_state()
    np.random.seed(random_seed)
    total_reward = 0.0
    try:
        for i in range(episode_count):
            obs = env.reset()
            done = False
            for step in range(max_steps):
                action = policy.choose_action(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
            if not done:
                raise Exception("Episode did not terminate")
    finally:
        np.random.set_state(random_number_generator_state)

    return total_reward / episode_count
