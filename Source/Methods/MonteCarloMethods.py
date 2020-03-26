"""
MonteCarloMethods
=================

Provides varieties of monte carlo methods.
"""

import numpy as np
import Utilities.Env as envutil
from Models.Model import Model
from Methods.Policies import EpsilonGreedyPolicy


class AveragingMcStats:
    def __init__(self):
        self.max_action_value_delta = None
        """
        Maximum delta of a state-action values in the last episode of state-action pairs that have been visited 
        more than once
        """

        self.episode_reward = None
        """Total reward of the last episode"""

        self.first_time_visited = 0
        """Number of states that have been visited at least once"""

        self.fifth_time_visited = 0
        """Number of states that have been visited at least five times"""


class AveragingMC:
    """
    A monte carlo method in which each state-action value is determined by the average first-visit reward across all
    observed episodes.
    """

    def __init__(self, env, model: Model, policy: EpsilonGreedyPolicy):
        self.env = env
        self.model = model
        self.policy = policy
        self.total_returns = np.zeros(envutil.obs_action_shape(env), dtype=np.int32)
        self.visit_count = np.zeros(envutil.obs_action_shape(env), dtype=np.int32)
        self.stats = AveragingMcStats()

    def run_episode(self):
        episode = envutil.record_episode(self.env, self.policy)
        episode_reward, total_reward = envutil.first_visit_rewards(self.env, episode)
        max_delta = 0
        for obs_action, reward in episode_reward.items():
            # Integrate new data
            self.total_returns[obs_action] += reward
            self.visit_count[obs_action] += 1

            # Calculate model update
            visit_count = self.visit_count[obs_action]
            updated_action_value = self.total_returns[obs_action] / visit_count

            # Calculate stats
            if visit_count == 1:
                self.stats.first_time_visited += 1
            else:
                max_delta = max(max_delta, abs(self.model.action_values(obs_action) - updated_action_value))
            if visit_count == 5:
                self.stats.fifth_time_visited += 1

            # Update the model
            self.model.update_action_value(obs_action, updated_action_value)
        self.stats.episode_reward = total_reward
        self.stats.max_action_value_delta = max_delta
        return total_reward


class ConstAlphaMCStats:
    def __init__(self):
        self.max_action_value_delta = None
        """
        Maximum delta of a state-action values in the last episode of state-action pairs that have been visited 
        more than once
        """

        self.episode_reward = None
        """Total reward of the last episode"""


class ConstAlphaMC:
    """
    Monte carlo method using the update

    model[observation, action] = alpha * (first_visit_reward - model[observation, action]])
    """

    def __init__(self, env, model: Model, policy: EpsilonGreedyPolicy):
        self.env = env
        self.model = model
        self.policy = policy
        self.alpha = 0.005
        self.stats = ConstAlphaMCStats()

    def run_episode(self):
        episode = envutil.record_episode(self.env, self.policy)
        episode_reward, total_reward = envutil.first_visit_rewards(self.env, episode)
        max_delta = 0
        for obs_action, reward in episode_reward.items():
            current_action_value = self.model.action_values(obs_action)
            action_value_delta = self.alpha * (reward - current_action_value)
            max_delta = max(max_delta, action_value_delta)
            self.model.update_action_value(obs_action, current_action_value + action_value_delta)
        self.stats.episode_reward = total_reward
        self.stats.max_action_value_delta = max_delta
        return total_reward
