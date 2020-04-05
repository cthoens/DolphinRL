"""
MonteCarloMethods
=================

Provides varieties of monte carlo methods.
"""

import numpy as np
from math import sqrt

import Utilities.Env as envutil
from Model import Model
from Policies import EpsilonGreedyPolicy


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
        first_visit_rewards, total_reward = envutil.first_visit_rewards(episode)
        max_delta = 0

        for state, action, reward in first_visit_rewards:
            state_action_index = envutil.to_table_index(state, action)
            # Integrate new data
            self.total_returns[state_action_index] += reward
            self.visit_count[state_action_index] += 1

            # Calculate model update
            visit_count = self.visit_count[state_action_index]
            updated_action_value = self.total_returns[state_action_index] / visit_count

            # Calculate stats
            if visit_count == 1:
                self.stats.first_time_visited += 1
            else:
                max_delta = max(max_delta, abs(self.model.action_value(state, action) - updated_action_value))
            if visit_count == 5:
                self.stats.fifth_time_visited += 1

            # Update the model
            self.model.update_action_value(state, action, updated_action_value)
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

        self.rms = None
        """The root mean square error"""


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
        first_visit_rewards, total_reward = envutil.first_visit_rewards(episode)
        max_delta = 0
        squared_residuals = 0
        for state, action, observed_reward in first_visit_rewards:
            predicted_reward = self.model.action_value(state, action)
            action_value_delta = self.alpha * (observed_reward - predicted_reward)
            squared_residuals += (observed_reward - predicted_reward)**2
            max_delta = max(max_delta, abs(action_value_delta))
            self.model.update_action_value(state, action, predicted_reward + action_value_delta)
        self.stats.episode_reward = total_reward
        self.stats.max_action_value_delta = max_delta
        self.stats.rms = sqrt(squared_residuals / len(first_visit_rewards))
        return total_reward
