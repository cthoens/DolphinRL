"""
Utilities.Env
============

Utility functions for dealing with environments
"""

from gym import Env
from typing import List, Tuple, Dict
from Methods.Policies import Policy
import numpy as np


class Interaction:
    """Represent a single interaction with the environment"""
    def __init__(self, obs, action, ret):
        self.obs = obs
        self.action = action
        self.reward = ret


def record_episode(env: Env, policy: Policy, max_steps=10000) -> List[Interaction]:
    """
    Record a sequence of interactions with the environment until the episode terminates or a maximum number of
    steps is reached.

    :returns:
        The sequence of interactions as a List[:class Interaction:].
    """
    episode = []
    obs = env.reset()

    for i in range(max_steps):
        action = policy.choose_action(obs)
        next_obs, reward, done, _ = env.step(action)
        episode.append(Interaction(np.ravel(obs), action, reward))
        if done:
            return episode
        obs = next_obs
    raise Exception("Episode did not terminate.")


def obs_action_shape(env):
    """Return the shape of an array that can hold all possible state-action pairs of an environment"""
    obs_space = env.observation_space
    return np.append(np.ravel(obs_space.high)+1, [env.action_space.n])


def to_table_index(obs, action=None):
    if action is not None:
        return tuple(np.ravel(obs)) + (action, )
    else:
        return tuple(np.ravel(obs))


def first_visit_rewards(episode: List[Interaction]) -> Tuple[Dict[tuple, float], float]:
    """
    For each state-action pair visited in the episode return the total reward after the first visit

    :returns:
        rewards, total_reward:
            reward: Dict[obs_action] -> first visit reward
            total: The sum of reward of the episode
    """
    rewards = {}
    total_reward = 0
    for interaction in reversed(episode):
        total_reward += interaction.reward
        rewards[(tuple(interaction.obs), interaction.action)] = total_reward
    return rewards, total_reward
