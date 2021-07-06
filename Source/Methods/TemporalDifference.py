from Model import Model
from Policies import EpsilonGreedyPolicy


class SarsaMetrics:
    def __init__(self):
        self.max_action_value_delta = None
        """
        Maximum delta of a state-action values in the last episode of state-action pairs that have been visited 
        more than once
        """

        self.episode_reward = None
        """Total reward of the last episode"""


class Sarsa:
    """
    Monte carlo method using the update

    model[observation, action] = alpha * (first_visit_reward - model[observation, action]])
    """

    def __init__(self, env, model: Model, policy: EpsilonGreedyPolicy, alpha=0.01, gamma=0.9):
        self.env = env
        self.model = model
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma
        self.metrics = SarsaMetrics()

    async def run_episode(self):
        state_0 = await self.env.reset()
        self.metrics.episode_reward = 0

        max_delta = 0
        for step in range(1000):
            action_0 = self.policy.choose_action(state_0)
            state_1, reward, done, _ = await self.env.step(action_0)
            action_1 = self.policy.choose_action(state_1)

            state_action_0_value = self.model.action_value(state_0, action_0)
            state_action_1_value = self.model.action_value(state_1, action_1)
            action_value_delta = self.alpha * (reward + self.gamma * state_action_1_value - state_action_0_value)
            self.model.update_action_value(state_0, action_0, state_action_0_value + action_value_delta)
            state_0 = state_1

            max_delta = max(max_delta, abs(action_value_delta))
            self.metrics.episode_reward += reward
            if done:
                break

        self.metrics.max_action_value_delta = max_delta
        return self.metrics.episode_reward
