import abc
import numpy as np
import Model


class Policy:
    """A policy decides what action to take given an observation of the environment"""

    @abc.abstractmethod
    def choose_action(self, observation):
        """
        Choose an action based on the observation given

        :returns:
            The chosen action
        """
        pass


class GreedyPolicy(Policy):
    """
    A policy that always chooses the action with the highest expected reward based on a model of the state-action
    value function

    :param model:
        The model of the state-action value function
    """

    def __init__(self, model: Model):
        self.model = model

    def choose_action(self, state):
        action_values = self.model.state_values(state)
        # If multiple actions have the same value, chose one at random
        maximums = np.argwhere(action_values == np.amax(action_values)).flatten()
        action = maximums[np.random.randint(0, len(maximums))]
        return action


class EpsilonGreedyPolicy(GreedyPolicy):
    """
    A greedy policy that chooses a random action a given percentage of the times. This makes the policy explore all
    possible state-action pair in the limit.
    """
    def __init__(self, model: Model, exploration: float):
        super().__init__(model)
        self.exploration = exploration

    def choose_action(self, observation):
        if np.random.random_sample() < self.exploration:
            action = np.random.randint(0, self.model.env.action_space.n)
            return action
        else:
            return super().choose_action(observation)
