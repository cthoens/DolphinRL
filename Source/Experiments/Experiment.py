from Utilities.Eval import validate_policy
from gym import Env
from Model import Model
from Policies import Policy

from typing import List, Dict, Callable


class Experiment:
    """
    Base class for experiment.

    TODO: Explain what this is good for
    """
    def __init__(self,  env: Env = None, model: Model = None, training_policy: Policy = None,
                 testing_policy: Policy = None, method=None):
        self.env = env
        self.model = model
        self.training_policy = training_policy
        self.testing_policy = testing_policy
        self.method = method
        self.name = type(self).__name__

    def validate(self, episode_count=200):
        return validate_policy(self.env, self.testing_policy, episode_count=episode_count)


class Suite:
    def __init__(self, episode_count, validation_frequency, validation_episode_count):
        self.experiments: [Experiment] = []
        self.episode_count = episode_count
        """"Number of episodes to train for. """

        self.validation_frequency = validation_frequency
        """Number of training episodes after which to run the validation episodes again"""

        self.validation_episode_count = validation_episode_count
        """Number of episodes in the validation set"""

    def validate(self, experiment: Experiment):
        return experiment.validate(episode_count=self.validation_episode_count)


class DefaultSuite(Suite):
    def __init__(self,
                 factory_function: Callable,
                 experiment_args: List[Dict[str, None]],
                 episode_count: int,
                 validation_frequency: int,
                 validation_episode_count: int):
        """
        Suite that uses a factory function that creates Experiment instances and a list of dictionaries that provides
        arguments to pass to the factory function.

        :param factory_function: The factory function that creates Experiment instances
        :param experiment_args: List of dictionaries that provides arguments to pass to the factory function.
        :param episode_count: The number of episodes to trains for
        :param validation_frequency: Number of training episodes after which to run the validation episodes again
        :param validation_episode_count: Number of episodes in the validation set
        """

        super().__init__(episode_count, validation_frequency, validation_episode_count)
        self.experiment_args = experiment_args
        self.experiments = [
            lambda args_dict=experiment: factory_function(**args_dict) for experiment in experiment_args
        ]

