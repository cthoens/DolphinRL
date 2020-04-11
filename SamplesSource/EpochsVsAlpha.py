"""
EpochsVsAlpha Experiments
=========================

A set of experiments to compare how the choice a alpha for the Sarsa method and the number of epochs used in
training the Keras model affect the speed of learning.
"""


import numpy as np
from CleanBotEnv import CleanBotEnv
from Models.KerasModel import KerasModel
from Methods.TemporalDifference import Sarsa
from Policies import EpsilonGreedyPolicy, GreedyPolicy
from Experiments.Experiment import Experiment, Suite
from KerasModelBuilders import conv1_model


class EpochsVsAlpha(Experiment):
    def __init__(self, epochs, alpha, batch_size):
        super().__init__()
        self.env = CleanBotEnv(4)
        self.model = KerasModel(self.env, model=conv1_model(self.env), batch_size=batch_size)
        self.training_policy = EpsilonGreedyPolicy(self.model, 0.1)
        self.testing_policy = GreedyPolicy(self.model)
        self.method = Sarsa(self.env, self.model, self.training_policy)

        self.training_policy.exploration = 0.1
        self.env.max_steps = 32
        self.method.alpha = alpha
        self.model.epochs = epochs

    def name(self):
        return f"{type(self).__name__}-{self.model.epochs:03}-{self.method.alpha:.2f}"


class EpochsVsAlphaSuite(Suite):
    def __init__(self):
        super().__init__()
        self.batch_size = 50
        self.episode_count = self.batch_size * 100
        """
        Number of episodes to train for. Has to be a multiple of batch_size, since the model is only trained every
        batch_size episodes 
        """

        self.testing_update_steps = self.episode_count
        """
        Run the validation episodes only once all the training episodes are completed
        """

        self.testing_episode_count = 200

        self.experiments = []
        for epochs in np.linspace(10, 210, num=4, dtype=np.int):
            for alpha in np.linspace(0.01, 0.1, num=4):
                self.experiments.append(
                    lambda e=epochs, a=alpha: EpochsVsAlpha(epochs=e, alpha=a, batch_size=self.batch_size)
                )


def experiment_suite():
    return EpochsVsAlphaSuite()
