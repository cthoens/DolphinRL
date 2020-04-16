"""
EpochsVsAlpha Experiments
=========================

A set of experiments to compare how the choice a alpha for the Sarsa method and the number of epochs used in
training the Keras model affect the speed of learning.
"""


from CleanBotEnv import CleanBotEnv
from Models.KerasModel import KerasModel
from Methods.MonteCarlo import AlphaMC
from Policies import EpsilonGreedyPolicy, GreedyPolicy
from Experiments.Experiment import Experiment, Suite, DefaultSuite
from KerasModelBuilders import conv1_model


class EpochsVsAlpha(Experiment):
    def __init__(self, epochs, alpha, batch_size):
        super().__init__()
        self.env = CleanBotEnv(4)
        self.model = KerasModel(self.env, model=conv1_model(self.env), batch_size=batch_size)
        self.training_policy = EpsilonGreedyPolicy(self.model, 0.1)
        self.testing_policy = GreedyPolicy(self.model)
        self.method = AlphaMC(self.env, self.model, self.training_policy)
        self.name = f"{type(self).__name__}-{self.batch_size:03}-{self.model.epochs:03}-{self.method.alpha:.2f}"

        self.training_policy.exploration = 0.1
        self.env.max_steps = 32
        self.method.alpha = alpha
        self.model.epochs = epochs
        self.batch_size = batch_size


def experiment_suite() -> Suite:
    batch_size_range = [32, 64, 80, 96]
    epoch_range = [20, 40, 60, 80]
    alpha_range = [0.01, 0.03, 0.05, 0.07, 0.09]
    experiments = [
        {'batch_size': b, 'epochs': e, 'alpha': a}
        for b in batch_size_range
        for e in epoch_range
        for a in alpha_range
    ]

    return DefaultSuite(
        EpochsVsAlpha,
        experiments,
        episode_count=5000,
        validation_frequency=250,
        validation_episode_count=50
    )
