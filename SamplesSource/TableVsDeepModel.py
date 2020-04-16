"""
TableVsDeepModel
================

A set of experiments using the Clean Bot environment to compare the performance of a number of reinforcement learning
methods and ways to represent the state-value-function
"""


from CleanBotEnv import CleanBotEnv
from Models.TableModel import TableModel
from Models.KerasModel import KerasModel
from Methods.MonteCarlo import AlphaMC
from Methods.TemporalDifference import Sarsa
from Policies import EpsilonGreedyPolicy, GreedyPolicy
from Experiments.Experiment import Experiment, Suite
from KerasModelBuilders import conv1_model


class AlphaMCArrayModel(Experiment):
    """
    state-value-function model: array
    training method: Alpha Monte Carlo
    """
    def __init__(self):
        super().__init__()
        self.env = CleanBotEnv(4)
        self.model = TableModel(self.env)
        self.training_policy = EpsilonGreedyPolicy(self.model, 0.1)
        self.testing_policy = GreedyPolicy(self.model)
        self.method = AlphaMC(self.env, self.model, self.training_policy)

        self.training_policy.exploration = 0.1
        self.env.max_steps = 32
        self.method.alpha = 0.01


class AlphaMcConv1KerasModel(Experiment):
    """
    state-value-function model: Deep learning model with 1 convolutional layer
    training method: Alpha Monte Carlo
    """
    def __init__(self):
        super().__init__()
        self.env = CleanBotEnv(4)
        self.model = KerasModel(self.env, model=conv1_model(self.env), batch_size=64)
        self.training_policy = EpsilonGreedyPolicy(self.model, 0.1)
        self.testing_policy = GreedyPolicy(self.model)
        self.method = AlphaMC(self.env, self.model, self.training_policy)

        self.training_policy.exploration = 0.1
        self.env.max_steps = 32
        self.method.alpha = 0.01
        self.model.epochs = 60


class SarsaArrayModel(Experiment):
    """
    state-value-function model: Deep learning model with 1 convolutional layer
    training method: Sarsa
    """
    def __init__(self):
        super().__init__()
        self.env = CleanBotEnv(4)
        self.model = TableModel(self.env)
        self.training_policy = EpsilonGreedyPolicy(self.model, 0.1)
        self.testing_policy = GreedyPolicy(self.model)
        self.method = Sarsa(self.env, self.model, self.training_policy)

        self.training_policy.exploration = 0.1
        self.env.max_steps = 32
        self.method.alpha = 0.01


class SarsaConv1KerasModel(Experiment):
    """
    state-value-function model: Deep learning model with 1 convolutional layer
    training method: Sarsa
    """
    def __init__(self):
        super().__init__()
        self.env = CleanBotEnv(4)
        self.model = KerasModel(self.env, model=conv1_model(self.env), batch_size=64)
        self.training_policy = EpsilonGreedyPolicy(self.model, 0.1)
        self.testing_policy = GreedyPolicy(self.model)
        self.method = Sarsa(self.env, self.model, self.training_policy)

        self.training_policy.exploration = 0.1
        self.env.max_steps = 32
        self.method.alpha = 0.01
        self.model.epochs = 60


class TableVsDeepModelSuite(Suite):
    def __init__(self):
        super().__init__()
        self.validation_frequency = 1000
        self.episode_count = 50000
        self.validation_episode_count = 50
        self.experiments = [AlphaMCArrayModel, AlphaMcConv1KerasModel, SarsaArrayModel, SarsaConv1KerasModel]


def experiment_suite() -> Suite:
    return TableVsDeepModelSuite()
