"""
CleanBotExperiments
===================

A set of experiments using the Clean Bot environment to compare the performance of a number of reinforcement learning
methods and ways to represent the state-value-function
"""


from Environments.CleanBotEnv import CleanBotEnv
from Models.TableModel import TableModel
from Models.KerasModel import KerasModel
from Methods.MonteCarlo import ConstAlphaMC
from Methods.TemporalDifference import Sarsa
from Methods.Policies import EpsilonGreedyPolicy, GreedyPolicy
from Experiment import Experiment
from Experiments.KerasModelBuilders import conv1_model


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
        self.method = ConstAlphaMC(self.env, self.model, self.training_policy)

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
        self.model = KerasModel(self.env, model=conv1_model(self.env))
        self.training_policy = EpsilonGreedyPolicy(self.model, 0.1)
        self.testing_policy = GreedyPolicy(self.model)
        self.method = ConstAlphaMC(self.env, self.model, self.training_policy)

        self.training_policy.exploration = 0.1
        self.env.max_steps = 32
        self.method.alpha = 0.01
        self.model.epochs = 100


class SarsaConv1KerasModel(Experiment):
    """
    state-value-function model: Deep learning model with 1 convolutional layer
    training method: Sarsa
    """
    def __init__(self):
        super().__init__()
        self.env = CleanBotEnv(4)
        self.model = KerasModel(self.env, model=conv1_model(self.env))
        self.training_policy = EpsilonGreedyPolicy(self.model, 0.1)
        self.testing_policy = GreedyPolicy(self.model)
        self.method = Sarsa(self.env, self.model, self.training_policy)

        self.training_policy.exploration = 0.1
        self.env.max_steps = 32
        self.method.alpha = 0.01
        self.model.epochs = 100


