import numpy as np
from Models.Model import Model

from Utilities import Env


class TableModel(Model):

    def __init__(self, env):
        self.env = env
        self.shape = Env.obs_action_shape(env)
        self.value_function = np.zeros(self.shape, dtype=np.float32)

    def action_values(self, observation):
        """Get the action values for a given observation"""
        return self.value_function[tuple(np.ravel(observation))]

    def update_action_value(self, obs_action, value):
        """Update a state-action value"""
        self.value_function[tuple(np.ravel(obs_action))] = value









