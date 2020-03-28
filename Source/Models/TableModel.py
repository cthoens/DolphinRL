import numpy as np
from Models.Model import Model
from Utilities.Env import to_table_index

from Utilities import Env


class TableModel(Model):

    def __init__(self, env):
        self.env = env
        self.shape = Env.obs_action_shape(env)
        self.value_function = np.zeros(self.shape, dtype=np.float32)

    def state_values(self, state):
        return self.value_function[to_table_index(state)]

    def action_value(self, state, action):
        """Get all action values for state."""
        return self.value_function[to_table_index(state, action)]

    def update_action_value(self, state, action, value):
        """Update a state-action value"""
        self.value_function[to_table_index(state, action)] = value









