import keras
from keras import backend as K

from gym import Env, spaces
import numpy as np

from Model import Model


class KerasModel(Model):
    """
    An implementation of Model that approximates the action-value function using a neural network implemented with
    Keras. The Keras model used has to be provided on instantiation. The implementation does not update the keras model
    for every call to update_action_value(), but collects batch_size updated before the model is fitted to the new
    observations

    :param env: The environment
    :param model: A compiled keras model
    :param batch-size: The number of updates that are collected before the model is fitted to the new experience
    """

    def __init__(self, env: Env, model: keras.Model, batch_size=128):
        assert isinstance(env.observation_space, spaces.Box), "Unsupported observation space"
        assert np.count_nonzero(env.observation_space.low) == 0, "Unsupported observation space"
        assert isinstance(env.action_space, spaces.Discrete), "Unsupported action space"

        self.env = env
        """The environment"""
        self.model = model
        """The keras model"""
        self.batch_size = batch_size
        """The number of updates that are collected before the model is fitted to the new experience"""
        self.epochs = 60
        """The number of epochs to run when the model is fitted to the new experience"""

        self._collected_count = 0
        self._rows_count, self._col_count = env.observation_space.shape
        self._x_train_shape = (self.batch_size,) + env.observation_space.shape
        self._x_train = np.zeros(self._x_train_shape, 'float32')
        """observations used as input during model fitting"""
        self._y_train = np.zeros((self.batch_size,) + (env.action_space.n,))
        """action values used as expectation during model fitting"""
        self._input_normalizer = 1.0 / env.observation_space.high

    def update_action_value(self, state, action, observed_reward):
        # Evaluate the model to retrieve the current action value estimation and update the actual
        # value of the action taken
        updated_state_values = self.state_values(state)
        updated_state_values[action] = observed_reward

        # Store observation and action value for model fitting
        self._x_train[self._collected_count] = state
        self._y_train[self._collected_count] = updated_state_values
        self._collected_count += 1
        if self._collected_count == self.batch_size:
            # If batch_size updates have been collected fit the model and clear the buffer
            self.train()
            self._collected_count = 0

    def train(self):
        # Normalize the input values.
        self._x_train *= self._input_normalizer

        # Reshape to match the image format of the backend used
        if K.image_data_format() == 'channels_first':
            x_train = self._x_train.reshape((self._x_train.shape[0], 1, self._rows_count, self._col_count,))
        else:
            x_train = self._x_train.reshape((self._x_train.shape[0], self._rows_count, self._col_count, 1,))

        self.model.fit(x_train, self._y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=0)

    def state_values(self, state):
        # Normalize the input values.
        normalized_state = state * self._input_normalizer
        # Add batch size of 1 to front, and 1 channel to back of the state shape
        shape = (1, ) + state.shape + (1, )
        # Predict and extract prediction for batch 0
        return self.model.predict(normalized_state.reshape(shape))[0]

    def action_value(self, state, action):
        return self.state_values(state)[action]

    def save(self, name):
        if not name[:-3] == ".h5":
            name = name+".h5"
        self.model.save(name)
