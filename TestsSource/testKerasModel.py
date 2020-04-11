import unittest
import numpy as np
from CleanBotEnv import CleanBotEnv
from Models.KerasModel import KerasModel
from Methods.MonteCarlo import AlphaMC
from Policies import EpsilonGreedyPolicy
from utilities import MockEnv
from KerasModelBuilders import conv1_model

SOUTH = CleanBotEnv.BotActions.SOUTH.value
EAST = CleanBotEnv.BotActions.EAST.value
WEST = CleanBotEnv.BotActions.WEST.value
CLEAN = CleanBotEnv.BotActions.CLEAN.value


class TestKerasModel(unittest.TestCase):
    def test_model(self):
        env = MockEnv(3)
        env.max_steps = 18
        keras_model = conv1_model(env)
        model = KerasModel(env, model=keras_model, batch_size=1)
        model.epochs = 30

        obs1 = np.array([[2, 0, 0],
                         [0, 0, 0],
                         [0, 0, 1]])
        expected1 = 13

        # Get the current value for obs1 predicted by the model and update the value for action EAST
        updated_state_values = model.state_values(obs1)
        updated_state_values[EAST] = expected1

        # Update the model, calculate and check mean square error
        model.update_action_value(obs1, EAST, expected1)
        predicted = model.state_values(obs1)
        mean_square_error1 = ((predicted - updated_state_values) ** 2).mean()

        # Second round with the same training data
        model.update_action_value(obs1, EAST, expected1)
        predicted = model.state_values(obs1)
        mean_square_error2 = ((predicted - updated_state_values) ** 2).mean()

        # Check that the error decreased
        self.assertGreater(mean_square_error1, mean_square_error2)

    def test_smoke(self):
        np.random.seed(643674)
        env = CleanBotEnv(3)
        keras_model = conv1_model(env)
        model = KerasModel(env, model=keras_model, batch_size=7)
        policy = EpsilonGreedyPolicy(model, 0.1)
        mc = AlphaMC(env, model, policy)

        policy.exploration = 0.1
        episode_count = 1
        # for i in range(episode_count):
        mc.run_episode()


if __name__ == "__main__":
    unittest.main()
