from Utilities.Eval import test_policy


class Experiment:
    """
    Base class for experiment.

    TODO: Explain what this is good for
    """

    def __init__(self):
        self.env = None
        self.model = None
        self.training_policy = None
        self.testing_policy = None
        self.method = None
        self.testing_episode_count = 200

    def validate(self):
        return test_policy(self.env, self.testing_policy, episode_count=self.testing_episode_count)


