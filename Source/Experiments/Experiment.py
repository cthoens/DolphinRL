from Utilities.Eval import validate_policy


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

    def validate(self, episode_count=200):
        return validate_policy(self.env, self.testing_policy, episode_count=episode_count)

    def name(self):
        return type(self).__name__


class Suite:
    def __init__(self):
        self.validation_frequency = None
        self.episode_count = None
        self.experiments: [Experiment] = []
        self.testing_episode_count = None

    def validate(self, experiment: Experiment):
        return experiment.validate(episode_count=self.testing_episode_count)
