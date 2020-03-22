import abc


class Model:
    """A model of the state-value function"""

    @abc.abstractmethod
    def action_values(self, observation):
        """Get all action values for state."""
        pass

    @abc.abstractmethod
    def update_action_value(self, observation, new_values):
        pass
