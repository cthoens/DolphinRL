import abc


class Model:
    """A model of the state-value function"""

    @abc.abstractmethod
    def state_values(self, state):
        """Get all action values for state."""
        pass

    @abc.abstractmethod
    def action_value(self, state, action):
        """Get all action values for state."""
        pass

    @abc.abstractmethod
    def update_action_value(self, state, action, new_values):
        pass
