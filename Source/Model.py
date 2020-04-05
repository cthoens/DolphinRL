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
        """Update the value of a single state-action pair"""
        pass

    @abc.abstractmethod
    def save(self, file):
        """"
        Save the model in it current state to file such that training can be resumed from the saved state.

        :param file:
            - string, path to the file to save the model to
            - any file-like object implementing the method `write` that accepts
                `bytes` data (e.g. `io.BytesIO`).
        """
        pass
