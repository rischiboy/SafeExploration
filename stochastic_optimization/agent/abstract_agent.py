from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    """Abstract class for an agent that interacts with an environment."""

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, obs, action, model_params, rng, *args, **kwargs):
        """Predict the next state, reward (and cost) given the current state and action."""
        pass

    @abstractmethod
    def validate(self, obs, action, model_params, rng, *args, **kwargs):
        """Validate the model predictions."""
        pass

    @abstractmethod
    def select_best_action(self, obs, model_params, rng, *args, **kwargs):
        """Select the best action based on the current model."""
        pass

    @abstractmethod
    def train_step(self, buffer, buffer_state, model_params, diff_states: bool):
        """Train the model and obtain the next model state."""
        pass

    @abstractmethod
    def reset_optimizer(self):
        """Reset the optimizer."""
        pass
