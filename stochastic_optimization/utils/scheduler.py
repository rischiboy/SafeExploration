from abc import ABC, abstractmethod
import jax.numpy as jnp


class Scheduler(ABC):

    def __init__(self, value: float, n_steps: int):
        self.value = value
        self.n_steps = n_steps

    @abstractmethod
    def update(self, step: int, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass


class LinearScheduler(Scheduler):
    def __init__(self, start_value: float, end_value: float, n_steps: int):
        super().__init__(start_value, n_steps)
        self.start_value = start_value
        self.end_value = end_value

    def update(self, step: int):
        self.value = self.start_value + (self.end_value - self.start_value) * min(
            step / self.n_steps, 1.0
        )
        return self.value

    def reset(self):
        self.value = self.start_value


class SigmoidScheduler(Scheduler):
    def __init__(
        self,
        start_value: float,
        end_value: float,
        scale: float,
        midpoint: float,
        n_steps: int,
    ):
        super().__init__(start_value, n_steps)
        self.start_value = start_value
        self.end_value = end_value
        self.scale = scale
        self.midpoint = midpoint

    def update(self, step: int):
        vertical_scale = self.end_value - self.start_value
        self.value = (
            vertical_scale * (1 / (1 + jnp.exp(-self.scale * (step - self.midpoint))))
            + self.start_value
        )
        return self.value

    def reset(self):
        self.value = self.start_value
