"""Interface for physical systems."""

from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
import gym
from gym import spaces


class AbstractEnv(gym.Env, metaclass=ABCMeta):
    """Interface for physical systems with continuous state-action spaces.

    Parameters
    ----------
    dim_state: Tuple
    dim_action: Tuple
    dim_observation: Tuple, optional

    Methods
    -------
    state: ndarray
        return the current state of the system.
    time: int or float
        return the current time step.
    reset(state):
        reset the state.
    step(action): ndarray
        execute a one step simulation and return the next state.
    """

    def __init__(
        self,
        dim_state: Tuple[int],
        dim_action: Tuple[int],
        dim_observation: Tuple[int] = None,
        min_action: float = -1.0,
        max_action: float = 1.0,
        *args,
        **kwargs
    ):
        self.dim_state = dim_state
        self.dim_action = dim_action
        if dim_observation is None:
            dim_observation = dim_state
        self.dim_observation = dim_observation

        self.min_action = min_action
        self.max_action = max_action

        self.dynamical_system = None

        self._time = 0

    @property  # type: ignore
    @abstractmethod
    def state(self):
        """Return the state of the system."""
        raise NotImplementedError

    @state.setter  # type: ignore
    @abstractmethod
    def state(self, value):
        raise NotImplementedError

    @property
    def time(self):
        """Return the current time of the system."""
        return self._time

    @property
    def action_space(self):
        """Return action space."""
        return spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=self.dim_action,
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        """Return observation space."""
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.dim_observation,
            dtype=np.float32,
        )
