from enum import Enum
from typing import NamedTuple
import jax.numpy as jnp


class SamplingMode(Enum):
    MEAN = 0
    NOISY_MEAN = 1
    DIST = 2
    TS = 3  # Trajectory sampling


class EvalEpisodeData(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    pred_obs: jnp.ndarray
    step: int
