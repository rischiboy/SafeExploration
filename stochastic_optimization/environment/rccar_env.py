import time
from gym import spaces
from functools import partial
from typing import Dict, Any, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    DynamicalSystem,
)
from stochastic_optimization.dynamical_system.rccar_system import (
    DEFAULT_DT,
    DEFAULT_GOAL,
    DEFAULT_INIT_POSE,
    DEFAULT_X_BOUNDS,
    DEFAULT_Y_BOUNDS,
    RCCarDynamics,
    CarParams,
    RCCarEnvReward,
    RCCarSystem,
)
from stochastic_optimization.environment.abstract_env import AbstractEnv
from stochastic_optimization.environment.env_utils import (
    encode_angles,
    decode_angles,
    plot_rc_trajectory,
)

OBS_NOISE_STD_SIM_CAR: jnp.array = 0.1 * jnp.exp(
    jnp.array([-4.5, -4.5, -4.0, -2.5, -2.5, -1.0])
)


class RCCarSimEnv(AbstractEnv):
    # max_steps: int = 200
    # _dt: float = 1 / 30.0
    dim_action: Tuple[int] = (2,)
    # _goal: jnp.array = jnp.array([0.0, 0.0, 0.0])
    # _init_pose: jnp.array = jnp.array([1.42, -1.02, jnp.pi])
    _angle_idx: int = 2
    _obs_noise_stds: jnp.array = OBS_NOISE_STD_SIM_CAR

    def __init__(
        self,
        init_pose: jnp.array = DEFAULT_INIT_POSE,
        goal: jnp.array = DEFAULT_GOAL,
        dt: float = DEFAULT_DT,
        ctrl_cost_weight: float = 0.005,
        encode_angle: bool = False,
        use_obs_noise: bool = True,
        use_tire_model: bool = False,
        action_delay: float = 0.0,
        car_model_params: dict = None,
        margin_factor: float = 10.0,
        max_throttle: float = 1.0,
        car_id: int = 2,
        ctrl_diff_weight: float = 0.0,
        seed: int = 230492394,
        max_steps: int = 200,
    ):
        """
        Race car simulator environment

        Args:
            ctrl_cost_weight: weight of the control penalty
            encode_angle: whether to encode the angle as cos(theta), sin(theta)
            use_obs_noise: whether to use observation noise
            use_tire_model: whether to use the (high-fidelity) tire model, if False just uses a kinematic bicycle model
            action_delay: whether to delay the action by a certain amount of time (in seconds)
            car_model_params: dictionary of car model parameters that overwrite the default values
            seed: random number generator seed
        """

        dim_state: Tuple[int] = (7,) if encode_angle else (6,)
        # self.dim_action = (2,)

        super().__init__(
            dim_state=dim_state,
            dim_action=self.dim_action,
            dim_observation=dim_state,
            min_action=-1,
            max_action=1,
        )

        self._init_pose = init_pose
        self._goal = goal
        self._dt = dt
        self.encode_angle: bool = encode_angle
        self._rds_key = jax.random.PRNGKey(seed)
        self.max_throttle = jnp.clip(max_throttle, 0.0, 1.0)
        self.max_steps = max_steps

        # set car id and corresponding parameters
        assert car_id in [1, 2]
        self.car_id = car_id
        self._set_car_params()

        # initialize dynamics and observation noise models
        self._dynamics_model = RCCarDynamics(dt=self._dt, encode_angle=False)
        self.set_dynamical_system()
        self.set_box()

        self.use_tire_model = use_tire_model
        if use_tire_model:
            self._default_car_model_params = self._default_car_model_params_blend
        else:
            self._default_car_model_params = self._default_car_model_params_bicycle

        if car_model_params is None:
            _car_model_params = self._default_car_model_params
        else:
            _car_model_params = self._default_car_model_params
            _car_model_params.update(car_model_params)
        # self._dynamics_params = CarParams(**_car_model_params)
        self._dynamics_params = CarParams()
        self._next_step_fn = jax.jit(
            partial(self._dynamics_model.next_state, params=self._dynamics_params)
        )

        self.use_obs_noise = use_obs_noise

        # initialize reward model
        self._reward_model = RCCarEnvReward(
            goal=self._goal,
            ctrl_cost_weight=ctrl_cost_weight,
            encode_angle=self.encode_angle,
            margin_factor=margin_factor,
        )

        # set up action delay
        assert action_delay >= 0.0, "Action delay must be non-negative"
        self.action_delay = action_delay
        if abs(action_delay % self._dt) < 1e-8:
            self._act_delay_interpolation_weights = jnp.array([1.0, 0.0])
        else:
            # if action delay is not a multiple of dt, compute weights to interpolate
            # between temporally closest actions
            weight_first = (action_delay % self._dt) / self._dt
            self._act_delay_interpolation_weights = jnp.array(
                [weight_first, 1.0 - weight_first]
            )
        action_delay_buffer_size = int(jnp.ceil(action_delay / self._dt)) + 1
        self._action_buffer = jnp.zeros((action_delay_buffer_size, self.dim_action[0]))

        # initialize time and state
        self._time: int = 0
        self._state: jnp.array = jnp.zeros(self.dim_state)
        self.ctrl_diff_weight = ctrl_diff_weight

    def set_dynamical_system(self, dynamical_system: DynamicalSystem = None):
        if dynamical_system is None:
            self.dynamical_system = RCCarSystem(dynamics=self._dynamics_model)
        else:
            self.dynamical_system = dynamical_system

        return

    def set_box(self):
        if hasattr(self.dynamical_system, "cost"):
            cost_params = self.dynamical_system.cost.default_params
            _, _, x_bounds, y_bounds = self.dynamical_system.cost.get_cost_params(
                cost_params
            )
        else:
            x_bounds = DEFAULT_X_BOUNDS
            y_bounds = DEFAULT_Y_BOUNDS

        width = x_bounds[1] - x_bounds[0]
        height = y_bounds[1] - y_bounds[0]

        # self.box = np.array([x_bounds[0], y_bounds[0], width, height])

        self.box = np.array([y_bounds[0], -x_bounds[1], height, width])

        return

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        if seed is not None:
            self.seed = seed
            self._rds_key = jax.random.PRNGKey(seed)

        return self.reset_env()

    def reset_env(self, rng_key: Optional[jax.random.PRNGKey] = None) -> jnp.array:
        """Resets the environment to a random initial state close to the initial pose"""
        rng_key = self.rds_key if rng_key is None else rng_key

        # sample random initial state
        key_pos, key_theta, key_vel, key_obs = jax.random.split(rng_key, 4)
        # init_pos = self._init_pose[:2] + jax.random.uniform(
        #     key_pos, shape=(2,), minval=-0.10, maxval=0.10
        # )
        # init_theta = self._init_pose[2:] + jax.random.uniform(
        #     key_pos, shape=(1,), minval=-0.10 * jnp.pi, maxval=0.10 * jnp.pi
        # )
        init_pos = self._init_pose[:2] + jax.random.uniform(
            key_pos, shape=(2,), minval=-0.05, maxval=0.05
        )
        init_theta = self._init_pose[2:] + jax.random.uniform(
            key_pos, shape=(1,), minval=-0.01 * jnp.pi, maxval=0.01 * jnp.pi
        )
        init_vel = jnp.zeros((3,)) + jnp.array(
            [0.005, 0.005, 0.02]
        ) * jax.random.normal(key_vel, shape=(3,))
        init_state = jnp.concatenate([init_pos, init_theta, init_vel])

        self._state = init_state
        self._time = 0
        return self._state_to_obs(self._state, rng_key=key_obs), {}

    def step(
        self, action: jnp.array, rng_key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.array, float, bool, Dict[str, Any]]:
        """Performs one step in the environment

        Args:
            action: array of size (2,) with [steering, throttle]
            rng_key: rng key for the observation noise (optional)
        """

        assert action.shape[-1:] == self.dim_action
        action = jnp.clip(action, -1.0, 1.0)
        action = action.at[0].set(self.max_throttle * action[0])
        # assert jnp.all(-1 <= action) and jnp.all(action <= 1), "action must be in [-1, 1]"
        rng_key = self.rds_key if rng_key is None else rng_key

        jitter_reward = jnp.zeros_like(action).sum(-1)
        if self.action_delay > 0.0:
            # pushes action to action buffer and pops the oldest action
            # computes delayed action as a linear interpolation between the relevant actions in the past
            action, jitter_reward = self._get_delayed_action(action)

        # compute next state
        self._state = self._next_step_fn(self._state, action)
        self._time += 1
        obs = self._state_to_obs(self._state, rng_key=rng_key)

        # compute reward
        reward = (
            self._reward_model.forward(obs=None, action=action, next_obs=obs)
            + jitter_reward
        )

        # check if done
        done = self._time >= self.max_steps

        # return observation, reward, done, info
        return (
            obs,
            reward,
            done,
            done,
            {"time": self._time, "state": self._state, "reward": reward},
        )

    def _state_to_obs(
        self, state: jnp.array, rng_key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.array:
        """Adds observation noise to the state"""
        assert state.shape[-1] == 6
        rng_key = self.rds_key if rng_key is None else rng_key

        # add observation noise
        if self.use_obs_noise:
            obs = state + self._obs_noise_stds * jax.random.normal(
                rng_key, shape=self._state.shape
            )
        else:
            obs = state

        # encode angle to sin(theta) and cos(theta) if desired
        if self.encode_angle:
            obs = encode_angles(obs, self._angle_idx)
        assert (obs.shape[-1] == 7 and self.encode_angle) or (
            obs.shape[-1] == 6 and not self.encode_angle
        )
        return obs

    def _get_delayed_action(self, action: jnp.array) -> Tuple[jnp.array, jnp.array]:
        # push action to action buffer
        last_action = self._action_buffer[-1]
        reward = -self.ctrl_diff_weight * jnp.sum((action - last_action) ** 2)
        self._action_buffer = jnp.concatenate(
            [self._action_buffer[1:], action[None, :]], axis=0
        )

        # get delayed action (interpolate between two actions if the delay is not a multiple of dt)
        delayed_action = jnp.sum(
            self._action_buffer[:2] * self._act_delay_interpolation_weights[:, None],
            axis=0,
        )
        assert delayed_action.shape == self.dim_action
        return delayed_action, reward

    @property
    def rds_key(self) -> jax.random.PRNGKey:
        self._rds_key, key = jax.random.split(self._rds_key)
        return key

    @property
    def time(self) -> float:
        return self._time

    def _set_car_params(self):
        from stochastic_optimization.environment.rccar_sim_config import (
            DEFAULT_PARAMS_BICYCLE_CAR1,
            DEFAULT_PARAMS_BLEND_CAR1,
            DEFAULT_PARAMS_BICYCLE_CAR2,
            DEFAULT_PARAMS_BLEND_CAR2,
        )

        if self.car_id == 1:
            self._default_car_model_params_bicycle: Dict = DEFAULT_PARAMS_BICYCLE_CAR1
            self._default_car_model_params_blend: Dict = DEFAULT_PARAMS_BLEND_CAR1
        elif self.car_id == 2:
            self._default_car_model_params_bicycle: Dict = DEFAULT_PARAMS_BICYCLE_CAR2
            self._default_car_model_params_blend: Dict = DEFAULT_PARAMS_BLEND_CAR2
        else:
            raise NotImplementedError(f"Car idx {self.car_id} not supported")

    @property  # type: ignore
    def state(self):
        """Return the state of the system."""
        return self._state

    @state.setter  # type: ignore
    def state(self, value):
        self._state = value

    @staticmethod
    def get_state(obs):
        return obs

    def plot_episode(self, episode_data: pd.DataFrame):

        obs = episode_data["Obs"].values
        last_obs = episode_data["Next_obs"].values[-1]

        obs = np.stack(obs)
        last_obs = np.stack([last_obs])

        traj = np.concatenate([obs, last_obs], axis=0)
        actions = np.stack(episode_data["Action"].values)

        fig, axes = plot_rc_trajectory(
            traj=traj,
            actions=actions,
            pos_domain_size=3,
            box=self.box,
            show=True,
            encode_angle=self.encode_angle,
        )

        return fig
