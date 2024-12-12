from functools import partial
from jax import jit, vmap
import jax
from matplotlib import pyplot as plt
import numpy as np
import jax.numpy as jnp
from typing import Optional
from gym.envs.classic_control import PendulumEnv
from gym import spaces
from brax.training.types import Transition
import pandas as pd

from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    DynamicalSystem,
)
from stochastic_optimization.dynamical_system.pendulum_system import (
    PendulumStateSystem,
    PendulumSystem,
)


# ------------------- Environment ------------------- #
class CustomPendulum(PendulumEnv):
    def __init__(self, g=10.0, *args, **kwargs):
        super().__init__(g=g, *args, **kwargs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        high = np.asarray([np.pi + 0.1, 0.1])
        low = np.asarray([np.pi - 0.1, -0.1])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])


class ConstrainedPendulum(PendulumEnv):
    def __init__(
        self,
        action_cost=0.001,
        angle_tolerance=0.1,
        stability_duration=10,
        max_steps=200,
        g=10.0,
        max_speed=8.0,
        max_torque=2.0,
        min_action=-1.0,
        max_action=1.0,
        *args,
        **kwargs
    ):
        super().__init__(g=g, *args, **kwargs)
        self.action_cost = action_cost
        self.max_steps = max_steps
        self.angle_tolerance = angle_tolerance
        self.hold_duration = stability_duration
        self.num_executed_steps = 0
        self.stablized_steps = 0

        # Overrride max_speed for constraint
        self.max_speed = max_speed
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Dimensions
        self.dim_observation = (3,)
        self.dim_action = (1,)
        self.dim_state = (2,)

        # Overrride max_torque for constraint (Actual control action space)
        self.min_torque = -max_torque
        self.max_torque = max_torque

        # Fixes the acceptable range of values for actions
        # This is used to rescale the action space of the base environment
        self.min_action = min_action
        self.max_action = max_action
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=self.dim_action,
            dtype=np.float32,
        )

        self.set_dynamical_sytem()

        self.state_var_names = ["Angle", "Angular Velocity"]

    def set_dynamical_sytem(self, dynamical_system: DynamicalSystem = PendulumSystem()):
        self.dynamical_system = dynamical_system

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.stablized_steps = 0
        self.num_executed_steps = 0

        high = np.asarray([np.pi + 0.1, 0.1])
        low = np.asarray([np.pi - 0.1, -0.1])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def reset_any(
        self, state=None, *, seed: Optional[int] = None, options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        self.stablized_steps = 0
        self.num_executed_steps = 0
        self.last_u = None

        low = np.asarray([-np.pi, -8])
        high = np.asarray([np.pi, 8])

        if state is not None:
            assert len(state) == 2, "State must be a 2D vector"
            assert np.all(
                (state >= low) & (state <= high)
            ), "State must be within bounds"
            self.state = state
        else:
            self.state = self.np_random.uniform(low=low, high=high)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    @staticmethod
    def get_state(obs):
        theta = jnp.arctan2(obs[1], obs[0])
        thetadot = obs[2]
        return jnp.array([theta, thetadot])

    @staticmethod
    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def _step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        action_cost = self.action_cost

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = self.angle_normalize(th) ** 2 + 0.1 * thdot**2 + action_cost * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -costs, False, False, {}

    def step(self, action):
        action = self.rescale_action(action)
        # next_obs, reward, finished, truncated, info = super().step(action)
        next_obs, reward, finished, truncated, info = self._step(
            action
        )  # For custom action cost
        angle = np.arctan2(next_obs[1], next_obs[0])

        # Stability check

        if np.abs(angle) < self.angle_tolerance:
            self.stablized_steps += 1
        else:
            self.stablized_steps = 0

        if self.stablized_steps >= self.hold_duration:
            finished = True

        # Max steps check
        self.num_executed_steps += 1
        if self.num_executed_steps >= self.max_steps:
            truncated = True

        return next_obs, reward, finished, truncated, info

    @partial(jax.jit, static_argnums=0)
    def rescale_action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        action = jnp.clip(action, self.min_action, self.max_action)
        low = self.min_torque
        high = self.max_torque
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = jnp.clip(action, low, high)
        return action

    def plot_episode(self, episode_data: pd.DataFrame):
        x_values = episode_data["Iteration"].values
        x_values = np.array(x_values, dtype=int)
        action = episode_data["Action"].map(lambda x: x[0]).values

        true_obs = np.stack(episode_data["Next_obs"].values)
        pred_obs = np.stack(episode_data["Mean_next_obs"].values)
        std_obs = np.stack(episode_data["Std_next_obs"].values)

        true_angle = vmap(self.get_state)(true_obs)[:, 0]
        pred_angle = vmap(self.get_state)(pred_obs)[:, 0]
        std_angle = np.zeros(
            pred_angle.shape
        )  # Since we cannot derive the std of the angle from its (x,y) components

        true_speed = true_obs[:, -1]
        pred_speed = pred_obs[:, -1]
        std_speed = std_obs[:, -1]

        low_angle = pred_angle - 2 * std_angle
        high_angle = pred_angle + 2 * std_angle

        low_speed = pred_speed - 2 * std_speed
        high_speed = pred_speed + 2 * std_speed

        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        # Plot the true and predicted states of the trajectory
        # Angle
        axs[0].plot(x_values, true_angle, label="True Angle")
        axs[0].plot(x_values, pred_angle, label="Mean Pred Angle")
        axs[0].fill_between(x_values, low_angle, high_angle, color="orange", alpha=0.3)
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Angle")
        axs[0].legend()

        # Speed
        axs[1].plot(x_values, true_speed, label="True Velocity")
        axs[1].plot(x_values, pred_speed, label="Mean Pred Velocity")
        axs[1].fill_between(x_values, low_speed, high_speed, color="orange", alpha=0.3)
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Angluar Velocity")
        axs[1].legend()

        # Plot the planned actions
        axs[2].plot(x_values, action)
        axs[2].set_xlabel("Iteration")
        axs[2].set_ylabel("Action")
        axs[2].set_ylim(bottom=self.action_space.low[0], top=self.action_space.high[0])

        return fig


class ConstrainedPendulumState(ConstrainedPendulum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Observation space
        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.dynamical_system = PendulumStateSystem()

        self.dim_observation = (2,)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = self.angle_normalize(self.state)
        return self.state, {}

    def reset_any(
        self, state=None, *, seed: Optional[int] = None, options: Optional[dict] = None
    ):
        super().reset_any(state=state, seed=seed)
        self.state = self.angle_normalize(self.state)
        return self.state, {}

    def step(self, action):
        next_obs, reward, finished, truncated, info = super().step(action)
        self.state[0] = self.angle_normalize(self.state[0])
        return self.state, reward, finished, truncated, info

    @staticmethod
    def get_state(state):
        return state

    @staticmethod
    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi
