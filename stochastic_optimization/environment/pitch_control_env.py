from functools import partial
import jax
from matplotlib import pyplot as plt
import numpy as np
import jax.numpy as jnp
from typing import Optional, Tuple
import pandas as pd
from scipy import integrate, stats
from gym.utils import seeding

from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    DynamicalSystem,
)
from stochastic_optimization.dynamical_system.pitch_control_system import (
    PitchControlSystem,
)
from stochastic_optimization.environment.abstract_env import AbstractEnv


# ------------------- Environment ------------------- #
class PitchControlEnv(AbstractEnv):

    def __init__(
        self,
        init_angle=-0.2,
        desired_angle=0.0,
        angle_tolerance=0.025,
        stability_duration=20,
        max_steps=200,
        min_action=-1.0,
        max_action=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            dim_state=(3,),
            dim_action=(1,),
            dim_observation=(3,),
            min_action=min_action,
            max_action=max_action,
            *args,
            **kwargs,
        )
        # Physical parameters
        self.omega = 56.7
        self.cld = 0.313
        self.cmld = 0.0139
        self.cw = 0.232
        self.cm = 0.426
        self.eta = 0.0875
        self.step_size = 0.05

        # Elevator angle limits (Actual control action space)
        self.min_elevator_angle = -1.4
        self.max_elevator_angle = 1.4

        # Initial angle
        self.init_angle = init_angle
        # Target angle
        self.desired_angle = desired_angle

        # Constrained environment parameters
        self.max_steps = max_steps
        self.angle_tolerance = angle_tolerance
        self.stability_duration = stability_duration
        self.num_executed_steps = 0
        self.stablized_steps = 0

        self.integrator = integrate.RK45

        self.set_dynamical_sytem()

        self.state_var_names = ["Attack Angle", "Pitch Rate", "Pitch Angle"]

    def set_dynamical_sytem(
        self, dynamical_system: DynamicalSystem = PitchControlSystem()
    ):
        self.dynamical_system = dynamical_system

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.stablized_steps = 0
        self.num_executed_steps = 0
        self._time = 0

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # Initial state
        low = jnp.array([-0.001, -0.001, self.init_angle - 0.001])
        high = jnp.array([0.001, 0.001, self.init_angle + 0.001])
        self.state = self._np_random.uniform(low=low, high=high)

        return self.state, {}

    def reset_any(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.stablized_steps = 0
        self.num_executed_steps = 0
        self._time = 0

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # Initial state
        low = jnp.array([0, 0, -0.5])
        high = jnp.array([0, 0, 0.5])
        self.state = self._np_random.uniform(low=low, high=high)

        return self.state, {}

    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, float, bool, dict]:
        obs = self.state
        alpha, q, theta = obs

        # Clip action
        action = self.rescale_action(action)[0]

        # Integrate
        integrator = self.integrator(
            fun=lambda t, x: self.ode(t, x, action),
            t0=0,
            y0=obs,
            t_bound=self.step_size,
        )

        while integrator.status == "running":
            integrator.step()
        next_obs = integrator.y

        # Reward function
        reward = -2.0 * (theta - self.desired_angle) ** 2 - 0.02 * action**2

        # Set next state
        self.state = next_obs

        # Check if done
        if np.abs(theta - self.desired_angle) < self.angle_tolerance:
            self.stablized_steps += 1
        else:
            self.stablized_steps = 0

        if self.stablized_steps >= self.stability_duration:
            done = True
        else:
            done = False

        # Check if truncated
        self.num_executed_steps += 1
        if self.num_executed_steps >= self.max_steps:
            truncate = True
        else:
            truncate = False

        self._time += self.step_size

        return next_obs, reward, done, truncate, {}

    def ode(self, t, obs, action):
        # Physical dynamics
        alpha, q, theta = obs
        u = action

        alpha_dot = -self.cld * alpha + self.omega * q + self.cw * u
        q_dot = -self.cmld * alpha - self.cm * q + self.eta * self.cw * u
        theta_dot = self.omega * q

        derivatives = jnp.array([alpha_dot, q_dot, theta_dot])
        return derivatives

    @partial(jax.jit, static_argnums=0)
    def rescale_action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        action = jnp.clip(action, self.min_action, self.max_action)
        low = self.min_elevator_angle
        high = self.max_elevator_angle
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = jnp.clip(action, low, high)
        return action

    def plot_episode(self, episode_data: pd.DataFrame):
        x_values = episode_data["Iteration"].values
        x_values = np.array(x_values, dtype=int)
        action = episode_data["Action"].map(lambda x: x[0]).values

        true_angle = episode_data["Next_obs"].map(lambda x: x[-1]).values
        pred_angle = episode_data["Mean_next_obs"].map(lambda x: x[-1]).values
        std_angle = episode_data["Std_next_obs"].map(lambda x: x[-1]).values

        low = pred_angle - 2 * std_angle
        high = pred_angle + 2 * std_angle

        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        # Plot the true and predicted states of the trajectory
        axs[0].plot(x_values, true_angle, label="True Pitch Angle")
        axs[0].plot(x_values, pred_angle, label="Pred Pitch Angle")
        axs[0].fill_between(x_values, low, high, color="orange", alpha=0.3)
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Observation")
        axs[0].legend()

        # Plot the planned actions
        axs[1].plot(x_values, action)
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Action")
        axs[1].set_ylim(bottom=self.action_space.low[0], top=self.action_space.high[0])

        return fig

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


if __name__ == "__main__":
    import gym

    env = PitchControlEnv()
    obs, _ = env.reset()
    low = env.action_space.low[0]
    print(low)
    print(isinstance(env, gym.Env))
    action = env.action_space.sample()
    next_obs, reward, done, truncate, _ = env.step(action)
    print(
        f"Observation: {obs}, Action: {action}, Next Observation: {next_obs}, Reward: {reward}"
    )
