from typing import Optional, Tuple
import gym
from gym import spaces
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from stochastic_optimization.dynamical_system.car_park_system import CarParkSystem
from stochastic_optimization.environment.env_consts import CarParkConsts
import jax.numpy as jnp
import jax


def long_tail_sigmoid(x: jnp.ndarray, w: float) -> jnp.ndarray:
    scale = jnp.sqrt(1 / w - 1)
    return 1 / (1 + (x * scale) ** 2)


class ToleranceReward:
    def __init__(
        self,
        bounds: Tuple[float, float] = (0.0, 0.0),
        margin: float = 0.0,
        value_at_margin: float = 0.1,
        scale: float = 1.0,
    ):
        self.bounds = bounds
        self.margin = margin
        self.value_at_margin = value_at_margin
        lower, upper = bounds
        self.lower = lower
        self.upper = upper
        self.scale = scale
        # if lower > upper:
        #     raise ValueError("Lower bound must be <= upper bound.")
        # if margin < 0:
        #     raise ValueError("margin must be non-negative.")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_bounds = jnp.logical_and(self.lower <= x, x <= self.upper)
        d = jnp.where(x < self.lower, self.lower - x, x - self.upper) / self.margin

        true_fn = lambda x: jnp.where(in_bounds, 1.0, 0.0)
        false_fn = lambda x: jnp.where(
            in_bounds, 1.0, long_tail_sigmoid(d, self.value_at_margin)
        )

        condition = self.margin == 0

        reward = jax.lax.cond(condition, true_fn, false_fn, x)

        return self.scale * reward

        # if self.margin == 0:
        #     return jnp.where(in_bounds, 1.0, 0.0)
        # else:
        #     d = jnp.where(x < self.lower, self.lower - x, x - self.upper) / self.margin
        #     return jnp.where(in_bounds, 1.0, long_tail_sigmoid(d, self.value_at_margin))


class CarParkEnv(gym.Env):

    def __init__(
        self,
        max_action,  # [meters/second^2]
        max_speed=CarParkConsts.MAX_SPEED,  # [meters/second]
        destination=CarParkConsts.DESTINATION,  # [meters]
        road_length=CarParkConsts.ROAD_LENGTH,  # [meters]
        position_bound=(
            CarParkConsts.MIN_POSITION,
            CarParkConsts.MAX_POSITION,
        ),
        margins=(
            CarParkConsts.BOTTOM_MARGIN,
            CarParkConsts.TOP_MARGIN,
        ),  # [meters]
        reward_scale=CarParkConsts.REWARD_SCALE,
        reward_margin=CarParkConsts.REWARD_MARGIN,
        value_at_margin=CarParkConsts.VALUE_AT_MARGIN,
        stability_duration=CarParkConsts.STABILITY_DURATION,
        max_steps=CarParkConsts.MAX_STEPS,
        *args,
        **kwargs,
    ):
        self.min_action = -max_action
        self.max_action = max_action
        self.rescale_action = lambda action: self._rescale_action(
            action, -1, 1, self.min_action, self.max_action
        )

        self.max_speed = max_speed
        self.destination = destination
        self.road_length = road_length
        self.bottom_margin, self.top_margin = margins

        self.reward_scale = reward_scale
        self.reward_margin = reward_margin
        self.value_at_margin = value_at_margin

        self.max_steps = max_steps
        self.stability_duration = stability_duration
        self.num_executed_steps = 0
        self.stablized_steps = 0

        # Constants
        self.dt = CarParkConsts.DT  # Time step [seconds]
        self.g = CarParkConsts.G  # Acceleration of gravity [meters/second^2]
        self.m = CarParkConsts.M  # Mass of the car [kilograms]
        self.mu_static = CarParkConsts.MU_STATIC  # Static friction coefficient
        self.mu_kinetic = CarParkConsts.MU_KINETIC  # Kinetic friction coefficient

        # Dimensions
        self.dim_observation = (2,)
        self.dim_action = (1,)
        self.dim_state = (2,)

        # Spaces
        a_low = -1
        a_high = 1
        self.action_space = spaces.Box(
            low=a_low, high=a_high, shape=self.dim_action, dtype=np.float32
        )  # Define the action space in the range of -1 to 1 and rescale it between min_action and max_action

        position_low, position_high = position_bound
        obs_low = np.array([position_low, -max_speed])
        obs_high = np.array([position_high, max_speed])
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=self.dim_observation, dtype=np.float32
        )

        # self.tolerance_reward = ToleranceReward(
        #     bounds=(self.bottom_margin, self.top_margin),
        #     margin=CarParkConsts.REWARD_MARGIN,
        #     value_at_margin=CarParkConsts.VALUE_AT_MARGIN,
        # )

        self.dynamical_system = CarParkSystem()

        self.state_var_names = ["Position", "Velocity"]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.stablized_steps = 0
        self.num_executed_steps = 0

        low = np.asarray([CarParkConsts.START, 0])
        high = np.asarray([CarParkConsts.START + CarParkConsts.NOISE_SCALE, 0])
        self.state = self.np_random.uniform(low=low, high=high)

        return self.state, {}

    def _step(self, action):

        def tolerance_reward(x):
            lower = self.bottom_margin
            upper = self.top_margin
            margin = self.reward_margin
            scale = self.reward_scale

            in_bound = 1.0 if (lower <= x and x <= upper) else 0.0

            if margin == 0:
                reward = 1.0

            if x < lower and in_bound == 0:
                d = (lower - x) / margin
                reward = long_tail_sigmoid(d, self.value_at_margin)
            elif x > upper and in_bound == 0:
                d = (x - upper) / margin
                reward = long_tail_sigmoid(d, self.value_at_margin)
            else:
                reward = 1.0

            return scale * reward

        # Fetch constants
        dt = self.dt
        g = self.g
        m = self.m
        mu_static = self.mu_static
        mu_kinetic = self.mu_kinetic

        u = self.rescale_action(action)[0]

        position, speed = self.state

        # Object is in motion, use kinetic friction
        if abs(speed) > 0:
            # Direction of friction force is opposite to movement direction
            friction_direction = -np.sign(speed)
            mu = friction_direction * mu_kinetic

            # Compute friction force
            friction = mu * m * g
            net_force = m * u + friction
            new_speed = speed + net_force / m * dt

            # if np.sign(new_speed) != np.sign(speed):
            #     new_speed = 0.0

        # Object is at rest, use static friction
        else:
            # Direction of friction force is opposite to direction to be moved
            friction_direction = -np.sign(u)
            mu = friction_direction * mu_static

            # Compute friction force
            friction = mu * m * g
            net_force = m * u + friction

            # Only move the object if the net force is positive
            if np.sign(net_force) != np.sign(u):
                new_speed = 0.0
            else:
                new_speed = net_force / m * dt

        new_speed = np.clip(new_speed, -self.max_speed, self.max_speed)

        # Update position
        new_position = position + new_speed * dt

        # Update state
        self.state = np.array([new_position, new_speed], dtype=np.float32)

        # Reward function
        # distanceToGoal = (
        #     self.destination - new_position
        # )  # Margin will be before the destination
        distanceToGoal = (
            new_position - self.destination
        )  # Margin will be after the destination

        # state_reward = -(distanceToGoal**2)
        # state_reward = self.tolerance_reward(distanceToGoal)
        state_reward = tolerance_reward(distanceToGoal)
        speed_cost = CarParkConsts.SPEED_COST
        speed_reward = -speed_cost * (new_speed**2)

        action_cost = CarParkConsts.ACTION_COST
        action_reward = -action_cost * (action**2)  # Action reward on unscaled action

        reward = state_reward + speed_reward + action_reward

        max_reward = self.reward_scale
        reward = reward - max_reward  # Cumulative reward gets stagnated

        return self.state, reward

    def step(self, action):
        next_obs, reward = self._step(action)
        finished = False
        truncated = False

        position, speed = next_obs
        # distanceToGoal = self.destination - position
        distanceToGoal = position - self.destination

        if not self.observation_space.contains(next_obs):
            truncated = True

        # Stability check
        if distanceToGoal >= self.bottom_margin and distanceToGoal <= self.top_margin:
            self.stablized_steps += 1
        else:
            self.stablized_steps = 0

        if self.stablized_steps >= self.stability_duration:
            finished = True

        # Max steps check
        self.num_executed_steps += 1
        if self.num_executed_steps >= self.max_steps:
            truncated = True

        return next_obs, reward, finished, truncated, {}

    @staticmethod
    def get_state(obs):
        return obs

    @staticmethod
    def _rescale_action(action, low, high, min_action, max_action):
        """
        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        if low is not None and high is not None:
            action = np.clip(action, low, high)
            action = min_action + (max_action - min_action) * (
                (action - low) / (high - low)
            )
            action = np.clip(action, min_action, max_action)
        return action

    def render(self, mode="human", close=False):
        pass

    def seed(self, seed=None):
        pass

    def plot_episode(self, episode_data: pd.DataFrame):
        x_values = episode_data["Iteration"].values
        x_values = np.array(x_values, dtype=int)
        action = episode_data["Action"].map(lambda x: x[0]).values

        true_position = episode_data["Next_obs"].map(lambda x: x[0]).values
        pred_position = episode_data["Mean_next_obs"].map(lambda x: x[0]).values
        std_position = episode_data["Std_next_obs"].map(lambda x: x[0]).values

        low = pred_position - 2 * std_position
        high = pred_position + 2 * std_position

        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        # Plot the true and predicted states of the trajectory
        axs[0].plot(x_values, true_position, label="True Car Position")
        axs[0].plot(x_values, pred_position, label="Pred Car Position")
        axs[0].fill_between(
            x_values,
            low,
            high,
            color="orange",
            alpha=0.3,
        )
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Observation")
        axs[0].legend()

        # Plot the planned actions
        axs[1].plot(x_values, action)
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Action")
        axs[1].set_ylim(bottom=env.action_space.low[0], top=env.action_space.high[0])

        return fig


if __name__ == "__main__":
    env = CarParkEnv(max_action=8.0)
    obs, _ = env.reset(seed=42)
    print(obs)
    for i in range(100):
        # action = env.action_space.sample()
        if obs[0] >= 5:
            action = np.array([0.0])
        else:
            action = np.array([1.0])
        print(f"Action: {action}")
        next_obs, reward, done, _, _ = env.step(action)
        print(f"Next observation: {next_obs}, Reward: {reward}, Done: {done}")
        obs = next_obs
    env.close()
