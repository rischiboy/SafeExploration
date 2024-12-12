from functools import partial
import gym
import jax
from jax import jit
from jax import vmap
import numpy as np
import jax.numpy as jnp
from typing import Any, NamedTuple, Optional, Tuple

from bsm.statistical_model.abstract_statistical_model import StatisticalModel
from bsm.utils.type_aliases import StatisticalModelOutput, StatisticalModelState

from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    CostModel,
    RewardModel,
    DynamicalSystem,
    SafeDynamicalSystem,
    SimpleDynamicsModel,
)
from stochastic_optimization.environment.env_consts import CarParkConsts
from stochastic_optimization.environment.car_park_env import ToleranceReward


# ------------------- Dynamics ------------------- #
class CarParkDynamicsParams(NamedTuple):
    dt = jnp.array(CarParkConsts.DT)
    g = jnp.array(CarParkConsts.G)
    m = jnp.array(CarParkConsts.M)
    mu_static = jnp.array(CarParkConsts.MU_STATIC)
    mu_kinetic = jnp.array(CarParkConsts.MU_KINETIC)
    max_action = jnp.array(CarParkConsts.MAX_ACTION)
    max_speed = jnp.array(CarParkConsts.MAX_SPEED)


class CarParkDynamics(SimpleDynamicsModel):

    def __init__(
        self,
        min_action=-1.0,
        max_action=1.0,
        dynamics_params=CarParkDynamicsParams(),
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.input_dim = 3
        self.output_dim = 2

        self.min_action = min_action
        self.max_action = max_action

        self.default_params = dynamics_params

    def __call__(self, input, dynamics_params, rng=None):
        assert input.shape == (self.input_dim,)
        outs = self.predict(input, dynamics_params, rng)

        return outs

    @staticmethod
    def get_dynamics_params(dynamics_params: CarParkDynamicsParams):
        dt = dynamics_params.dt
        g = dynamics_params.g
        m = dynamics_params.m
        mu_static = dynamics_params.mu_static
        mu_kinetic = dynamics_params.mu_kinetic
        max_action = dynamics_params.max_action
        max_speed = dynamics_params.max_speed

        return (dt, g, m, mu_static, mu_kinetic, max_action, max_speed)

    def next_state(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        dynamics_params: CarParkDynamicsParams = None,
    ) -> Tuple[jnp.ndarray, CarParkDynamicsParams]:
        if not dynamics_params:
            dynamics_params = self.default_params

        dt, g, m, mu_static, mu_kinetic, max_action, max_speed = (
            self.get_dynamics_params(dynamics_params)
        )

        position, speed = obs
        u = self.rescale_action(action)[0]

        def object_in_motion(speed):
            # Direction of friction force is opposite to movement direction
            friction_direction = -jnp.sign(speed)
            mu = friction_direction * mu_kinetic

            # Compute friction force
            friction = mu * m * g
            net_force = m * u + friction
            new_speed = speed + net_force / m * dt

            # speed_update = jnp.sign(new_speed) == jnp.sign(speed)

            # new_speed = jax.lax.cond(
            #     speed_update, lambda x: x, lambda x: 0.0, new_speed
            # )
            return new_speed

        def object_at_rest(speed):
            # Direction of friction force is opposite to direction to be moved
            friction_direction = -jnp.sign(u)
            mu = friction_direction * mu_static

            # Compute friction force
            friction = mu * m * g
            net_force = m * u + friction

            # Check if the object can be moved
            stationary_net_force = lambda net_force: 0.0
            movable_net_force = lambda net_force: net_force / m * dt

            condition = jnp.sign(net_force) != jnp.sign(u)

            new_speed = jax.lax.cond(
                condition, stationary_net_force, movable_net_force, net_force
            )

            return new_speed

        condition = abs(speed) > 0
        new_speed = jax.lax.cond(condition, object_in_motion, object_at_rest, speed)

        new_speed = jnp.clip(new_speed, -max_speed, max_speed)

        # Update position
        new_position = position + new_speed * dt

        # Update state
        next_obs = jnp.array([new_position, new_speed])

        return next_obs

    @partial(jax.jit, static_argnums=0)
    def rescale_action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        action = jnp.clip(action, self.min_action, self.max_action)
        low = -self.default_params.max_action
        high = self.default_params.max_action
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = jnp.clip(action, low, high)
        return action


# ------------------- Reward ------------------- #


class CarParkRewardParams(NamedTuple):
    destination = jnp.array(CarParkConsts.DESTINATION)
    bottom_margin = jnp.array(CarParkConsts.BOTTOM_MARGIN)
    top_margin = jnp.array(CarParkConsts.TOP_MARGIN)
    reward_scale = jnp.array(CarParkConsts.REWARD_SCALE)
    reward_margin = jnp.array(CarParkConsts.REWARD_MARGIN)
    value_at_margin = jnp.array(CarParkConsts.VALUE_AT_MARGIN)
    action_cost = jnp.array(CarParkConsts.ACTION_COST)


class CarParkReward(RewardModel):
    def __init__(self) -> None:
        super().__init__()
        reward_params = CarParkRewardParams()
        self.default_params = reward_params

        self.tolerance_reward = ToleranceReward(
            bounds=(reward_params.bottom_margin, reward_params.top_margin),
            margin=reward_params.reward_margin,
            value_at_margin=reward_params.value_at_margin,
            scale=reward_params.reward_scale,
        )

    def init(self, key=None):
        return self.default_params

    @partial(jit, static_argnums=(0, 4))
    def predict(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        reward_params: CarParkRewardParams = None,
    ) -> Any:

        if not reward_params:
            reward_params = self.default_params

        position, speed = next_obs

        # distanceToGoal = (
        #     reward_params.destination - position
        # )  # Margin will be before the destination
        distanceToGoal = (
            position - reward_params.destination
        )  # Margin will be after the destination

        # state_reward = -(distanceToGoal**2)

        state_reward = self.tolerance_reward(distanceToGoal)

        speed_cost = CarParkConsts.SPEED_COST
        speed_reward = -speed_cost * (speed**2)

        action_cost = reward_params.action_cost
        action_reward = -action_cost * (action**2)  # Action reward on unscaled action

        reward = state_reward + speed_reward + action_reward

        max_reward = reward_params.reward_scale
        reward = reward - max_reward  # Cumulative reward gets stagnated

        return reward


# ------------------- Cost ------------------- #


class CarParkCostParams(NamedTuple):
    max_position: jnp.array
    max_speed: jnp.array


class CarParkCost(CostModel):
    def __init__(
        self,
        max_position: float = CarParkConsts.ROAD_LENGTH,
        max_speed: float = CarParkConsts.MAX_SPEED,
    ) -> None:
        super().__init__()
        self.default_params = CarParkCostParams(
            max_position=jnp.array(max_position), max_speed=jnp.array(max_speed)
        )

    def init(self, key=None):
        return self.default_params

    @staticmethod
    def get_cost_params(cost_params: CarParkCostParams):
        max_position = cost_params.max_position
        max_speed = cost_params.max_speed

        return (max_position, max_speed)

    @partial(jit, static_argnums=(0, 4))
    def predict(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        cost_params: CarParkCostParams = None,
    ) -> jnp.ndarray:
        if not cost_params:
            cost_params = self.default_params

        max_position, max_speed = self.get_cost_params(cost_params)

        # position, speed = obs
        position, speed = next_obs

        # Hard constraint on position
        condition = position > max_position

        def true_branch(position):
            return jnp.ones_like(position)

        def false_branch(position):
            return jnp.zeros_like(position)

        cost = jax.lax.cond(condition, true_branch, false_branch, position)

        return cost


# ------------------- System ------------------- #


class CarParkSystem(DynamicalSystem):
    def __init__(self, dynamics=CarParkDynamics(), reward=CarParkReward()):
        self.dynamics = dynamics
        self.reward = reward

    def __call__(self, obs, action, rng, dynamics_params=None, reward_params=None):
        next_state, reward = self.evaluate(
            obs, action, rng, dynamics_params, reward_params
        )
        return (next_state, reward)

    def init(self, key=None):
        dynamics_params = self.dynamics.init(key)
        reward_params = self.reward.init(key)

        return dynamics_params, reward_params

    # @partial(jax.jit, static_argnums=(0, 3, 4))
    def evaluate(self, obs, action, rng, dynamics_params=None, reward_params=None):
        input = jnp.concatenate([obs, action], axis=-1)
        next_obs = self.dynamics(input, dynamics_params, rng)
        reward = self.reward(obs, action, next_obs, reward_params)

        return (next_obs, reward)


class SafeCarParkSystem(SafeDynamicalSystem):

    def __init__(
        self,
        dynamics=CarParkDynamics(),
        reward=CarParkReward(),
        cost=CarParkCost(),
    ):
        self.dynamics = dynamics
        self.reward = reward
        self.cost = cost

        max_position = cost.default_params.max_position
        self.constraint = lambda x: (x[0] > max_position)
        self.constraint_deviation = lambda x: jnp.abs(x[0]) - max_position

    def __call__(
        self,
        obs,
        action,
        rng,
        dynamics_params=None,
        reward_params=None,
        cost_params=None,
    ):
        next_state, reward, cost = self.evaluate(
            obs, action, rng, dynamics_params, reward_params, cost_params
        )
        return (next_state, reward, cost)

    def init(self, key=None):
        dynamics_params = self.dynamics.init(key)
        reward_params = self.reward.init(key)
        cost_params = self.cost.init(key)

        return dynamics_params, reward_params, cost_params

    # @partial(jax.jit, static_argnums=(0, 3, 4))
    def evaluate(
        self,
        obs,
        action,
        rng,
        dynamics_params=None,
        reward_params=None,
        cost_params=None,
    ):
        input = jnp.concatenate([obs, action], axis=-1)
        next_obs = self.dynamics(input, dynamics_params, rng)
        reward = self.reward(obs, action, next_obs, reward_params)
        cost = self.cost(obs, action, next_obs, cost_params)

        return (next_obs, reward, cost)
