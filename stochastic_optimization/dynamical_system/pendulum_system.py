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
    SimpleDynamicalSystem,
    SimpleDynamicsModel,
    CostModel,
    RewardModel,
    SimpleSafeDynamicalSystem,
)


# ------------------- Dynamics ------------------- #
class PendulumDynamicsParams(NamedTuple):
    g = jnp.array(10.0)
    m = jnp.array(1.0)
    l = jnp.array(1.0)
    dt = jnp.array(0.05)
    max_speed = jnp.array(12.0)
    max_torque = jnp.array(2.0)


class PendulumTrueDynamics(SimpleDynamicsModel):
    def __init__(
        self,
        min_action=-1.0,
        max_action=1.0,
        dynamics_params=PendulumDynamicsParams(),
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.input_dim = 4
        self.output_dim = 3

        self.min_action = min_action
        self.max_action = max_action

        self.default_params = dynamics_params

    def __call__(self, input, dynamics_params, rng=None):
        assert input.shape == (self.input_dim,)
        outs = self.predict(input, dynamics_params, rng)

        return outs

    @staticmethod
    def get_dynamics_params(dynamics_params: PendulumDynamicsParams):
        g = dynamics_params.g
        m = dynamics_params.m
        l = dynamics_params.l
        dt = dynamics_params.dt
        max_speed = dynamics_params.max_speed
        max_torque = dynamics_params.max_torque

        return (g, m, l, dt, max_speed, max_torque)

    @staticmethod
    @jit
    def get_obs(state):
        theta, thdot = state
        obs = jnp.array([jnp.cos(theta), jnp.sin(theta), thdot])

        return obs

    @staticmethod
    @jit
    def get_state(obs):
        theta = jnp.arctan2(obs[1], obs[0])
        thdot = obs[-1]

        return jnp.array([theta, thdot])

    def next_state(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        dynamics_params: PendulumDynamicsParams = None,
    ):
        if not dynamics_params:
            dynamics_params = self.default_params

        # g, m, l, dt, max_speed, max_torque = dynamics_params.get_params()
        g, m, l, dt, max_speed, max_torque = self.get_dynamics_params(dynamics_params)

        th, thdot = self.get_state(obs)

        # action = jnp.clip(action, -max_torque, max_torque)[0]
        action = self.rescale_action(action)[0]

        omega = 3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l**2) * action
        newthdot = thdot + omega * dt
        # newthdot = jnp.clip(newthdot, -max_speed, max_speed)

        newth = th + newthdot * dt
        next_obs = self.get_obs(jnp.array([newth, newthdot]))
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
        low = -self.default_params.max_torque
        high = self.default_params.max_torque
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = jnp.clip(action, low, high)
        return action


class PendulumStateDynamics(PendulumTrueDynamics):
    def __init__(self, dynamics_params: PendulumDynamicsParams = None):
        super().__init__(dynamics_params)

        self.input_dim = 3
        self.output_dim = 2

    @partial(jit, static_argnums=(0, 2))
    def predict_single(self, input, dynamics_params=None, rng=None):
        state = input[:2]
        action = input[-1]
        next_state = self.next_state(state, action, dynamics_params)
        return next_state

    def next_state(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        dynamics_params: PendulumDynamicsParams = None,
    ):
        obs = self.get_obs(state)
        next_obs = super().next_state(obs, action, dynamics_params)
        next_state = self.get_state(next_obs)

        return next_state


# ------------------- Reward ------------------- #


class PendulumRewardParams(NamedTuple):
    control_cost: jnp.ndarray
    angle_cost = jnp.array(1.0)
    target_angle = jnp.array(0.0)
    min_action = jnp.array(-2.0)
    max_action = jnp.array(2.0)


class PendulumReward(RewardModel):
    def __init__(self, control_cost: float = 0.001) -> None:
        super().__init__()
        self.default_params = PendulumRewardParams(control_cost=jnp.array(control_cost))

    def init(self, key=None):
        return self.default_params

    @staticmethod
    def get_reward_params(reward_params: PendulumRewardParams):
        control_cost = reward_params.control_cost
        angle_cost = reward_params.angle_cost
        target_angle = reward_params.target_angle
        min_action = reward_params.min_action
        max_action = reward_params.max_action

        return (control_cost, angle_cost, target_angle, min_action, max_action)

    @partial(jit, static_argnums=(0, 4))
    def predict(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        reward_params: PendulumRewardParams = None,
    ) -> Any:
        def angle_normalize(x):
            return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        if not reward_params:
            reward_params = self.default_params

        # control_cost, angle_cost, target_angle = reward_params.get_params()
        (
            control_cost,
            angle_cost,
            target_angle,
            min_action,
            max_action,
        ) = self.get_reward_params(reward_params)

        action = jnp.clip(action, min_action, max_action)[0]

        x, y, thdot = obs
        theta = jnp.arctan2(y, x)

        angle_diff = angle_normalize(theta - target_angle)

        cost = angle_cost * angle_diff**2 + 0.1 * thdot**2 + control_cost * (action**2)

        reward = -cost

        return reward


class PendulumStateReward(PendulumReward):
    def __init__(self, control_cost: float = 0.001) -> None:
        super().__init__(control_cost=control_cost)

    @partial(jit, static_argnums=(0, 4))
    def predict(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        next_state: jnp.ndarray,
        reward_params: PendulumRewardParams = None,
    ):
        obs = PendulumTrueDynamics.get_obs(state)
        next_obs = PendulumTrueDynamics.get_obs(next_state)

        reward = super().predict(obs, action, next_obs, reward_params)
        return reward


# ------------------- Cost ------------------- #


class DefaultPendulumCostParams(NamedTuple):
    max_speed: float = 6.0


class PendulumCostParams(NamedTuple):
    max_speed: jnp.array


class PendulumCost(CostModel):
    def __init__(self, max_speed: float = 8.0) -> None:
        super().__init__()
        self.default_params = PendulumCostParams(max_speed=jnp.array(max_speed))

    def init(self, key=None):
        return self.default_params

    @staticmethod
    def get_cost_params(cost_params: PendulumCostParams):
        max_speed = cost_params.max_speed

        return max_speed

    @partial(jit, static_argnums=(0, 4))
    def predict(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        cost_params: PendulumCostParams = None,
    ) -> jnp.ndarray:
        if not cost_params:
            cost_params = self.default_params

        max_speed = self.get_cost_params(cost_params)

        # x, y, thdot = obs
        x, y, thdot = next_obs

        # Hard constraint on velocity
        # cost = 1 * [thdot > max_speed or thdot < -max_speed]

        condition = (thdot > max_speed) | (thdot < -max_speed)

        def true_branch(thdot):
            return jnp.ones_like(thdot)

        def false_branch(thdot):
            return jnp.zeros_like(thdot)

        cost = jax.lax.cond(condition, true_branch, false_branch, thdot)

        return cost


class PendulumStateCost(PendulumCost):

    def __init__(self, max_speed: float = 8.0) -> None:
        super().__init__(max_speed)

    @partial(jit, static_argnums=(0, 4))
    def predict(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        next_state: jnp.ndarray,
        cost_params: PendulumCostParams = None,
    ) -> jnp.ndarray:

        obs = PendulumTrueDynamics.get_obs(state)
        next_obs = PendulumTrueDynamics.get_obs(next_state)

        cost = super().predict(obs, action, next_obs, cost_params)
        return cost


# ------------------- System ------------------- #


class PendulumSystem(SimpleDynamicalSystem):
    def __init__(
        self, env=None, dynamics=PendulumTrueDynamics(), reward=PendulumReward()
    ):
        super().__init__(dynamics, reward)

        self.env = env

        # For evaluation purposes
        max_speed = DefaultPendulumCostParams().max_speed
        self.constraint = lambda x: (x[-1] > max_speed) | (x[-1] < -max_speed)
        self.constraint_deviation = lambda x: jnp.abs(x[-1]) - max_speed


class PendulumStateSystem(PendulumSystem):
    def __init__(self, dynamics=PendulumStateDynamics(), reward=PendulumStateReward()):
        super().__init__(dynamics, reward)


class SafePendulumSystem(SimpleSafeDynamicalSystem):
    def __init__(
        self,
        env=None,
        dynamics=PendulumTrueDynamics(),
        reward=PendulumReward(),
        cost=PendulumCost(),
    ):
        super().__init__(dynamics, reward, cost)

        self.env = env

        max_speed = cost.default_params.max_speed
        self.constraint = lambda x: (x[-1] > max_speed) | (x[-1] < -max_speed)
        self.constraint_deviation = lambda x: jnp.abs(x[-1]) - max_speed


class SafePendulumStateSystem(SafePendulumSystem):
    def __init__(
        self,
        dynamics=PendulumStateDynamics(),
        reward=PendulumStateReward(),
        cost=PendulumStateCost(),
    ):
        super().__init__(dynamics, reward, cost)
