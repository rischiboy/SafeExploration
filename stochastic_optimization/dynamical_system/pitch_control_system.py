from functools import partial
import jax
from jax import jit
from jax import vmap
import numpy as np
import jax.numpy as jnp
from typing import Any, NamedTuple

from bsm.statistical_model.abstract_statistical_model import StatisticalModel
from bsm.utils.type_aliases import StatisticalModelOutput, StatisticalModelState

from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    CostModel,
    RewardModel,
    SimpleDynamicalSystem,
    SimpleDynamicsModel,
    SimpleSafeDynamicalSystem,
)
from stochastic_optimization.utils.integrator import runge_kutta_45


# ------------------- Dynamics ------------------- #


class PitchControlDynamicsParams(NamedTuple):
    omega = jnp.array(56.7)
    cld = jnp.array(0.313)
    cmld = jnp.array(0.0139)
    cw = jnp.array(0.232)
    cm = jnp.array(0.426)
    eta = jnp.array(0.0875)
    step_size = jnp.array(0.05)
    max_elevator_angle = jnp.array(1.4)


class PitchControlDynamics(SimpleDynamicsModel):

    def __init__(
        self,
        num_integration_steps=1,
        min_action=-1.0,
        max_action=1.0,
        dynamics_params=PitchControlDynamicsParams(),
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.input_dim = 4
        self.output_dim = 3

        self.min_action = min_action
        self.max_action = max_action

        self.num_integration_steps = num_integration_steps

        self.default_params = dynamics_params

    def __call__(self, input, dynamics_params, rng=None):
        assert input.shape == (self.input_dim,)
        outs = self.predict(input, dynamics_params, rng)

        return outs

    @staticmethod
    def get_dynamics_params(dynamics_params: PitchControlDynamicsParams):
        omega = dynamics_params.omega
        cld = dynamics_params.cld
        cmld = dynamics_params.cmld
        cw = dynamics_params.cw
        cm = dynamics_params.cm
        eta = dynamics_params.eta
        step_size = dynamics_params.step_size

        return (omega, cld, cmld, cw, cm, eta, step_size)

    def next_state(
        self, obs, action, dynamics_params: PitchControlDynamicsParams = None
    ):
        if not dynamics_params:
            dynamics_params = self.default_params

        action = self.rescale_action(action)[0]

        def ode(t, obs):
            alpha, q, theta = obs
            u = action

            omega, cld, cmld, cw, cm, eta, step_size = self.get_dynamics_params(
                dynamics_params
            )

            alpha_dot = -cld * alpha + omega * q + cw * u
            q_dot = -cmld * alpha - cm * q + eta * cw * u
            theta_dot = omega * q

            derivatives = jnp.array([alpha_dot, q_dot, theta_dot])
            return derivatives

        h = dynamics_params.step_size / self.num_integration_steps
        next_obs = runge_kutta_45(
            ode, 0, obs, step_size=h, num_steps=self.num_integration_steps
        )

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
        low = -self.default_params.max_elevator_angle
        high = self.default_params.max_elevator_angle
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = jnp.clip(action, low, high)
        return action


# ------------------- Reward ------------------- #


class PitchControlRewardParams(NamedTuple):
    action_cost = jnp.array(0.02)
    pitch_cost = jnp.array(2.0)
    indicator = False


class PitchControlReward(RewardModel):
    def __init__(self, desired_angle=0.0, reward_params=None):
        super().__init__()

        self.desired_angle = desired_angle
        self.default_params = PitchControlRewardParams()

    def init(self, key=None):
        return self.default_params

    @staticmethod
    def get_reward_params(reward_params):
        action_cost = reward_params.action_cost
        pitch_cost = reward_params.pitch_cost
        indicator = reward_params.indicator

        return (action_cost, pitch_cost, indicator)

    @partial(jit, static_argnums=(0, 4))
    def predict(self, obs, action, next_obs, reward_params=None):
        if not reward_params:
            reward_params = self.default_params

        action_cost, pitch_cost, indicator = self.get_reward_params(reward_params)

        alpha, q, theta = obs
        (u,) = action

        # Cost function
        reward = -pitch_cost * (theta - self.desired_angle) ** 2 - action_cost * u**2
        cost = -reward

        # if indicator:
        #     state_constraint = 1 * [theta >= 0]
        #     action_constraint = jnp.abs(u) >= 1.0
        # else:
        #     state_constraint = theta
        #     action_constraint = jnp.abs(u) - 1.0

        return reward

    def set_desired_angle(self, desired_angle):
        self.desired_angle = desired_angle


# ------------------- Cost ------------------- #


class DefaultPitchControlCostParams(NamedTuple):
    max_angle: jnp.array = jnp.array(0.0)


class PitchControlCostParams(NamedTuple):
    max_angle: jnp.array


class PitchControlCost(CostModel):
    def __init__(self, max_angle: float = 0.0) -> None:
        super().__init__()
        self.default_params = PitchControlCostParams(max_angle=jnp.array(max_angle))

    def init(self, key=None):
        return self.default_params

    @staticmethod
    def get_cost_params(cost_params: PitchControlCostParams):
        max_angle = cost_params.max_angle

        return max_angle

    @partial(jit, static_argnums=(0, 4))
    def predict(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        cost_params: PitchControlCostParams = None,
    ) -> jnp.ndarray:
        if not cost_params:
            cost_params = self.default_params

        max_angle = self.get_cost_params(cost_params)

        # alpha, q, theta = obs
        alpha, q, theta = next_obs

        condition = theta >= max_angle

        def true_branch(theta):
            return jnp.ones_like(theta)

        def false_branch(theta):
            return jnp.zeros_like(theta)

        cost = jax.lax.cond(condition, true_branch, false_branch, theta)

        return cost


# ------------------- System ------------------- #


class PitchControlSystem(SimpleDynamicalSystem):
    def __init__(
        self, env=None, dynamics=PitchControlDynamics(), reward=PitchControlReward()
    ):
        super().__init__(dynamics, reward)

        self.env = env

        # For evaluation purposes
        max_angle = DefaultPitchControlCostParams().max_angle
        self.constraint = lambda x: x[-1] >= max_angle
        self.constraint_deviation = lambda x: x[-1] - max_angle


class SafePitchControlSystem(SimpleSafeDynamicalSystem):

    def __init__(
        self,
        env=None,
        dynamics=PitchControlDynamics(),
        reward=PitchControlReward(),
        cost=PitchControlCost(),
    ):
        super().__init__(dynamics, reward, cost)

        self.env = env

        max_angle = cost.default_params.max_angle
        self.constraint = lambda x: x[-1] >= max_angle
        self.constraint_deviation = lambda x: x[-1] - max_angle
