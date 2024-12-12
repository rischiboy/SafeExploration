import gym
import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
from typing import Callable, Union

from functools import partial
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from stochastic_optimization.agent.abstract_agent import AbstractAgent
from stochastic_optimization.dynamical_system.dynamics_model import BNNDynamicsModel

from stochastic_optimization.optimizer.cem_planner import CEMPlanner
from stochastic_optimization.utils.trainer_utils import prepare_model_input
from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    CostModel,
    DynamicalSystem,
    RewardModel,
    SafeDynamicalSystem,
)

from bsm.utils.normalization import Data
from bsm.utils.type_aliases import (
    ModelState,
    StatisticalModelOutput,
    StatisticalModelState,
)
from bsm.bayesian_regression.bayesian_neural_networks.bnn import BNNState

from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState


class CEMAgent(AbstractAgent):
    def __init__(
        self,
        action_space: gym.spaces.box,
        observation_space: gym.spaces.box,
        optimize_fn: Callable,
        policy_optimizer: CEMPlanner,
        dynamical_system: Union[DynamicalSystem, SafeDynamicalSystem],
    ) -> None:
        self.action_space = action_space
        self.observation_space = observation_space
        self.policy_optimizer = policy_optimizer
        self.dynamical_system = dynamical_system
        self.optimize_fn = optimize_fn

        self.transition_model: BNNDynamicsModel = dynamical_system.dynamics
        self.reward_model: RewardModel = dynamical_system.reward

        if isinstance(dynamical_system, SafeDynamicalSystem):
            self.cost_model: CostModel = dynamical_system.cost
        else:
            self.cost_model = None

    def init(self, key: jnp.ndarray = None):
        if isinstance(self.dynamical_system, SafeDynamicalSystem):
            model_params, reward_params, cost_params = self.dynamical_system.init(
                key=key
            )
        else:
            model_params, reward_params = self.dynamical_system.init(key=key)
            cost_params = None

        return model_params, reward_params, cost_params

    def predict(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        model_params: StatisticalModelState[BNNState],
        rng: jnp.ndarray,
    ):
        input = prepare_model_input(obs, action)

        next_state_pred = self.transition_model(input, model_params, rng)
        reward = None
        cost = None

        # Batched input
        if input.ndim > 1:
            reward = vmap(self.reward_model, in_axes=(0, 0, 0))(
                obs, action, next_state_pred
            )
            if self.cost_model is not None:
                cost = vmap(self.cost_model, in_axes=(0, 0, 0))(
                    obs, action, next_state_pred
                )
        else:
            reward = self.reward_model(obs, action, next_state_pred, reward_params=None)
            if self.cost_model is not None:
                cost = self.cost_model(obs, action, next_state_pred, cost_params=None)

        # transition = Transition(
        #     observation=obs,
        #     action=action,
        #     reward=reward,
        #     discount=1.0,
        #     next_observation=next_state_pred,
        # )

        return next_state_pred, reward, cost

    def validate(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        model_params: StatisticalModelState[BNNState],
        rng: jnp.ndarray,
    ):
        input = prepare_model_input(obs, action)

        model_out, next_obs = self.transition_model.validate(input, model_params, rng)
        reward = None
        cost = None

        # Batched input
        if len(input.shape) > 1:
            reward = vmap(self.reward_model, in_axes=(0, 0, 0))(obs, action, next_obs)
            if self.cost_model is not None:
                cost = vmap(self.cost_model, in_axes=(0, 0, 0))(obs, action, next_obs)
        else:
            reward = self.reward_model(obs, action, next_obs, reward_params=None)
            if self.cost_model is not None:
                cost = self.cost_model(obs, action, next_obs, cost_params=None)

        return model_out, next_obs, reward, cost

    def select_best_action(
        self,
        obs: jnp.ndarray,
        model_params: StatisticalModelState[BNNState],
        rng: jnp.ndarray = None,
    ):
        """
        Pick the best action for a given observation by optimizing the trajectory that admits the lowest cost.
        """

        rng, eval_key, optimizer_key = jax.random.split(rng, 3)

        optimize_trajectory = self.policy_optimizer.optimize_trajectory

        best_seq, best_cost = optimize_trajectory(
            optimize_fn=self.optimize_fn,
            init_obs=obs,
            model_params=model_params,
            eval_key=eval_key,
            optimizer_key=optimizer_key,
        )
        best_action = best_seq[0]

        return best_action

    def train_step(
        self,
        buffer: UniformSamplingQueue,
        buffer_state: ReplayBufferState,
        model_params: StatisticalModelState[BNNState],
        diff_states: bool = False,
    ) -> StatisticalModelState[BNNState]:
        curr_buffer_size = buffer_state.insert_position
        transition_data = buffer_state.data[:curr_buffer_size]
        transitions = buffer._unflatten_fn(transition_data)
        x = prepare_model_input(transitions.observation, transitions.action)
        if diff_states:
            y = transitions.next_observation - transitions.observation
        else:
            y = transitions.next_observation

        train_data = Data(inputs=x, outputs=y)
        model_params = self.transition_model.update(
            stats_model_state=model_params, data=train_data
        )
        return model_params

    def reset_optimizer(self):
        self.policy_optimizer.reset()
