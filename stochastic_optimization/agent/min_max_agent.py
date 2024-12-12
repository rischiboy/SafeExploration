import gym
import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
from typing import Callable, Optional, Union

from functools import partial
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from stochastic_optimization.agent.abstract_agent import AbstractAgent
from stochastic_optimization.dynamical_system.dynamics_model import BNNDynamicsModel

from stochastic_optimization.optimizer.cem_planner import CEMPlanner
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
from stochastic_optimization.optimizer.min_max_planner import (
    MinMaxPlanner,
    simulate_trajectory,
)

from stochastic_optimization.utils.trainer_utils import prepare_model_input
from stochastic_optimization.utils.scheduler import Scheduler


class MinMaxAgent(AbstractAgent):

    def __init__(
        self,
        action_space: gym.spaces.box,
        observation_space: gym.spaces.box,
        optimize_fn: Callable,
        policy_optimizer: MinMaxPlanner,
        dynamical_system: SafeDynamicalSystem,
        pes_alpha_scheduler: Optional[Scheduler] = None,
    ) -> None:
        self.action_space = action_space
        self.observation_space = observation_space
        self.optimize_fn = optimize_fn
        self.policy_optimizer = policy_optimizer
        self.dynamical_system = dynamical_system
        self.pes_alpha_scheduler = pes_alpha_scheduler

        self.transition_model: BNNDynamicsModel = dynamical_system.dynamics
        self.reward_model: RewardModel = dynamical_system.reward

        if isinstance(dynamical_system, SafeDynamicalSystem):
            self.cost_model: CostModel = dynamical_system.cost
        else:
            self.cost_model = None

    def init(self, key: jnp.ndarray = None):
        assert isinstance(self.dynamical_system, SafeDynamicalSystem)
        model_params, reward_params, cost_params = self.dynamical_system.init(key=key)

        return model_params, reward_params, cost_params

    def update_alpha(self, step: int):
        if self.pes_alpha_scheduler is not None:
            new_alpha = self.pes_alpha_scheduler.update(step)
            self.policy_optimizer.set_pes_alpha(new_alpha)

        return

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
        return_both: bool = False,
    ):
        """
        Pick the best action for a given observation by optimizing the trajectory that admits the lowest cost.
        """

        rng, eval_key, optimizer_key = jax.random.split(rng, 3)

        optimize_trajectory = self.policy_optimizer.optimize_trajectory

        best_seq, best_hal_seq = optimize_trajectory(
            optimize_fn=self.optimize_fn,
            init_obs=obs,
            model_params=model_params,
            eval_key=eval_key,
            optimizer_key=optimizer_key,
        )
        best_action = best_seq[0]

        if return_both:
            best_hal_action = best_hal_seq[0]
            return best_action, best_hal_action

        return best_action

    def predict_violations(
        self,
        constraint: Callable,
        init_obs: jnp.ndarray,
        actions: jnp.ndarray,
        hal_actions: jnp.ndarray,
        model_params: StatisticalModelState[ModelState],
        rng: jnp.ndarray,
    ):

        pes_alpha = self.policy_optimizer.get_pes_alpha()

        results = simulate_trajectory(
            system=self.dynamical_system,
            model_params=model_params,
            init_obs=init_obs,
            action_seq=actions,
            hal_action_seq=hal_actions,
            pes_alpha=pes_alpha,
            key=rng,
        )

        opt_trajectory, pes_trajectory, rewards, costs = results
        violations = vmap(constraint, in_axes=(0))(pes_trajectory)
        num_violations = jnp.sum(violations)

        return num_violations

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
