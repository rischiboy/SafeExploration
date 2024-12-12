from functools import partial
import time
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    SafeDynamicalSystem,
)
from bsm.utils.type_aliases import (
    ModelState,
    StatisticalModelOutput,
    StatisticalModelState,
)

from stochastic_optimization.optimizer.min_max_planner import MinMaxPlanner

# Flags

RENDER = True
DEBUG = False


class OptMinMaxPlanner(MinMaxPlanner):

    def __init__(self, opt_alpha: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.opt_alpha = opt_alpha
        self.output_dim = self.dynamical_system.dynamics.output_dim
        return

    def get_opt_alpha(self):
        return self.opt_alpha

    def set_opt_alpha(self, opt_alpha: float):
        self.opt_alpha = opt_alpha

    @partial(jax.jit, static_argnums=(0, 1))
    def optimize_trajectory(
        self,
        optimize_fn: Callable,
        init_obs: jnp.ndarray,
        model_params,
        eval_key: jnp.ndarray,
        optimizer_key: jnp.ndarray,
    ):

        def objective_fn(opt_action_seq, pes_action_seq):
            def trajectory_reward(obs, rollout_key):
                rollout_fn = simulate_trajectory

                _, _, rewards, costs = rollout_fn(
                    system=self.dynamical_system,  # Provide the entire system such that one obtains mean/std of the next state prediction
                    model_params=model_params,
                    init_obs=obs,
                    opt_action_seq=opt_action_seq,
                    pes_action_seq=pes_action_seq,
                    opt_alpha=self.opt_alpha,
                    pes_alpha=self.pes_alpha,
                    key=rollout_key,
                )
                total_value = optimize_fn(rewards, costs)
                return total_value

            rollout_key = jax.random.split(eval_key, num=self.num_particles)
            rewards = jax.vmap(trajectory_reward, in_axes=(None, 0))(
                init_obs, rollout_key
            )
            avg_rewards = jnp.mean(rewards, axis=0)  # Average over the particles
            return avg_rewards

        (
            best_opt_action_seq,
            best_pes_action_seq,
            opt_action_params,
            pes_action_params,
        ) = self.optimizer.update(objective_fn, optimizer_key, self.iterations)

        best_action_seq = best_opt_action_seq[..., self.output_dim :]
        best_opt_action_seq = best_opt_action_seq[..., : self.output_dim]

        self.update_optimizer_params(opt_action_params, pes_action_params)

        # Only return the action sequences not the updates
        return best_action_seq, best_opt_action_seq, best_pes_action_seq


@partial(jax.jit, static_argnums=0)
def simulate_trajectory(
    system: SafeDynamicalSystem,
    model_params: StatisticalModelState[ModelState],
    init_obs: jnp.ndarray,
    opt_action_seq: jnp.ndarray,
    pes_action_seq: jnp.ndarray,
    opt_alpha: float,
    pes_alpha: float,
    key: jnp.ndarray,
):
    def step(carry, ins):
        optimistic_obs, pessimistic_obs, key = carry
        ext_action, hal_action_pes = ins

        output_dim = system.dynamics.output_dim
        hal_action_opt = ext_action[:output_dim]
        action = ext_action[output_dim:]

        ################
        ### Optimism ###
        ################

        optimistic_next_obs, reward, _ = system.evaluate_with_eta(
            optimistic_obs, action, opt_alpha, hal_action_opt, model_params
        )

        ################
        ### Pessimsm ###
        ################

        pessimistic_next_obs, _, cost = system.evaluate_with_eta(
            pessimistic_obs, action, pes_alpha, hal_action_pes, model_params
        )

        carry = (optimistic_next_obs, pessimistic_next_obs, key)
        outs = (optimistic_next_obs, pessimistic_next_obs, reward, cost)

        return carry, outs

    carry = (init_obs, init_obs, key)
    carry, outs = jax.lax.scan(step, carry, (opt_action_seq, pes_action_seq))

    return outs
