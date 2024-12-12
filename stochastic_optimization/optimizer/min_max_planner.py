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
from stochastic_optimization.optimizer.min_max import (
    MinMaxOptimizer,
)

from bsm.utils.type_aliases import (
    ModelState,
    StatisticalModelOutput,
    StatisticalModelState,
)

# Flags

RENDER = True
DEBUG = False


class MinMaxPlanner:

    def __init__(
        self,
        dynamical_system: SafeDynamicalSystem,
        optimizer: MinMaxOptimizer,
        num_particles: int = 3,
        pes_alpha: float = 1.0,
        iterations: int = 1,
    ) -> None:
        self.dynamical_system = dynamical_system
        self.optimizer = optimizer
        self.num_particles = num_particles
        self.pes_alpha = pes_alpha
        self.iterations = iterations

    def get_pes_alpha(self):
        return self.pes_alpha

    def set_pes_alpha(self, pes_alpha: float):
        self.pes_alpha = pes_alpha

    @partial(jax.jit, static_argnums=(0, 1))
    def optimize_trajectory(
        self,
        optimize_fn: Callable,
        init_obs: jnp.ndarray,
        model_params,
        eval_key: jnp.ndarray,
        optimizer_key: jnp.ndarray,
    ):

        def objective_fn(action_seq, hal_action_seq):
            def trajectory_reward(obs, rollout_key):
                rollout_fn = simulate_trajectory

                _, _, rewards, costs = rollout_fn(
                    system=self.dynamical_system,  # Provide the entire system such that one obtains mean/std of the next state prediction
                    model_params=model_params,
                    init_obs=obs,
                    action_seq=action_seq,
                    hal_action_seq=hal_action_seq,
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

        best_action_seq, best_hal_action_seq, action_params, hal_action_params = (
            self.optimizer.update(objective_fn, optimizer_key, self.iterations)
        )

        self.update_optimizer_params(action_params, hal_action_params)

        # Only return the action sequences not the updates
        return best_action_seq, best_hal_action_seq

    def update_optimizer_params(self, x_params, y_params):
        mean_update = self.get_seq_update(x_params["mean"])
        # std_update = self.get_seq_update(x_params["std"])
        elites_update = self.get_seq_update(x_params["fixed_elites"])

        self.optimizer.var_x.mean = mean_update
        self.optimizer.var_x.set_elites(elites_update)

        mean_update = self.get_seq_update(y_params["mean"])
        # std_update = self.get_seq_update(y_params["std"])
        elites_update = self.get_seq_update(y_params["fixed_elites"])

        self.optimizer.var_y.mean = mean_update
        self.optimizer.var_y.set_elites(elites_update)

        return

    def get_seq_update(self, seq):
        next_seq = jnp.zeros_like(seq)

        next_seq = next_seq.at[:-1].set(seq[1:])
        next_seq = next_seq.at[-1].set(seq[-1])

        return next_seq

    def reset(self):
        self.optimizer.reset()


@partial(jax.jit, static_argnums=0)
def simulate_trajectory(
    system: SafeDynamicalSystem,
    model_params: StatisticalModelState[ModelState],
    init_obs: jnp.ndarray,
    action_seq: jnp.ndarray,
    hal_action_seq: jnp.ndarray,
    pes_alpha: float,
    key: jnp.ndarray,
):
    def step(carry, ins):
        obs, pessimistic_obs, key = carry
        action, hal_action = ins

        key, eval_key = jax.random.split(key, 2)

        next_obs, reward, _ = system.evaluate(obs, action, eval_key, model_params)

        ################
        ### Pessimsm ###
        ################

        pessimistic_next_obs, _, cost = system.evaluate_with_eta(
            pessimistic_obs, action, pes_alpha, hal_action, model_params
        )

        carry = (next_obs, pessimistic_next_obs, key)
        outs = (next_obs, pessimistic_next_obs, reward, cost)

        return carry, outs

    carry = (init_obs, init_obs, key)
    carry, outs = jax.lax.scan(step, carry, (action_seq, hal_action_seq))

    return outs
