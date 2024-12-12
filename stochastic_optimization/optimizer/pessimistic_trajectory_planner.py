from functools import partial
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.nn import relu

from stochastic_optimization.optimizer.cem import CrossEntropyMethod
from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    DynamicalSystem,
    SafeDynamicalSystem,
)
from stochastic_optimization.optimizer.cem_planner import CEMPlanner

DEBUG = False


class PesTrajectoryPlanner(CEMPlanner):

    def __init__(
        self,
        safe_dynamical_system: SafeDynamicalSystem,
        optimizer: CrossEntropyMethod,
        num_particles: int = 3,
    ):
        super().__init__(safe_dynamical_system, optimizer, num_particles)
        return

    @partial(jax.jit, static_argnums=(0, 1))
    def optimize_trajectory(
        self, optimize_fn, init_obs, model_params, eval_key, optimizer_key
    ):
        def objective_fn(action_seq):
            def trajectory_reward(obs, rollout_key):
                rollout_fn = simulate_trajectory

                _, rewards, costs = rollout_fn(
                    dynamics=self.dynamical_system.evaluate,
                    model_params=model_params,
                    init_obs=obs,
                    action_seq=action_seq,
                    key=rollout_key,
                )
                total_value = (rewards.mean(), costs.sum())
                return total_value

            rollout_key = jax.random.split(eval_key, num=self.num_particles)
            rewards, costs = jax.vmap(trajectory_reward, in_axes=(None, 0))(
                init_obs, rollout_key
            )
            avg_rewards = optimize_fn(rewards, costs)
            return avg_rewards

        mean_estimate = self.mean_estimate

        best_seq, best_cost, elites, elites_cost = self.optimizer.update(
            objective_fn, init_mean=mean_estimate, rng=optimizer_key
        )

        self.set_mean_estimate(best_seq)

        return best_seq, best_cost


@partial(jax.jit, static_argnums=0)
def simulate_trajectory(dynamics, model_params, init_obs, action_seq, key):
    def step(carry, ins):
        obs, pes_obs, key = carry
        action = ins

        key, eval_key = jax.random.split(key, 2)
        reward_key, cost_key = jax.random.split(eval_key, 2)

        next_obs, reward, _ = dynamics(
            obs=obs, action=action, rng=reward_key, dynamics_params=model_params
        )

        pes_next_obs, _, cost = dynamics(
            obs=pes_obs, action=action, rng=cost_key, dynamics_params=model_params
        )

        carry = (next_obs, pes_next_obs, key)
        outs = (next_obs, reward, cost)

        return carry, outs

    carry = (init_obs, init_obs, key)
    carry, outs = jax.lax.scan(step, carry, action_seq)

    return outs
