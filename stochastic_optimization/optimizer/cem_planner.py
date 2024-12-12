from functools import partial
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from stochastic_optimization.optimizer.cem import CrossEntropyMethod
from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    DynamicalSystem,
)

# Flags

RENDER = True
DEBUG = False


class CEMPlanner:

    def __init__(
        self,
        dynamical_system: DynamicalSystem,
        optimizer: CrossEntropyMethod,
        num_particles: int = 3,
    ) -> None:
        self.dynamical_system = dynamical_system
        self.optimizer = optimizer
        self.num_particles = num_particles
        self.mean_estimate = None

    @partial(jax.jit, static_argnums=(0, 1))
    def optimize_trajectory(
        self, optimize_fn, init_obs, model_params, eval_key, optimizer_key
    ):
        def objective_fn(action_seq):
            def trajectory_reward(obs, rollout_key):
                rollout_fn = simulate_trajectory

                _, rewards = rollout_fn(
                    dynamics=self.dynamical_system.evaluate,
                    model_params=model_params,
                    init_obs=obs,
                    action_seq=action_seq,
                    key=rollout_key,
                )
                avg_rewards = optimize_fn(rewards)
                return avg_rewards

            rollout_key = jax.random.split(eval_key, num=self.num_particles)
            rewards = jax.vmap(trajectory_reward, in_axes=(None, 0))(
                init_obs, rollout_key
            )
            avg_rewards = jnp.mean(rewards, axis=0)  # Average over the particles
            return avg_rewards

        mean_estimate = self.mean_estimate

        best_seq, best_cost, _, _ = self.optimizer.update(
            objective_fn, init_mean=mean_estimate, rng=optimizer_key
        )

        self.set_mean_estimate(best_seq)

        return best_seq, best_cost

    def set_mean_estimate(self, best_seq):
        next_best_seq = jnp.zeros_like(best_seq)
        next_best_seq = next_best_seq.at[:-1].set(best_seq[1:])
        next_best_seq = next_best_seq.at[-1].set(best_seq[-1])

        self.mean_estimate = next_best_seq

    def reset(self):
        self.mean_estimate = None


@partial(jax.jit, static_argnums=0)
def simulate_trajectory(dynamics, model_params, init_obs, action_seq, key):
    def step(carry, ins):
        obs, key = carry
        action = ins

        key, eval_key = jax.random.split(key, 2)

        next_obs, reward = dynamics(
            obs=obs, action=action, rng=eval_key, dynamics_params=model_params
        )

        carry = (next_obs, key)
        outs = (next_obs, reward)

        return carry, outs

    carry = (init_obs, key)
    carry, outs = jax.lax.scan(step, carry, action_seq)

    return outs
