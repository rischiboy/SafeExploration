from functools import partial
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from typing import Optional, Tuple, Union
from jax.random import multivariate_normal
from jax.scipy.stats import multivariate_normal as dist

MIN_STD = 1e-6


class CrossEntropyMethod:
    def __init__(
        self,
        action_dim: Tuple,
        num_elites: int,
        num_iter: int,
        num_samples: int,
        alpha: float = 0.0,
        horizon: int = 0,
        seed: int = 0,
        lower_bound: float = -jnp.inf,
        upper_bound: float = jnp.inf,
        *args,
        **kwargs,
    ):
        self.action_dim = action_dim
        self.alpha = alpha
        self.num_elites = num_elites
        self.horizon = horizon
        self.num_iter = num_iter
        self.num_samples = num_samples
        self.seed = seed

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if horizon == 0:
            self.sample_dim = action_dim
        else:
            self.sample_dim = (horizon, *action_dim)

    @partial(jit, static_argnums=(0, 1))
    def update(
        self,
        cost_fn,
        init_mean: Optional[jax.Array] = None,
        init_std: Optional[Union[float, jax.Array]] = 5.0,
        rng=None,
    ):
        if init_mean is None:
            mean = jnp.zeros(self.sample_dim, dtype=float)
        else:
            mean = init_mean.astype(float)

        assert mean.shape == self.sample_dim
        mean = mean.squeeze()

        init_std = jax.lax.max(init_std, MIN_STD)
        std = jnp.ones(self.sample_dim) * init_std
        std = std.squeeze()

        if rng is None:
            rng = jax.random.PRNGKey(self.seed)

        # best_cost = jnp.inf
        # best_action = mean
        best_elites = jnp.zeros((self.num_elites, *self.sample_dim))
        elites_costs = jnp.ones(self.num_elites) * jnp.inf

        get_curr_best_action = lambda best_cost, best_action, cost, action: (
            cost,
            action,
        )
        get_best_action = lambda best_cost, best_action, cost, action: (
            best_cost,
            best_action,
        )

        if self.num_iter == 0:
            cost = jnp.zeros(1)
            zero_cost = jnp.zeros_like(elites_costs)
            return mean, cost, best_elites, zero_cost

        def step(carry, ins):
            key, mean, std, best_elites, elites_costs = carry
            best_action = best_elites[0]
            best_cost = elites_costs[0]

            key, sample_key = jax.random.split(key, 2)
            mean = mean.reshape(-1, 1).flatten()
            std = std.reshape(-1, 1).flatten()
            cov = jnp.diag(std**2)

            # Sample trajectory (vector of iid actions) over a fixed horizon
            action_samples = multivariate_normal(
                key=sample_key,
                mean=mean,
                cov=cov,
                shape=(self.num_samples,),
            )

            action_samples = action_samples.reshape(
                (self.num_samples, *self.sample_dim)
            )

            action_samples = self.clip_actions(action_samples)

            # Pick K best-performing samples according to the cost_fn
            sorted_samples, sorted_costs = self.evaluate_samples(
                cost_fn, action_samples
            )
            elites = sorted_samples[: self.num_elites].reshape(best_elites.shape)
            costs = sorted_costs[: self.num_elites].reshape(elites_costs.shape)

            # Updated mean and std according to elites
            mean_update = jnp.mean(elites, axis=0)
            std_update = jnp.std(elites, axis=0)

            # Momentum term
            mean = (
                self.alpha * mean.reshape(self.sample_dim)
                + (1 - self.alpha) * mean_update
            ).squeeze()
            std = (
                self.alpha * std.reshape(self.sample_dim)
                + (1 - self.alpha) * std_update
            ).squeeze()

            # Corresponds to the elite admitting the lowest cost
            curr_best_action = elites[0].squeeze()
            curr_best_cost = costs[0].squeeze()

            # Update best seen action and cost
            # best = jax.lax.cond(
            #     curr_best_cost <= best_cost,
            #     get_curr_best_action,
            #     get_best_action,
            #     best_cost,
            #     best_action,
            #     curr_best_cost,
            #     curr_best_action,
            # )

            best = jax.lax.cond(
                curr_best_cost <= best_cost,
                get_curr_best_action,
                get_best_action,
                elites_costs,
                best_elites,
                costs,
                elites,
            )

            elites_costs, best_elites = best
            best_cost = elites_costs[0]
            best_action = best_elites[0]

            carry = (key, mean, std, best_elites, elites_costs)
            outs = (best_action, best_cost, best_elites, elites_costs)

            return carry, outs

        carry = (rng, mean, std, best_elites, elites_costs)
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.num_iter)

        best_action, best_cost, elites, costs = outs

        # Return the state of the best elites and corresponding costs after the final iteration
        return best_action[-1], best_cost, elites[-1], costs

    @partial(jit, static_argnums=(0, 1))
    def evaluate_samples(self, cost_fn, samples):
        costs = vmap(cost_fn)(samples)

        sorted_indices = jnp.argsort(costs, axis=0)

        sorted_samples = samples[sorted_indices]
        sorted_costs = costs[sorted_indices]

        return sorted_samples, sorted_costs

    def clip_actions(self, actions: jnp.ndarray):
        clipped = jnp.clip(actions, self.lower_bound, self.upper_bound)
        return clipped
