# import sys

# sys.path.append(r"C:\Users\risch\Desktop\MasterThesis\Code\safe_exploration")

import pytest
from stochastic_optimization.optimizer.cem import CrossEntropyMethod
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import time

# Objective functions


def sum_of_squares(x: jnp.array):
    return jnp.sum(x**2)


def sum_squared_error(x, bias):
    return jnp.sum(jnp.square(x - bias))


# Main

# CEM parameters
action_dim = (5, 2)
num_elites = 50
num_iter = 20
num_samples = 500
lower_bound = -5
upper_bound = 5

# Test parameters
EPS = 5e-2
num_bias_tests = 5

# Create and run optimizer
config = {
    "action_dim": action_dim,
    "num_elites": num_elites,
    "num_iter": num_iter,
    "num_samples": num_samples,
    "lower_bound": lower_bound,
    "upper_bound": upper_bound,
}
optimizer = CrossEntropyMethod(**config)

objective_func = sum_of_squares

best_action, best_cost, elites, elites_costs = optimizer.update(
    objective_func, init_std=5.0
)


# Helper functions


@partial(jit, static_argnums=(0))
def optimize_for_bias(optimizer: CrossEntropyMethod, bias: float):
    func = lambda x: sum_squared_error(x, bias)
    start_time = time.time()
    sequence, cost, elites, costs = optimizer.update(func, init_std=5.0)
    elapsed_time = time.time() - start_time
    print(f"Optimization time: {elapsed_time}s")
    return sequence, cost


# Tests


def test_optimizer_accuracy():
    for bias in jnp.linspace(-5, 5, num_bias_tests):
        sequence, cost = optimize_for_bias(optimizer, bias)
        deviation = jnp.max(jnp.abs(sequence - bias))

        assert deviation <= EPS


def test_output_shapes():
    assert best_action.shape == action_dim
    assert best_cost.shape == (num_iter,)
    assert elites.shape == (num_elites, *action_dim)
    assert elites_costs.shape == (num_iter, num_elites)


def test_output_types():
    assert best_action.dtype == jnp.float32
    assert best_cost.dtype == jnp.float32
    assert elites.dtype == jnp.float32
    assert elites_costs.dtype == jnp.float32


def test_action_validity():
    assert jnp.all(best_action >= lower_bound)
    assert jnp.all(best_action <= upper_bound)
    assert jnp.all(elites >= lower_bound)
    assert jnp.all(elites <= upper_bound)


def test_elites_validity():
    # Costs are sorted in ascending order
    for i in range(num_iter):
        assert jnp.allclose(elites_costs[i], jnp.sort(elites_costs[i]))

    # Elite candidates correspond to the respective costs
    for j in range(num_elites):
        candidate = elites[j]
        candidate_cost = elites_costs[-1][j]
        assert jnp.isclose(candidate_cost, objective_func(candidate), rtol=1e-5)

    assert jnp.all(best_action == elites[0])
