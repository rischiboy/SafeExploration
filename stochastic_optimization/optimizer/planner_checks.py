from functools import partial
import time
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from stochastic_optimization.optimizer.cem import CrossEntropyMethod
from stochastic_optimization.dynamical_system.pendulum_system import (
    PendulumCost,
    PendulumSystem,
    SafePendulumSystem,
)
from stochastic_optimization.environment.pendulum_env import (
    ConstrainedPendulum,
)

from stochastic_optimization.dynamical_system.dynamics_model import BNNDynamicsModel
from stochastic_optimization.optimizer.cem_planner import CEMPlanner

from stochastic_optimization.optimizer.min_max import (
    MinMaxOptimizer,
    OptVarConstants,
    OptVarParams,
)
from stochastic_optimization.optimizer.min_max_planner import MinMaxPlanner
from stochastic_optimization.optimizer.safe_cem_planner import SafeCEMPlanner
from stochastic_optimization.optimizer.utils import (
    mean_reward,
    plan,
    relu_augmented_lagragian,
)

from bsm.bayesian_regression.bayesian_neural_networks.fsvgd_ensemble import (
    ProbabilisticFSVGDEnsemble,
)


###################
### CEM Planner ###
###################
def run_cem_planner(env, model_params, num_sim_steps, obs, rng):

    print("Running CEM planner")

    print("Initial observation: ", obs)

    start_time = time.time()
    transition, _ = plan(
        env=env,
        planner=cem_planner,
        optimize_fn=optimize_fn,
        init_obs=obs,
        model_params=model_params,
        rng=rng,
        num_steps=num_sim_steps,
    )
    elapsed_time = time.time() - start_time
    print(f"Total planning time: {elapsed_time:.3f}s")

    return transition


########################
### Safe CEM Planner ###
########################
def run_safe_cem_planner(env, model_params, num_sim_steps, obs, rng):

    print("Running Safe CEM planner")

    print("Initial observation: ", obs)

    start_time = time.time()
    transition, _ = plan(
        env=env,
        planner=safe_cem_planner,
        optimize_fn=safe_optimize_fn,
        init_obs=obs,
        model_params=model_params,
        rng=rng,
        num_steps=num_sim_steps,
    )
    elapsed_time = time.time() - start_time
    print(f"Total planning time: {elapsed_time:.3f}s")

    return transition


#####################
### MinMaxPlanner ###
#####################
def run_min_max_planner(env, model_params, num_sim_steps, obs, rng):

    print("Running Min-Max planner")

    print("Initial observation: ", obs)

    start_time = time.time()
    transition, _ = plan(
        env=ENV,
        planner=min_max_planner,
        optimize_fn=safe_optimize_fn,
        init_obs=obs,
        model_params=model_params,
        rng=rng,
        num_steps=num_sim_steps,
    )
    elapsed_time = time.time() - start_time
    print(f"Total planning time: {elapsed_time:.3f}s")

    return transition


def constraint_check(observations, max_speed):

    violation_idx = (observations[:, -1] > max_speed) | (
        observations[:, -1] < -max_speed
    )
    violations = observations[:, -1][violation_idx]
    return violations


if __name__ == "__main__":
    # CEM parameters for Planning
    horizon = 10  # Planning horizon
    num_elites = 5
    num_iter = 2
    num_samples = 50  # Number of random control sequences to sample

    num_episodes = 1
    num_sim_steps = 20

    # Environment parameters
    original_max_speed = 6.0
    max_speed = 12.0
    max_torque = 2.0
    ENV = ConstrainedPendulum(max_torque=max_torque, max_speed=max_speed)

    # Model and Optimizer Configuration

    state_dim = ENV.observation_space.shape
    action_dim = ENV.action_space.shape

    # CEM configuration
    cem_config = {
        "action_dim": action_dim,
        "horizon": horizon,
        "num_elites": num_elites,
        "num_iter": num_iter,
        "num_samples": num_samples,
        "lower_bound": -2,
        "upper_bound": 2,
    }

    # MinMaxOptimizer configuration
    action_config = {
        "action_dim": (horizon, *action_dim),  # Regular action dimension
        # "horizon": horizon,
        "num_fixed_elites": num_elites,
        "num_elites": num_elites,
        "num_iter": num_iter,
        "num_samples": num_samples,
        "lower_bound": -2,
        "upper_bound": 2,
        "minimum": True,  # Maximize the reward
    }

    hal_action_config = {
        "action_dim": (horizon, *state_dim),  # Hallucinated action dimension
        # "horizon": horizon,
        "num_fixed_elites": num_elites,
        "num_elites": num_elites,
        "num_iter": num_iter,
        "num_samples": num_samples,
        "lower_bound": -1,
        "upper_bound": 1,
        "minimum": False,  # Maximize the violation
    }

    # BNN configuration
    model_config = {
        "seed": 561,
        "input_dim": 4,
        "output_dim": 3,
        "bnn_type": ProbabilisticFSVGDEnsemble,
        "output_stds": jnp.array([0.1, 0.1, 0.1]),
        "beta": jnp.array([1.0, 1.0, 1.0]),
        "weight_decay": 0.001,
        "num_particles": 1,
        "features": [10, 10],
        "train_share": 0.8,
        "num_training_steps": 5,
        "batch_size": 256,
        "eval_frequency": 5,
        "eval_batch_size": 2048,
        "logging_wandb": False,
    }

    # Initialize the system and the planner
    key = jax.random.PRNGKey(seed=404)
    key, model_key = jax.random.split(key, 2)

    # System without safety constraints
    true_dynamical_system = PendulumSystem()

    statistical_model = BNNDynamicsModel(**model_config)
    bnn_dynamical_system = PendulumSystem(dynamics=statistical_model)

    # System with safety constraints
    cost = PendulumCost(max_speed=original_max_speed)

    safe_true_system = SafePendulumSystem(cost=cost)
    safe_bnn_system = SafePendulumSystem(dynamics=statistical_model, cost=cost)

    # Fix the system to be used for planning
    # system = true_dynamical_system
    # safe_system = safe_true_system

    system = bnn_dynamical_system
    safe_system = safe_bnn_system

    if isinstance(system, SafePendulumSystem):
        model_params, reward_params, cost_params = system.init(model_key)
    else:
        model_params, reward_params = system.init(model_key)

    # Initialize the optimization function for Safe CEM planner
    optimize_fn = mean_reward
    safe_optimize_fn = lambda reward, cost: relu_augmented_lagragian(
        reward, cost, d=0.0, lmbda=100.0
    )

    # Initialize the CEM optimizer
    cem_optimizer = CrossEntropyMethod(**cem_config)

    # Initialize the MinMax optimizer
    x_consts = OptVarConstants(**action_config)
    y_consts = OptVarConstants(**hal_action_config)
    var_x = OptVarParams(x_consts)
    var_y = OptVarParams(y_consts)

    min_max_optimizer = MinMaxOptimizer(var_x, var_y)

    # Initialize the CEMPlanner and SafeCEMPlanner
    cem_planner = CEMPlanner(system, cem_optimizer)
    safe_cem_planner = SafeCEMPlanner(safe_system, cem_optimizer)

    # Initialize the MinMaxPlanner
    min_max_planner = MinMaxPlanner(safe_system, min_max_optimizer)

    #################
    ### Run Tests ###
    #################

    obs, _ = ENV.reset_any()
    state = np.array([np.arctan2(obs[1], obs[0]), obs[2]])

    transitions = run_cem_planner(
        env=ENV,
        model_params=model_params,
        num_sim_steps=num_sim_steps,
        obs=obs,
        rng=key,
    )
    violations = constraint_check(transitions.observation, original_max_speed)
    print(violations)

    obs, _ = ENV.reset_any(state=state)
    transitions = run_safe_cem_planner(
        env=ENV,
        model_params=model_params,
        num_sim_steps=num_sim_steps,
        obs=obs,
        rng=key,
    )
    violations = constraint_check(transitions.observation, original_max_speed)
    print(violations)

    obs, _ = ENV.reset_any(state=state)
    transitions = run_min_max_planner(
        env=ENV,
        model_params=model_params,
        num_sim_steps=num_sim_steps,
        obs=obs,
        rng=key,
    )
    violations = constraint_check(transitions.observation, original_max_speed)
    print(violations)
