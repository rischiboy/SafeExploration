import time
import pytest
import jax
import jax.numpy as jnp
import logging

from stochastic_optimization.optimizer.cem import CrossEntropyMethod
from stochastic_optimization.optimizer.cem_planner import CEMPlanner
from stochastic_optimization.optimizer.utils import mean_reward, plan
from stochastic_optimization.dynamical_system.pendulum_system import PendulumSystem
from stochastic_optimization.environment.pendulum_env import (
    CustomPendulum,
    ConstrainedPendulum,
)
from stochastic_optimization.dynamical_system.pitch_control_system import (
    PitchControlSystem,
)
from stochastic_optimization.environment.pitch_control_env import PitchControlEnv

seed = 14
opt_key = jax.random.PRNGKey(seed=seed)

# CEM parameters for Planning
horizon = 20
num_elites = 50
num_iter = 10
num_samples = 500  # Number of random control sequences to sample


# ------------------- Pendulum ------------------- #

# Pendulum test parameters
num_last_obs = 5
target_angle = 0
tolerance = 0.05
stability_duration = 5
num_simulation_steps = 200


@pytest.fixture
def pendulum_env():
    # return CustomPendulum()
    return ConstrainedPendulum(
        angle_tolerance=tolerance,
        stability_duration=stability_duration,
        max_steps=num_simulation_steps,
    )


@pytest.fixture
def pendulum_system():
    return PendulumSystem()


@pytest.fixture
def pendulum_cem_opt(pendulum_env):
    state_dim = pendulum_env.observation_space.shape
    action_dim = pendulum_env.action_space.shape

    config = {
        "action_dim": action_dim,
        "horizon": horizon,
        "num_elites": num_elites,
        "num_iter": num_iter,
        "num_samples": num_samples,
        "lower_bound": -2,
        "upper_bound": 2,
    }

    cem_optimizer = CrossEntropyMethod(**config)
    return cem_optimizer


@pytest.fixture
def pendulum_planner(pendulum_system, pendulum_cem_opt):
    return CEMPlanner(pendulum_system, pendulum_cem_opt)


def is_upright(obs, target_angle, tolerance=0.05):
    theta = jnp.arctan2(obs[:, 1], obs[:, 0])
    return jnp.allclose(theta, target_angle, atol=tolerance)


def test_pendulum_plan(pendulum_env, pendulum_planner):
    init_obs, _ = pendulum_env.reset()
    model_params, reward_params = pendulum_planner.dynamical_system.init()
    start = time.time()
    transitions, done = plan(
        env=pendulum_env,
        planner=pendulum_planner,
        optimize_fn=mean_reward,
        init_obs=init_obs,
        model_params=model_params,
        rng=opt_key,
        num_steps=num_simulation_steps,
    )
    elapsed = time.time() - start
    print(f"Planning time for Pendulum: {elapsed:.3f}s")

    obs_dim = pendulum_env.observation_space.shape
    action_dim = pendulum_env.action_space.shape

    obs = transitions.observation
    actions = transitions.action
    rewards = transitions.reward
    next_obs = transitions.next_observation

    assert obs.shape == next_obs.shape and obs.shape == (num_simulation_steps, *obs_dim)
    assert actions.shape == (num_simulation_steps, *action_dim)
    assert rewards.shape == (num_simulation_steps,)

    assert jnp.logical_and(
        actions >= pendulum_env.action_space.low,
        actions <= pendulum_env.action_space.high,
    ).all()

    # Check for CustomPendulum
    # evaluate_states = states[-num_last_obs:, :]
    # assert is_upright(evaluate_states, target_angle, tolerance=tolerance)

    # Check for ConstrainedPendulum
    assert done[-1] == True

    theta = jnp.arctan2(next_obs[:, 1], next_obs[:, 0])
    print(f"Final pendulum angles: {theta[-10:]}")


# ------------------- Pitch Control ------------------- #

# Pitch control test parameters
init_angle = -0.2
target_angle = 0
tolerance = 0.025
stability_duration = 10
num_simulation_steps = 200


@pytest.fixture
def pitch_control_env():
    return PitchControlEnv(
        init_angle=init_angle,
        desired_angle=target_angle,
        angle_tolerance=tolerance,
        stability_duration=stability_duration,
        max_steps=num_simulation_steps,
    )


@pytest.fixture
def pitch_control_system():
    return PitchControlSystem()


@pytest.fixture
def pitch_control_cem_opt(pitch_control_env):
    state_dim = pitch_control_env.observation_space.shape
    action_dim = pitch_control_env.action_space.shape

    config = {
        "action_dim": action_dim,
        "horizon": horizon,
        "num_elites": num_elites,
        "num_iter": num_iter,
        "num_samples": num_samples,
        "lower_bound": -1.4,
        "upper_bound": 1.4,
    }

    cem_optimizer = CrossEntropyMethod(**config)
    return cem_optimizer


@pytest.fixture
def pitch_control_planner(pitch_control_system, pitch_control_cem_opt):
    return CEMPlanner(pitch_control_system, pitch_control_cem_opt)


def test_pitch_control_plan(pitch_control_env, pitch_control_planner):
    init_obs, _ = pitch_control_env.reset()
    model_params, reward_params = pitch_control_planner.dynamical_system.init()
    start = time.time()
    transitions, done = plan(
        env=pitch_control_env,
        planner=pitch_control_planner,
        optimize_fn=mean_reward,
        init_obs=init_obs,
        model_params=model_params,
        rng=opt_key,
        num_steps=num_simulation_steps,
    )
    elapsed = time.time() - start
    print(f"Planning time for Pitch Control: {elapsed:.3f}s")

    obs_dim = pitch_control_env.observation_space.shape
    action_dim = pitch_control_env.action_space.shape

    obs = transitions.observation
    actions = transitions.action
    rewards = transitions.reward
    next_obs = transitions.next_observation

    assert obs.shape == next_obs.shape and obs.shape == (num_simulation_steps, *obs_dim)
    assert actions.shape == (num_simulation_steps, *action_dim)
    assert rewards.shape == (num_simulation_steps,)

    assert jnp.logical_and(
        actions >= pitch_control_env.action_space.low,
        actions <= pitch_control_env.action_space.high,
    ).all()

    assert done[-1] == True
    print(f"Final pitch angles: {next_obs[-10:, -1]}")
