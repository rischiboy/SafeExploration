from functools import partial
import time
from typing import Callable
from jax import jit
import jax.numpy as jnp
import jax
import numpy as np
from brax.training.types import Transition

from bsm.utils.type_aliases import StatisticalModelOutput
from bsm.utils.particle_distribution import ParticleDistribution
from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    SafeDynamicalSystem,
)
from stochastic_optimization.utils.type_utils import SamplingMode


DEBUG = False


def get_dummy_transition(env):
    transition = Transition(
        observation=jnp.zeros(env.observation_space.shape),
        action=jnp.zeros(env.action_space.shape),
        reward=jnp.asarray(0),
        discount=jnp.asarray(0),
        next_observation=jnp.zeros(env.observation_space.shape),
    )
    return transition


def rollout_trajectory(
    env,
    policy: Callable,
    init_obs: jnp.ndarray,
    model_params,
    optimizer_rng: jnp.ndarray = None,
    num_steps: int = 100,
    reset_on_end: bool = False,
):
    """
    Rollout a trajectory in the environment using a given policy.

    Args:
        env: The environment on which the rollout is performed.
        policy: The policy model to use for selecting actions.
        init_obs: The initial state of the environment.
        model_params: The parameters of the policy model.
        num_steps: The number of steps to rollout the trajectory for.

    Returns:
        transition: The transition object containing the rollout trajectory.
        obs: The final observation after the trajectory rollout.
    """
    if optimizer_rng is None:
        key = jax.random.PRNGKey(seed=0)
    else:
        key = optimizer_rng

    obs = init_obs

    observations = []
    actions = []
    rewards = []
    for i in range(num_steps):
        key, optimizer_key = jax.random.split(key, 2)

        start = time.time()
        best_action = policy(obs, model_params, optimizer_key)
        predict_time = time.time() - start
        next_obs, reward, finished, truncate, _ = env.step(best_action)
        step_time = time.time() - start - predict_time

        if DEBUG:
            print(f"Predict time: {predict_time}s")
            print(f"Step time: {step_time}s")
            print(f"Total time: {predict_time + step_time}")

            if finished:
                print("Episode done!")
            if truncate:
                print("Episode truncated!")

        observations.append(obs)
        actions.append(best_action)
        rewards.append(reward)

        if reset_on_end:
            if finished or truncate:
                next_obs, _ = env.reset()

        obs = next_obs

    # Append the last observation which belongs to the next_state
    observations.append(obs)

    transition = Transition(
        observation=jnp.array(observations[:-1]),
        action=jnp.array(actions),
        reward=jnp.array(rewards),
        discount=jnp.ones((num_steps,)),
        next_observation=jnp.array(observations[1:]),
    )

    return transition, obs


def generate_random_transitions(env, num_transitions: int):
    obs_shape = (num_transitions,) + env.observation_space.shape
    action_space = (num_transitions,) + env.action_space.shape
    obs_vec = np.zeros(obs_shape)
    action_vec = np.zeros(action_space)
    reward_vec = np.zeros((num_transitions,))
    next_obs_vec = np.zeros(obs_shape)

    dynamics = env.dynamical_system

    for i in range(num_transitions):
        obs = env.observation_space.sample()
        action = env.action_space.sample()
        obs_vec[i] = obs
        action_vec[i] = action
        if isinstance(dynamics, SafeDynamicalSystem):
            next_obs, reward, _ = dynamics.evaluate(obs, action, rng=None)
        else:
            next_obs, reward = dynamics.evaluate(obs, action, rng=None)
        reward_vec[i] = reward
        next_obs_vec[i] = next_obs

    transitions = Transition(
        observation=obs_vec,
        action=action_vec,
        reward=reward_vec,
        discount=np.ones((num_transitions,)),
        next_observation=next_obs_vec,
    )

    return transitions


def prepare_model_input(observations: jnp.ndarray, actions: jnp.ndarray):
    assert (
        observations.shape[:-1] == actions.shape[:-1]
    ), "Shapes of observations and actions do not match along leading dimensions"

    input = jnp.concatenate([observations, actions], axis=-1)
    return input


def generate_train_data(env, episode_len: int, num_episodes: int, rng: jnp.ndarray):
    """
    Generate training data for the dynamics model by rolling out trajectories in the environment.

    Args:
        episode_len: The length of each episode.
        num_episodes: The number of episodes to rollout.

    Returns:
        transitions: The transitions object containing the training data.
    """

    random_policy = lambda obs, model_params, rng: env.action_space.sample()

    training_data = None

    for i in range(num_episodes):
        obs, _ = env.reset_any()
        transitions, _ = rollout_trajectory(
            env, random_policy, obs, rng, num_steps=episode_len
        )

        if training_data is None:
            training_data = transitions
        else:
            training_data = merge_transitions(training_data, transitions)

    return training_data


def merge_transitions(tran1: Transition, tran2: Transition):
    obs = jnp.concatenate([tran1.observation, tran2.observation], axis=0)
    action = jnp.concatenate([tran1.action, tran2.action], axis=0)
    reward = jnp.concatenate([tran1.reward, tran2.reward], axis=0)
    discount = jnp.concatenate([tran1.discount, tran2.discount], axis=0)
    next_obs = jnp.concatenate([tran1.next_observation, tran2.next_observation], axis=0)

    merged = Transition(
        observation=obs,
        action=action,
        reward=reward,
        discount=discount,
        next_observation=next_obs,
    )

    return merged


# Function to format floats to 4 decimals
def format_floats(val):
    if isinstance(val, float):
        return "{:.4f}".format(val)
    elif isinstance(val, list):
        return [format_floats(v) for v in val]
    return val


# Sampling functions


@jit
def uniform_sampling(low, high, rng):
    return jax.random.uniform(rng, minval=low, maxval=high, shape=low.shape)


@jit
def sample_normal_dist(mu, sig, rng):
    return mu + jax.random.normal(rng, mu.shape) * sig


@partial(jit, static_argnums=(1))
def sample(pred: StatisticalModelOutput, sampling_mode: int, rng: jnp.ndarray = None):
    if sampling_mode == SamplingMode.MEAN.value:
        next_obs = pred.mean
    elif sampling_mode == SamplingMode.NOISY_MEAN.value:
        assert rng is not None, "RNG must be provided for sampling mode NOISY_MEAN!"
        next_obs = sample_normal_dist(
            pred.mean,
            pred.aleatoric_std,
            rng,
        )
    elif sampling_mode == SamplingMode.DIST.value:
        assert rng is not None, "RNG must be provided for sampling mode DIST!"
        total_std = get_total_std(pred)
        next_obs = sample_normal_dist(pred.mean, total_std, rng)
    else:
        raise NotImplementedError(f"Sampling mode {sampling_mode} is not implemented!")

    return next_obs


@jit
def sample_from_particle_dist(dist: ParticleDistribution, rng: jnp.ndarray):
    assert (
        rng is not None
    ), "RNG must be provided for sampling from particle distribution!"

    next_obs = dist.sample_particle(rng)
    return next_obs


def get_total_std(pred: StatisticalModelOutput):
    assert pred is not None, "Prediction must not be None!"
    assert pred.aleatoric_std is not None, "Prediction aleatoric std must not be None!"
    assert pred.epistemic_std is not None, "Prediction epistemic std must not be None!"

    return jnp.sqrt(pred.aleatoric_std**2 + pred.epistemic_std**2)
