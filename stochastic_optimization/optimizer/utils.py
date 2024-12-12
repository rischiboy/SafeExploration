from enum import Enum
import jax
import time
import jax.numpy as jnp
import numpy as np
from typing import Callable, Union
from jax.nn import relu

from brax.training.types import Transition

from stochastic_optimization.optimizer.cem_planner import CEMPlanner
from stochastic_optimization.optimizer.min_max_planner import MinMaxPlanner


""" MPC planning using the Cross-Entropy Method (CEM) """


def plan(
    env,
    planner: CEMPlanner,
    optimize_fn: Callable,
    init_obs: jnp.ndarray,
    model_params,
    rng: jnp.ndarray,
    num_steps: int,
    debug: bool = False,
):
    obs = init_obs
    observations = []
    actions = []
    rewards = []
    done_list = []
    planning_times = []

    key = rng

    for i in range(num_steps):
        key, eval_key, optimizer_key = jax.random.split(key, 3)

        start_time = time.time()
        best_seq, best_cost = planner.optimize_trajectory(
            optimize_fn=optimize_fn,
            init_obs=obs,
            model_params=model_params,
            eval_key=eval_key,
            optimizer_key=optimizer_key,
        )

        opt_time = time.time() - start_time

        best_action = best_seq[0]
        next_obs, reward, done, truncate, info = env.step(best_action)

        step_time = time.time() - opt_time - start_time

        observations.append(obs)
        actions.append(best_action)
        rewards.append(reward)
        done_list.append(done)

        planning_times.append(opt_time)

        obs = next_obs

        if debug:
            # print(f"It {i}: Optimization time -- {opt_time:.8f}s")
            # print(f"It {i}: Step time -- {step_time:.8f}s")
            print(f"Num stabilized steps: {env.stablized_steps}")

    # Append the last observation which belongs to the next_state
    observations.append(obs)

    if debug:
        avg_planning_time = np.mean(planning_times)
        print(f"Average planning time per step: {avg_planning_time:.3f}s")

    transition = Transition(
        observation=jnp.array(observations[:-1]),
        action=jnp.array(actions),
        reward=jnp.array(rewards),
        discount=jnp.ones((num_steps,)),
        next_observation=jnp.array(observations[1:]),
    )

    return transition, done_list


""" MPC Planning using the Min-Max Optimization Framework for Pessimistic Planning """


def min_max_plan(
    env,
    planner: MinMaxPlanner,
    optimize_fn: Callable,
    init_obs: jnp.ndarray,
    model_params,
    rng: jnp.ndarray,
    num_steps: int,
    debug: bool = False,
):

    obs = init_obs
    observations = []
    actions = []
    rewards = []
    done_list = []
    planning_times = []

    key = rng

    for i in range(num_steps):
        key, eval_key, optimizer_key = jax.random.split(key, 3)

        start_time = time.time()
        best_action_seq, best_hal_action_seq = planner.optimize_trajectory(
            optimize_fn=optimize_fn,
            init_obs=obs,
            model_params=model_params,
            eval_key=eval_key,
            optimizer_key=optimizer_key,
        )

        opt_time = time.time() - start_time

        best_action = best_action_seq[0]
        best_hal_action = best_hal_action_seq[0]
        next_obs, reward, done, truncate, info = env.step(best_action)

        step_time = time.time() - opt_time - start_time

        observations.append(obs)
        actions.append(best_action)
        rewards.append(reward)
        done_list.append(done)

        planning_times.append(opt_time)

        obs = next_obs

        if debug:
            print(f"It {i}: Optimization time -- {opt_time:.8f}s")
            print(f"It {i}: Step time -- {step_time:.8f}s")

    # Append the last observation which belongs to the next_state
    observations.append(obs)

    if debug:
        avg_planning_time = np.mean(planning_times)
        print(f"Average planning time per step: {avg_planning_time:.3f}s")

    transition = Transition(
        observation=jnp.array(observations[:-1]),
        action=jnp.array(actions),
        reward=jnp.array(rewards),
        discount=jnp.ones((num_steps,)),
        next_observation=jnp.array(observations[1:]),
    )

    return transition, done_list


""" Optimization objective functions with constraints """


class BarrierType(Enum):
    RELU = "relu"
    HUBER = "huber"
    QUAD = "quadratic"
    EXP = "exponential"


@jax.jit
def huber(x):
    fn = jnp.where(jnp.abs(x) < 1, 0.5 * x**2, jnp.abs(x) - 0.5)
    return jnp.maximum(0, fn)


@jax.jit
def quadratic(x):
    fn = jnp.where(jnp.abs(x) < 1, x, x**2 - x + 1)
    return jnp.maximum(0, fn)


@jax.jit
def exponential(x):
    fn = jnp.exp(relu(x)) - 1
    return jnp.maximum(0, fn)


class BarrierAugmentedLagragian:
    def __init__(
        self,
        d: Union[float, jnp.ndarray] = 0.0,
        lmbda: float = 1.0,
        barrier_type: Union[str, BarrierType] = BarrierType.RELU,
    ):
        self.d = d
        self.lmbda = lmbda

        if isinstance(barrier_type, str):
            barrier_type = BarrierType(barrier_type)

        self.barrier_type = barrier_type
        self.cost_fn = self.init_cost_fn(barrier_type)

    @staticmethod
    def init_cost_fn(barrier_type: BarrierType):
        if barrier_type == BarrierType.RELU:
            return relu
        elif barrier_type == BarrierType.HUBER:
            return huber
        elif barrier_type == BarrierType.QUAD:
            return quadratic
        elif barrier_type == BarrierType.EXP:
            return exponential
        else:
            raise ValueError(f"Barrier type {barrier_type} not supported")

    def get_function(self):

        def barrier_augmented_lagragian(
            reward: jnp.ndarray,
            cost: jnp.ndarray,
        ):
            if isinstance(self.d, float) and cost.ndim > 1:
                self.d = self.d * jnp.ones_like(cost.shape[-1])

            if cost.ndim == 1:
                assert isinstance(self.d, float)

            avg_reward = reward.mean(axis=0)  # Mean over the trajectory rewards
            violation_cost = self.lmbda * self.cost_fn(
                cost.sum(axis=0) - self.d
            )  # Sum over the trajectory costs

            return violation_cost - avg_reward  # Minimization objective

        return barrier_augmented_lagragian

    def get_pes_function(self):

        def pes_augmented_lagragian(
            reward: jnp.ndarray,
            cost: jnp.ndarray,
        ):
            if isinstance(self.d, float) and cost.ndim > 1:
                self.d = self.d * jnp.ones_like(cost.shape[-1])

            if cost.ndim == 1:
                assert isinstance(self.d, float)

            avg_reward = reward.mean(axis=0)  # Mean over the trajectory rewards
            violation_cost = self.lmbda * self.cost_fn(
                jnp.max(cost, axis=0) - self.d
            )  # Take the trajectory with the highest cost

            return violation_cost - avg_reward  # Minimization objective

        return pes_augmented_lagragian


# Augmented Lagragian with RELU relaxation
def relu_augmented_lagragian(
    reward: jnp.ndarray,
    cost: jnp.ndarray,
    d: Union[float, jnp.ndarray] = 0.0,
    lmbda: float = 1.0,
):
    if isinstance(d, float) and cost.ndim > 1:
        d = d * jnp.ones_like(cost.shape[-1])

    if cost.ndim == 1:
        assert isinstance(d, float)

    avg_reward = reward.mean(axis=0)  # Mean over the trajectory rewards
    violation_cost = lmbda * relu(cost.sum(axis=0) - d)  # Sum over the trajectory costs

    return violation_cost - avg_reward  # Minimization objective


# Augmented Lagragian with proximal relaxation
def proximal_augmented_lagragian(reward, cost, lmbda=1):
    raise NotImplementedError


# Optimize the trajectory wrt to the worst performing particle in terms of cost (pessimistic planning)
def pes_trajectory_reward(
    reward: jnp.ndarray,
    cost: jnp.ndarray,
    d: Union[float, jnp.ndarray] = 0.0,
    lmbda: float = 1.0,
):
    if isinstance(d, float) and cost.ndim > 1:
        d = d * jnp.ones_like(cost.shape[-1])

    if cost.ndim == 1:
        assert isinstance(d, float)

    avg_reward = reward.mean(axis=0)  # Mean over the trajectory
    violation_cost = lmbda * relu(
        jnp.max(cost, axis=0) - d
    )  # Take the trajectory with the highest cost

    return violation_cost - avg_reward  # Minimization objective


""" Optimization objective functions without constraints """


def mean_reward(reward: jnp.ndarray):
    return -reward.mean(axis=0)


def sum_reward(reward: jnp.ndarray):
    return -reward.sum(axis=0)


def discounted_sum_reward(reward: jnp.ndarray, discount: float = 0.99):
    return -jnp.sum(reward * discount ** jnp.arange(reward.shape[0]), axis=0)
