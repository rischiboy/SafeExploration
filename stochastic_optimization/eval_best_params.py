import pickle
import time

import jax
import jax.numpy as jnp
from stochastic_optimization.agent.cem_agent import CEMAgent
from stochastic_optimization.agent.min_max_agent import MinMaxAgent
from stochastic_optimization.dynamical_system.car_park_system import (
    CarParkCost,
    CarParkReward,
    CarParkSystem,
    SafeCarParkSystem,
)
from stochastic_optimization.dynamical_system.dynamics_model import BNNDynamicsModel
from stochastic_optimization.dynamical_system.pendulum_system import (
    PendulumCost,
    PendulumReward,
    PendulumSystem,
    SafePendulumSystem,
)
from stochastic_optimization.environment.car_park_env import CarParkEnv

from stochastic_optimization.environment.pendulum_env import ConstrainedPendulum
from stochastic_optimization.optimizer.cem import CrossEntropyMethod
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
    relu_augmented_lagragian,
)

from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import (
    DeterministicEnsemble,
)

from stochastic_optimization.utils.type_utils import SamplingMode
from jax.config import config

# config.update("jax_log_compiles", 1)

###################
### Environment ###
###################

pendulum_env = ConstrainedPendulum(
    angle_tolerance=0.1,
    stability_duration=10,
    max_steps=200,
    max_speed=12.0,
    max_torque=2.0,
)

car_env = CarParkEnv(
    margins=(-0.15, 0.15),
    stability_duration=10,
    max_steps=200,
    max_action=8.0,
    max_speed=10.0,
)

env = car_env

########################
### Dynamical System ###
########################


def create_dynamical_system(env, dynamics, reward=None, cost=None):
    if isinstance(env, ConstrainedPendulum):
        if cost is None:
            dynamical_system = PendulumSystem(dynamics, reward)
        else:
            dynamical_system = SafePendulumSystem(dynamics, reward, cost)

    elif isinstance(env, CarParkEnv):
        if cost is None:
            dynamical_system = CarParkSystem(dynamics, reward)
        else:
            dynamical_system = SafeCarParkSystem(dynamics, reward, cost)

    else:
        raise ValueError("Invalid environment")

    return dynamical_system


#################
### CEM-Agent ###
#################

config = {
    "action_dim": (1,),
    "horizon": 20,
    "num_elites": 50,
    "num_iter": 10,
    "num_samples": 500,
    "lower_bound": -2,
    "upper_bound": 2,
}

####################
### Model Config ###
####################

# Probabilistic model
model_config = {
    "seed": 326176,
    "input_dim": 4,  # Change depededing on environment
    "output_dim": 3,  # Change depededing on environment
    "bnn_type": DeterministicEnsemble,
    "sampling_mode": SamplingMode.MEAN,
    "output_stds": jnp.array([0.0001, 0.0001, 0.0001]),
    "num_training_steps": 10,
    "beta": jnp.array([1.0, 1.0, 1.0]),
    "features": [256, 256],
    "lr_rate": 0.0001,
    "weight_decay": 0.001,
    "num_particles": 3,
    "train_share": 0.8,
    "batch_size": 64,
    "eval_frequency": 5,
    "eval_batch_size": 2048,
    "return_best_model": False,
}

dynamics = BNNDynamicsModel(**model_config)

#################
### CEM-Agent ###
#################

true_pendulum_system = PendulumSystem()
true_car_system = CarParkSystem()

true_safe_pendulum_system = SafePendulumSystem(cost=PendulumCost(max_speed=6.0))
true_safe_car_system = SafeCarParkSystem()

unsafe_car_system = create_dynamical_system(car_env, dynamics, CarParkReward())
unsafe_pendulum_system = create_dynamical_system(
    pendulum_env, dynamics, reward=PendulumReward()
)

safe_car_system = create_dynamical_system(
    car_env, dynamics, CarParkReward(), cost=CarParkCost()
)
safe_pendulum_system = create_dynamical_system(
    pendulum_env, dynamics, reward=PendulumReward(), cost=PendulumCost(max_speed=6.0)
)

cem_optimizer = CrossEntropyMethod(**config)


def create_cem_agent(env, dynamical_system, num_particles=1):
    cem_planner = CEMPlanner(
        dynamical_system=dynamical_system,
        optimizer=cem_optimizer,
        num_particles=num_particles,
    )

    agent = CEMAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        optimize_fn=mean_reward,
        policy_optimizer=cem_planner,
        dynamical_system=dynamical_system,
    )
    return agent


def create_safe_cem_agent(env, dynamical_system, num_particles=1):
    cem_planner = SafeCEMPlanner(
        safe_dynamical_system=dynamical_system,
        optimizer=cem_optimizer,
        num_particles=num_particles,
    )

    optimize_fn = lambda reward, cost: relu_augmented_lagragian(
        reward=reward, cost=cost, d=0.0, lmbda=100.0
    )

    safe_agent = CEMAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        optimize_fn=optimize_fn,
        policy_optimizer=cem_planner,
        dynamical_system=dynamical_system,
    )
    return safe_agent


####################
### MinMax-Agent ###
####################

action_config = {
    "action_dim": (20, 1),  # Regular action dimension
    # "horizon": horizon,
    "num_fixed_elites": 5,
    "num_elites": 50,
    "num_iter": 3,
    "num_samples": 500,
    "lower_bound": -2,
    "upper_bound": 2,
    "minimum": True,  # Maximize the reward
}

hal_action_config = {
    "action_dim": (20, 3),  # Hallucinated action dimension
    # "horizon": horizon,
    "num_fixed_elites": 5,
    "num_elites": 50,
    "num_iter": 3,
    "num_samples": 500,
    "lower_bound": -1,
    "upper_bound": 1,
    "minimum": False,  # Minimize the cost
}


def min_max_optimizer(action_config, hal_action_config, num_iter=3):
    action_config["num_iter"] = num_iter
    hal_action_config["num_iter"] = num_iter

    # Initialize the MinMax optimizer
    x_consts = OptVarConstants(**action_config)
    y_consts = OptVarConstants(**hal_action_config)
    var_x = OptVarParams(x_consts)
    var_y = OptVarParams(y_consts)

    optimizer = MinMaxOptimizer(var_x, var_y)

    return optimizer


def create_min_max_agent(
    env,
    dynamical_system,
    action_config,
    hal_action_config,
    num_particles=1,
    alpha=1.0,
    num_iter=3,
    iterations=10,
):
    optimizer = min_max_optimizer(action_config, hal_action_config, num_iter)

    planner = MinMaxPlanner(
        dynamical_system=dynamical_system,
        optimizer=optimizer,
        num_particles=num_particles,
        pes_alpha=alpha,
        iterations=iterations,
    )

    optimize_fn = lambda reward, cost: relu_augmented_lagragian(
        reward=reward, cost=cost, d=0.0, lmbda=100.0
    )

    agent = MinMaxAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        optimize_fn=optimize_fn,
        policy_optimizer=planner,
        dynamical_system=dynamical_system,
    )

    return agent


##########################
### Best Params Loader ###
##########################

pendulum_best_params = "./experiments/results/pendulum/CEM-Pendulum/20240311_17-46-35/782984_best_params.pkl"
car_best_params = (
    "./experiments/results/carpark/CEM-CarPark/20240312_19-18-07/510616_best_params.pkl"
)
pendulum_worst_params = "./experiments/results/pendulum/CEM-Pendulum/20240318_00-24-01/worst_model_params.pkl"  # MinMax
safe_pen_worst_params = "./experiments/results/pendulum/CEM-Pendulum/20240317_02-53-13/worst_model_params.pkl"  # SafeCEM


def load_params(path):
    with open(path, "rb") as f:
        params = pickle.load(f)

    params = params.statistical_model_state
    return params


#######################
### Plan an episode ###
#######################


def plan_episode(env, agent, best_params, rng, num_steps):
    # Reset the environment
    # obs, _ = env.reset()
    state = jnp.array([-0.9207, -3.1663])
    obs, _ = env.reset_any(state=state)
    done = False
    iter_count = 0
    num_violations = 0

    planning_time = 0

    for i in range(num_steps):
        rng, sample_key = jax.random.split(key=rng, num=2)
        start = time.time()
        action = agent.select_best_action(obs, best_params, sample_key)
        elapsed = time.time() - start
        print(f"Elapsed time: {elapsed:.3f} s")

        planning_time += elapsed

        next_obs, reward, done, truncate, info = env.step(action)
        print(f"Obs: {obs} | Action: {action} | Next: {next_obs} | Reward: {reward}")

        violated = agent.dynamical_system.constraint(next_obs)
        if violated:
            num_violations += 1

        obs = next_obs
        iter_count += 1

    print(f"Number of violations: {num_violations}")
    print(f"Planning time: {planning_time:.3f} s")
    print(f"Avg planning time: {planning_time / iter_count:.3f} s")

    return


############
### Main ###
############

if __name__ == "__main__":
    seed = 777
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)

    # Optimization parameters
    num_ensembles = 3
    num_particles = 5

    # MinMax
    num_iter = 5
    iterations = 10
    alpha = 1.0

    num_sim_steps = 20

    cem_agent = create_cem_agent(pendulum_env, unsafe_pendulum_system, num_particles)
    safe_cem_agent = create_safe_cem_agent(
        pendulum_env, safe_pendulum_system, num_particles
    )

    min_max_agent = create_min_max_agent(
        pendulum_env,
        safe_pendulum_system,
        action_config,
        hal_action_config,
        num_particles,
        alpha,
        num_iter,
        iterations,
    )

    # best_params, _, _ = min_max_agent.dynamical_system.init(init_key)
    worst_params = load_params(pendulum_worst_params)

    plan_episode(pendulum_env, min_max_agent, worst_params, rng, num_sim_steps)
