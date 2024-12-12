import pickle
import jax
import jax.numpy as jnp
import numpy as np
import wandb
import yaml
import time
import os
import sys
from argparse import ArgumentParser
from tabulate import tabulate

from gym.envs.classic_control import PendulumEnv

from stochastic_optimization.dynamical_system.pitch_control_system import (
    PitchControlSystem,
)
from stochastic_optimization.environment.pitch_control_env import PitchControlEnv
from stochastic_optimization.optimizer.utils import mean_reward

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from bsm.bayesian_regression.bayesian_neural_networks.fsvgd_ensemble import (
    ProbabilisticFSVGDEnsemble,
)
from stochastic_optimization.optimizer.cem_planner import CEMPlanner, plan
from stochastic_optimization.dynamical_system.pendulum_system import PendulumSystem
from stochastic_optimization.environment.pendulum_env import (
    CustomPendulum,
    ConstrainedPendulum,
)
from stochastic_optimization.dynamical_system.dynamics_model import BNNDynamicsModel
from stochastic_optimization.optimizer.cem import CrossEntropyMethod
from stochastic_optimization.agent.cem_agent import CEMAgent
from stochastic_optimization.model_based_trainer import ModelBasedTrainer
from stochastic_optimization.utils import rollout_trajectory

# from jax.config import config

# config.update("jax_log_compiles", 1)


def store_params(params, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(params, f)

    return


def load_params(file_name):
    with open(file_name, "rb") as f:
        params = pickle.load(f)

    return params


def print_config_params(**kwargs):
    for key, value in kwargs.items():
        if isinstance(value, dict):
            print(key.upper())
            tabulated_value = tabulate([[k, v] for k, v in value.items()])
            print(tabulated_value)

        else:
            tabulated_value = tabulate([[key.upper(), value]])
            print(tabulated_value)

    return


def list_to_jax_array(attr):
    if isinstance(attr, list):
        attr = jnp.array(attr)

    return attr


def get_config_params(config_file, print_config=True):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if print_config:
        print_config_params(**config)

    wandb_config = config["wandb"]
    cem_config = config["cem"]

    if "model" not in config:
        model_config = None
    else:
        model_config = config["model"]
        model_config["output_stds"] = list_to_jax_array(model_config["output_stds"])
        model_config["beta"] = list_to_jax_array(model_config["beta"])
        model_config["logging_wandb"] = wandb_config["logging_wandb"]

    train_config = config["train"]

    return wandb_config, cem_config, model_config, train_config


def create_agent(cem_config, model_config, env):
    # Environment parameters
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    cem_config = {
        "action_dim": action_dim,
        "lower_bound": -2,
        "upper_bound": 2,
        **cem_config,
    }
    cem_optimizer = CrossEntropyMethod(**cem_config)

    input_dim = state_dim[0] + action_dim[0]
    output_dim = state_dim[0]

    # Choose between a deterministic and a probabilistic model
    kwargs = {}
    if model_config is not None:
        model_config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "bnn_type": ProbabilisticFSVGDEnsemble,
            **model_config,
        }
        dynamics = BNNDynamicsModel(**model_config)
        kwargs["dynamics"] = dynamics

    if isinstance(env, PendulumEnv):
        dynamical_system = PendulumSystem(**kwargs)
    elif isinstance(env, PitchControlEnv):
        dynamical_system = PitchControlSystem(**kwargs)

    cem_planner = CEMPlanner(dynamical_system, cem_optimizer)

    agent = CEMAgent(
        env.action_space,
        env.observation_space,
        cem_planner,
        dynamical_system,
    )

    return agent


def train_model_based_agent(env, config_file, eval_key=None):
    wandb_config, cem_config, model_config, train_config = get_config_params(
        config_file, print_config=True
    )

    agent = create_agent(cem_config, model_config, env)

    logging_wandb = wandb_config["logging_wandb"]
    train_config = {
        "logging_wandb": logging_wandb,
        "out_dir": log_dir,
        **train_config,
    }

    # Setup wandb logging
    if logging_wandb:
        project_name = wandb_config["project_name"]
        run_name = wandb_config["run_name"]
        config_dict = {"CEM": cem_config, "Agent": model_config, "Train": train_config}
        wandb.init(project=project_name, name=run_name, config=config_dict)

    # Train the agent
    print("Running Pendulum Trainer...")
    trainer = ModelBasedTrainer(env, agent, **train_config)
    model_params = trainer.train()
    print("Training finished!")

    # Save the latest model parameters
    file_name = trainer.params_file
    store_params(model_params, file_name)

    # Evaluate the trained model by performing simulation rollouts
    if eval_key is None:
        eval_key = jax.random.PRNGKey(seed=0)

    evaluate_trained_model(
        env=env,
        config_file=config_file,
        params_file=file_name,
        opt_key=eval_key,
        agent=agent,
        num_steps=200,
    )

    if logging_wandb:
        wandb.finish()

    return


def evaluate_trained_model(
    env, config_file, params_file, opt_key=None, agent=None, num_steps=200
):
    assert agent is not None or config_file is not None

    # Default config file is used if no config file is provided in the arguments
    if agent is None:
        _, cem_config, model_config, done = get_config_params(
            config_file, print_config=False
        )
        agent = create_agent(cem_config, model_config, env)

    model_params = load_params(params_file)

    if opt_key is None:
        opt_key = jax.random.PRNGKey(seed=0)

    obs, _ = env.reset()

    print("Evaluation started!")

    transitions, done = plan(
        env=env,
        planner=agent.policy_optimizer,
        optimize_fn=mean_reward,
        init_obs=obs,
        model_params=model_params,
        rng=opt_key,
        num_steps=num_steps,
    )
    rewards = transitions.reward
    last_obs = transitions.next_observation[-1]

    eval_metrics = [
        ["Average reward", jnp.sum(rewards)],
        ["Last state", last_obs],
        ["Planning successful", done[-1]],
    ]

    # print("Average reward: ", jnp.mean(planned_rewards))
    # print("Last state: ", planned_states[-1])
    # print("Planning successful: ", done[-1])

    if done[-1]:
        false_ind = np.where(~np.array(done))[0]
        last_false_ind = false_ind[-1]
        stability_period = num_steps - last_false_ind
        # print("Pendulum upright for last: ", stability_period, " steps")
        eval_metrics.append(["Stability period", stability_period])

    tabulated_metrics = tabulate(eval_metrics)
    print(tabulated_metrics)

    print("Evaluation finished!")

    return


def test_env_rollout(env, agent, num_steps=200):
    key = jax.random.PRNGKey(seed=0)
    key, init_key = jax.random.split(key)
    obs, _ = env.reset()
    model_params, reward_params = agent.dynamical_system.init(key=init_key)

    key, rollout_key = jax.random.split(key)
    start = time.time()
    transitions = rollout_trajectory(
        env=env,
        policy=agent.select_best_action,
        init_state=obs,
        model_params=model_params,
        optimizer_rng=rollout_key,
        num_steps=num_steps,
    )
    end = time.time()
    print("Time taken: ", end - start)
    print(transitions)

    return transitions


if __name__ == "__main__":
    default_config = "run/config/bnn_dynamics_pendulum.yaml"
    # default_config = "run/config/bnn_dynamics_pitch_control.yaml"
    default_log_dir = "run/output/"

    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default=default_config, required=False
    )
    parser.add_argument("--log_dir", type=str, default=default_log_dir, required=False)

    args = parser.parse_args()

    config_file = args.config_file
    log_dir = args.log_dir

    env = ConstrainedPendulum(max_steps=50)
    # env = PitchControlEnv(max_steps=10)

    eval_key = jax.random.PRNGKey(seed=14)
    train_model_based_agent(env, config_file, eval_key=eval_key)

    # env = CustomPendulum(render_mode="human")
    # params_file = f"{log_dir}/model_params.pkl"
    # evaluate_trained_model(env, config_file, params_file, None, num_steps=200)
