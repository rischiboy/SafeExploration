from pprint import pprint
from typing import List
import numpy as np
import jax.numpy as jnp
import time
import json
import os
import sys
import argparse
from omegaconf import DictConfig, OmegaConf
import wandb
import yaml
import hydra
from hydra.core.config_store import ConfigStore
from types import SimpleNamespace
from stochastic_optimization.agent.min_max_agent import MinMaxAgent
from stochastic_optimization.dynamical_system.car_park_system import (
    CarParkCost,
    CarParkSystem,
    SafeCarParkSystem,
)
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
from util import (
    Logger,
    flatten_dict,
    hash_dict,
    NumpyArrayEncoder,
    set_eval_env,
)

from stochastic_optimization.dynamical_system.car_park_system import (
    CarParkSystem,
)
from stochastic_optimization.environment.car_park_env import (
    CarParkEnv,
)
from stochastic_optimization.optimizer.cem import CrossEntropyMethod
from stochastic_optimization.agent.cem_agent import CEMAgent
from stochastic_optimization.model_based_trainer import ModelBasedTrainer
from stochastic_optimization.optimizer.cem_planner import CEMPlanner
from stochastic_optimization.dynamical_system.dynamics_model import BNNDynamicsModel
from stochastic_optimization.utils.type_utils import SamplingMode

from bsm.bayesian_regression.bayesian_neural_networks.bnn import BayesianNeuralNet
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import (
    DeterministicEnsemble,
)
from bsm.bayesian_regression.bayesian_neural_networks.probabilistic_ensembles import (
    ProbabilisticEnsemble,
)
from bsm.bayesian_regression.bayesian_neural_networks.fsvgd_ensemble import (
    DeterministicFSVGDEnsemble,
    ProbabilisticFSVGDEnsemble,
)

from type_utils import DynamicsType, OptimizerType

SAFE = False


def run_experiment():
    global config
    experiment_cfg = config.carpark
    exp_wandb = experiment_cfg.wandb

    experiment_cfg_dict = OmegaConf.to_container(experiment_cfg)
    experiment_cfg_dict = flatten_dict(experiment_cfg_dict)
    experiment_cfg_dict.update({"exp_result_folder": config.exp_result_folder})

    """Initialize wandb"""
    if exp_wandb.logging_wandb:
        group_name = exp_wandb.group_name
        job_type = exp_wandb.job_name

        wandb.init(
            dir=exp_wandb.logs_dir,
            project=exp_wandb.project_name,
            group=group_name,
            job_type=job_type,
        )

    results = experiment(experiment_cfg_dict)
    return results


def experiment(config: dict):
    """Run experiment for a given method and environment."""
    cfg_dict = config
    config = SimpleNamespace(**cfg_dict)

    if config.logging_wandb:
        wandb.config.update(cfg_dict)  # Log config to wandb

    """ Environment """
    env_args = {
        "margins": tuple(config.margins),
        "stability_duration": config.stability_duration,
        "max_steps": config.max_steps,
        "max_action": config.max_action,
        "max_speed": config.max_speed,
    }

    env = CarParkEnv(**env_args)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    """ Optimizer """
    if config.agent_type == OptimizerType.CEM:
        optimizer = CrossEntropyMethod(
            action_dim=action_dim,
            horizon=config.horizon,
            num_iter=config.num_iter,
            num_elites=config.num_elites,
            num_samples=config.num_samples,
            lower_bound=-1,
            upper_bound=1,
        )
    elif config.agent_type == OptimizerType.MinMax:
        action_config = {
            "action_dim": (config.horizon_x, *action_dim),  # Regular action dimension
            "num_fixed_elites": config.num_fixed_elites_x,
            "num_elites": config.num_elites_x,
            "num_iter": config.num_iter_x,
            "num_samples": config.num_samples_x,
            "lower_bound": -1,
            "upper_bound": 1,
            "minimum": True,  # Maximize the reward
        }
        hal_action_config = {
            "action_dim": (
                config.horizon_y,
                *state_dim,
            ),  # Hallucinated action dimension
            "num_fixed_elites": config.num_fixed_elites_y,
            "num_elites": config.num_elites_y,
            "num_iter": config.num_iter_y,
            "num_samples": config.num_samples_y,
            "lower_bound": -1,
            "upper_bound": 1,
            "minimum": False,  # Maximize the violation
        }
        # Initialize the MinMax optimizer
        x_consts = OptVarConstants(**action_config)
        y_consts = OptVarConstants(**hal_action_config)
        var_x = OptVarParams(x_consts)
        var_y = OptVarParams(y_consts)

        optimizer = MinMaxOptimizer(var_x, var_y)
    else:
        raise NotImplementedError

    """ Model """
    dynamical_system = None

    if config.dynamics_type == DynamicsType.TRUE:
        if SAFE:
            cost = CarParkCost(max_position=config.max_position)
            dynamical_system = SafeCarParkSystem(cost=cost)
        else:
            dynamical_system = CarParkSystem()

    elif config.dynamics_type == DynamicsType.BNN:
        input_dim = state_dim[0] + action_dim[0]
        output_dim = state_dim[0]

        config.output_stds = jnp.array(config.output_stds)
        config.beta = jnp.array(config.beta)
        config.bnn_type = eval(config.bnn_type)

        # Probabilistic model
        model_config = {
            # "seed": config.seed,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "bnn_type": config.bnn_type,
            # "sampling_mode": config.sampling_mode,
            "output_stds": config.output_stds,
            "num_training_steps": config.num_training_steps,
            "beta": config.beta,
            "features": config.features,
            "lr_rate": config.lr_rate,
            "weight_decay": config.weight_decay,
            "num_particles": config.num_ensembles,
            "train_share": config.train_share,
            "batch_size": config.batch_size,
            "eval_frequency": config.eval_frequency,
            "eval_batch_size": config.eval_batch_size,
            "logging_wandb": config.logging_wandb,
            "return_best_model": config.return_best_model,
        }

        dynamics = BNNDynamicsModel(
            seed=config.seed, sampling_mode=config.sampling_mode, **model_config
        )

        if SAFE:
            cost = CarParkCost(max_position=config.max_position)
            dynamical_system = SafeCarParkSystem(dynamics=dynamics, cost=cost)
        else:
            dynamical_system = CarParkSystem(dynamics=dynamics)

    else:
        raise NotImplementedError

    """ Planner """
    assert (
        dynamical_system is not None
    ), "Could not initialize the Carpark dynamical system."

    if config.dynamics_type == DynamicsType.TRUE:
        num_particles = 1
    else:
        num_particles = config.num_particles

    if isinstance(optimizer, CrossEntropyMethod):
        if SAFE:
            planner = SafeCEMPlanner(
                safe_dynamical_system=dynamical_system,
                optimizer=optimizer,
                num_particles=num_particles,
            )
        else:
            planner = CEMPlanner(
                dynamical_system=dynamical_system,
                optimizer=optimizer,
                num_particles=num_particles,
            )
    elif isinstance(optimizer, MinMaxOptimizer):
        assert SAFE, "MinMax only works with safe environments"
        planner = MinMaxPlanner(
            dynamical_system=dynamical_system,
            optimizer=optimizer,
            num_particles=num_particles,
            pes_alpha=config.alpha,
            iterations=config.iterations,
        )
    else:
        raise NotImplementedError

    """ Agent """

    if SAFE:
        optimize_fn = lambda reward, cost: relu_augmented_lagragian(
            reward=reward, cost=cost, d=config.d, lmbda=config.lmbda
        )
    else:
        optimize_fn = mean_reward

    if config.agent_type == OptimizerType.CEM:
        assert isinstance(
            planner, CEMPlanner
        ), f"Expected planner to be an instance of CEMPlanner, but got {type(planner).__name__}."

        agent = CEMAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            optimize_fn=optimize_fn,
            policy_optimizer=planner,
            dynamical_system=dynamical_system,
        )

    elif config.agent_type == OptimizerType.MinMax:
        assert isinstance(
            planner, MinMaxPlanner
        ), f"Expected planner to be an instance of MinMaxPlanner, but got {type(planner).__name__}."

        agent = MinMaxAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            optimize_fn=optimize_fn,
            policy_optimizer=planner,
            dynamical_system=dynamical_system,
        )

    else:
        raise NotImplementedError

    """ Trainer """
    params_file = f"{config.seed}_params.pkl"
    best_params_file = f"{config.seed}_best_params.pkl"

    # Evaluation environment
    if config.eval_episodes == 0:
        eval_env = None
    else:
        num_envs = config.eval_episodes
        video_dir = config.exp_result_folder + f"/video_{config.seed}"
        eval_env = set_eval_env(
            env=CarParkEnv,
            env_args=env_args,
            num_envs=num_envs,
            video_dir=video_dir,
            seed=config.seed,
            render=config.render,
        )

    trainer = ModelBasedTrainer(
        env=env,
        eval_env=eval_env,
        agent=agent,
        seed=config.seed,
        sample_batch_size=config.sample_batch_size,
        buffer_size=config.buffer_size,
        num_model_updates=config.num_model_updates,
        num_rollout_steps=config.num_rollout_steps,
        num_exploration_steps=config.num_exploration_steps,
        eval_episodes=config.eval_episodes,
        eval_frequency=config.eval_model_freq,
        val_buffer_size=config.val_buffer_size,
        val_batch_size=config.val_batch_size,
        diff_states=config.diff_states,
        logging_wandb=config.logging_wandb,
        out_dir=config.exp_result_folder,
        params_file=params_file,
        best_params_file=best_params_file,
        verbose=config.verbose,
        render=config.render,
    )

    model_params = trainer.train()

    result_dict = {}
    return result_dict


@hydra.main(
    config_path="config",
    # config_name="config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    """Create log directory"""
    # exp_hash = create_log_dir(cfg)    # Only needed if the experiment is run locally
    pprint(OmegaConf.to_yaml(cfg))

    """ Evaluate arguments """
    assert_proper_args(cfg)
    cfg = eval_string_args(cfg)

    cfg_dict = OmegaConf.to_container(cfg)
    experiment_cfg = cfg.carpark
    exp_wandb = experiment_cfg.wandb
    exp_result_folder = cfg.exp_result_folder

    global SAFE
    if "constraint" in experiment_cfg:
        SAFE = True

    """ Create global variable for config to use in run_experiment """
    global config
    config = cfg

    """Experiment core"""
    t_start = time.time()
    # np.random.seed(cfg.seed)
    np.random.seed(experiment_cfg.train.seed)

    eval_metrics = run_experiment()

    t_end = time.time()

    total_time_hours, total_time_minutes = divmod(t_end - t_start, 3600)
    total_time_minutes //= 60
    print(f"Total time: {total_time_hours:02d}:{total_time_minutes:02d}")

    """ Save experiment results and configuration """
    # results_dict = {
    #     "evals": eval_metrics,
    #     "params": cfg_dict,
    #     "duration_total": t_end - t_start,
    # }

    # if exp_result_folder is None:
    #     from pprint import pprint

    #     pprint(results_dict)
    # else:
    #     exp_result_file = os.path.join(exp_result_folder, "%s.json" % exp_hash)
    #     with open(exp_result_file, "w") as f:
    #         json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
    #     print("Dumped results to %s" % exp_result_file)


def create_log_dir(cfg):
    """generate experiment hash and set up redirect of output streams"""

    cfg_dict = OmegaConf.to_container(cfg)
    exp_hash = hash_dict(cfg_dict)
    if cfg.exp_result_folder is not None:
        os.makedirs(cfg.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(cfg.exp_result_folder, "%s.log " % exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    pprint(OmegaConf.to_yaml(cfg))
    print("\n ------------------------------------ \n")

    return exp_hash


def eval_string_args(cfg):
    """Evaluate string arguments to lists or custom types."""
    cfg.carpark.model.dynamics_type = eval(
        "DynamicsType." + cfg.carpark.model.dynamics_type
    )
    cfg.carpark.model.agent_type = eval("AgentType." + cfg.carpark.model.agent_type)

    if cfg.carpark.model.dynamics_type == DynamicsType.TRUE:
        return cfg

    else:
        cfg.carpark.model.sampling_mode = eval(
            "SamplingMode." + cfg.carpark.model.sampling_mode
        )
        # cfg.carpark.model.output_stds = eval(cfg.carpark.model.output_stds)
        # cfg.carpark.model.beta = eval(cfg.carpark.model.beta)
        # cfg.carpark.model.features = eval(cfg.carpark.model.features)

        return cfg


def assert_proper_args(cfg):
    """Assert that the arguments are valid."""

    assert cfg.carpark is not None, "Please provide a carpark configuration."
    assert (
        cfg.carpark.model is not None
    ), "Please provide a model configuration for the carpark."

    model_cfg = cfg.carpark.model

    assert isinstance(
        eval("DynamicsType." + model_cfg.dynamics_type), DynamicsType
    ), "Invalid dynamics type."
    assert isinstance(
        eval("AgentType." + model_cfg.agent_type), OptimizerType
    ), "Invalid agent type."

    if eval("DynamicsType." + model_cfg.dynamics_type) == DynamicsType.BNN:
        assert model_cfg.sampling_mode is not None, "Please provide a sampling mode."
        assert isinstance(
            eval("SamplingMode." + model_cfg.sampling_mode), SamplingMode
        ), "Invalid sampling mode."

        assert model_cfg.bnn_type is not None, "Please provide a BNN model type."
        assert issubclass(
            eval(model_cfg.bnn_type),
            BayesianNeuralNet,
        ), "Invalid bnn type"

    if eval("AgentType." + model_cfg.agent_type) == OptimizerType.MinMax:
        assert cfg.carpark.constraint is not None, "Please provide a constraint."
        assert cfg.carpark.minmax is not None, "Please provide a minmax configuration."

    elif eval("AgentType." + model_cfg.agent_type) == OptimizerType.CEM:
        assert cfg.carpark.cem is not None, "Please provide a cem configuration."
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
