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

from config.pendulum.pendulum_dataclass import DefaultPendulumTrainer, PendulumTrainer
from stochastic_optimization.optimizer.safe_cem_planner import SafeCEMPlanner
from stochastic_optimization.optimizer.utils import (
    mean_reward,
    relu_augmented_lagragian,
)
from util import (
    Logger,
    flatten_dict,
    generate_sweeper_command,
    hash_dict,
    NumpyArrayEncoder,
)

from stochastic_optimization.dynamical_system.pendulum_system import (
    PendulumCost,
    PendulumSystem,
    SafePendulumSystem,
)
from stochastic_optimization.environment.pendulum_env import (
    ConstrainedPendulum,
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

""" Function ran by the sweep agent """


def run_experiment():
    global config
    global evaluated_params
    experiment_cfg = config.pendulum
    exp_wandb = experiment_cfg.wandb

    experiment_cfg_dict = OmegaConf.to_container(experiment_cfg)
    experiment_cfg_dict = flatten_dict(experiment_cfg_dict)
    experiment_cfg_dict.update({"exp_result_folder": config.exp_result_folder})

    for k, v in evaluated_params.items():
        experiment_cfg_dict[k] = v

    """Initialize wandb"""
    if exp_wandb.logging_wandb:
        group_name = exp_wandb.group_name
        if experiment_cfg_dict["dynamics_type"] is DynamicsType.TRUE:
            job_type = f"{experiment_cfg.train.seed}"
        else:
            sampling_mode = str(experiment_cfg.model.sampling_mode).split(".")[-1]
            job_type = f"{sampling_mode}"

        wandb.init(
            dir=exp_wandb.logs_dir,
            project=exp_wandb.project_name,
            group=group_name,
            job_type=job_type,
        )

        sweep_config = wandb.config

        experiment_cfg_dict = {
            k: sweep_config[k] if k in sweep_config else v
            for k, v in experiment_cfg_dict.items()
        }

    results = experiment(experiment_cfg_dict)
    return results


def experiment(config: dict):
    """Run experiment for a given method and environment."""
    cfg_dict = config
    config = SimpleNamespace(**cfg_dict)

    if config.logging_wandb:
        wandb.config.update(cfg_dict)  # Log config to wandb

    """ Environment """
    env = ConstrainedPendulum(
        angle_tolerance=config.angle_tolerance,
        stability_duration=config.stability_duration,
        max_steps=config.max_steps,
        max_speed=config.max_speed,
    )

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    """ Optimizer """
    cem_optimizer = CrossEntropyMethod(
        action_dim=action_dim,
        horizon=config.horizon,
        num_iter=config.num_iter,
        num_elites=config.num_elites,
        num_samples=config.num_samples,
        lower_bound=-2,
        upper_bound=2,
    )

    """ Model """
    dynamical_system = None

    if config.dynamics_type == DynamicsType.TRUE:
        if SAFE:
            cost = PendulumCost(max_speed=config.speed_threshold)
            dynamical_system = SafePendulumSystem(cost=cost)
        else:
            dynamical_system = PendulumSystem()

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
            "num_particles": config.num_particles,
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
            cost = PendulumCost(max_speed=config.speed_threshold)
            dynamical_system = SafePendulumSystem(dynamics=dynamics, cost=cost)
        else:
            dynamical_system = PendulumSystem(dynamics=dynamics)

    else:
        raise NotImplementedError

    """ Planner """
    assert (
        dynamical_system is not None
    ), "Could not initialize the Pitch Control dynamical system."

    if SAFE:
        cem_planner = SafeCEMPlanner(
            safe_dynamical_system=dynamical_system,
            optimizer=cem_optimizer,
        )
    else:
        cem_planner = CEMPlanner(
            dynamical_system=dynamical_system,
            optimizer=cem_optimizer,
        )

    """ Agent """

    if SAFE:
        optimize_fn = lambda reward, cost: relu_augmented_lagragian(
            reward=reward, cost=cost, d=config.d, lmbda=config.lmbda
        )
    else:
        optimize_fn = mean_reward

    agent = CEMAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        optimize_fn=optimize_fn,
        policy_optimizer=cem_planner,
        dynamical_system=dynamical_system,
    )

    """ Trainer """
    params_file = f"{config.seed}_params.pkl"
    best_params_file = f"{config.seed}_best_params.pkl"

    trainer = ModelBasedTrainer(
        env=env,
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


def create_log_dir(cfg: DefaultPendulumTrainer):
    from pprint import pprint

    """ generate experiment hash and set up redirect of output streams """

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


def eval_string_args(cfg: DefaultPendulumTrainer):
    """Evaluate string arguments to lists or custom types."""
    evaluated_params = {}

    evaluated_params["dynamics_type"] = eval(
        "DynamicsType." + cfg.pendulum.model.dynamics_type
    )
    evaluated_params["agent_type"] = eval("AgentType." + cfg.pendulum.model.agent_type)
    if evaluated_params["dynamics_type"] == DynamicsType.BNN:
        evaluated_params["sampling_mode"] = eval(
            "SamplingMode." + cfg.pendulum.model.sampling_mode
        )
        evaluated_params["output_stds"] = eval(cfg.pendulum.model.output_stds)
        evaluated_params["beta"] = eval(cfg.pendulum.model.beta)
        evaluated_params["features"] = eval(cfg.pendulum.model.features)

    return evaluated_params

    # cfg.pendulum.model.dynamics_type = eval(
    #     "DynamicsType." + cfg.pendulum.model.dynamics_type
    # )
    # cfg.pendulum.model.agent_type = eval("AgentType." + cfg.pendulum.model.agent_type)

    # if cfg.pendulum.model.dynamics_type == DynamicsType.TRUE:
    #     return cfg

    # else:
    #     cfg.pendulum.model.sampling_mode = eval(
    #         "SamplingMode." + cfg.pendulum.model.sampling_mode
    #     )
    #     cfg.pendulum.model.output_stds = eval(cfg.pendulum.model.output_stds)
    #     cfg.pendulum.model.beta = eval(cfg.pendulum.model.beta)
    #     cfg.pendulum.model.features = eval(cfg.pendulum.model.features)

    #     return cfg


def assert_proper_args(cfg: DefaultPendulumTrainer):
    """Assert that the arguments are valid."""

    assert cfg.pendulum is not None, "Please provide a pendulum configuration."
    assert (
        cfg.pendulum.model is not None
    ), "Please provide a model configuration for the pendulum."

    model_cfg = cfg.pendulum.model

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

        assert model_cfg.bnn_type is not None, "Please provide a sampling mode."
        assert issubclass(
            eval(model_cfg.bnn_type),
            BayesianNeuralNet,
        ), "Invalid bnn type"


cs = ConfigStore.instance()
cs.store(name="default", node=DefaultPendulumTrainer)


@hydra.main(
    config_path="config",
    config_name="default",
    version_base=None,
)
def main(cfg: DefaultPendulumTrainer):
    """Check whether the provided configuration is valid"""

    print(OmegaConf.to_yaml(cfg))

    # Validate the configuration for unexpected parameters
    def validate_fields(dataclass_type, config_data):
        allowed_params = set(dataclass_type.__annotations__.keys())
        for key, value in config_data.items():
            if key not in allowed_params:
                raise ValueError(f"Unexpected parameter '{key}' in the configuration.")
            if isinstance(value, dict):
                validate_fields(dataclass_type.__annotations__[key], value)

    validate_fields(DefaultPendulumTrainer, OmegaConf.to_container(cfg))

    """Create log directory"""
    # exp_hash = create_log_dir(cfg)

    """ Evaluate arguments """
    assert_proper_args(cfg)

    cfg_dict = OmegaConf.to_container(cfg)
    experiment_cfg = cfg.pendulum

    global SAFE
    if "constraint" in experiment_cfg:
        SAFE = True

    """ Create global variable for config to use in run_experiment """
    global config
    global evaluated_params
    config = cfg
    evaluated_params = eval_string_args(cfg)

    """Experiment core"""
    t_start = time.time()
    np.random.seed(experiment_cfg.train.seed)

    if cfg.sweep_id is not None:
        wandb.agent(cfg.sweep_id, function=run_experiment)
    else:
        eval_metrics = run_experiment()

    t_end = time.time()

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


if __name__ == "__main__":
    main()
