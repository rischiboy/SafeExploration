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
from stochastic_optimization.agent.opt_min_max_agent import OptMinMaxAgent
from stochastic_optimization.dynamical_system.gp_dynamical_system import GPDynamicsModel
from stochastic_optimization.optimizer.icem import ICEM
from stochastic_optimization.optimizer.min_max import (
    MinMaxOptimizer,
    OptVarConstants,
    OptVarParams,
)
from stochastic_optimization.optimizer.min_max_planner import MinMaxPlanner
from stochastic_optimization.optimizer.opt_min_max_planner import OptMinMaxPlanner
from stochastic_optimization.optimizer.pessimistic_trajectory_planner import (
    PesTrajectoryPlanner,
)
from stochastic_optimization.optimizer.safe_cem_planner import SafeCEMPlanner
from stochastic_optimization.optimizer.utils import (
    mean_reward,
    pes_trajectory_reward,
    relu_augmented_lagragian,
    BarrierAugmentedLagragian,
    BarrierType,
)
from stochastic_optimization.utils.scheduler import LinearScheduler, SigmoidScheduler
from util import (
    Logger,
    flatten_dict,
    hash_dict,
    set_eval_env,
)

from stochastic_optimization.dynamical_system.pendulum_system import (
    PendulumCost,
    PendulumReward,
    PendulumStateCost,
    PendulumStateDynamics,
    PendulumStateSystem,
    PendulumSystem,
    PendulumTrueDynamics,
    SafePendulumStateSystem,
    SafePendulumSystem,
)
from stochastic_optimization.environment.pendulum_env import (
    ConstrainedPendulum,
    ConstrainedPendulumState,
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

from type_utils import AgentType, DynamicsType, OptimizerType

SAFE = False


def run_experiment():
    global config
    experiment_cfg = config.pendulum
    exp_wandb = experiment_cfg.wandb

    experiment_cfg_dict = OmegaConf.to_container(experiment_cfg)
    experiment_cfg_dict = flatten_dict(experiment_cfg_dict)
    experiment_cfg_dict.update({"exp_result_folder": config.exp_result_folder})

    """Initialize wandb"""
    if exp_wandb.logging_wandb:
        group_name = exp_wandb.group_name
        job_type = exp_wandb.job_name

        # if experiment_cfg.model.dynamics is DynamicsType.TRUE:
        #     job_type = f"{experiment_cfg.train.seed}"
        # else:
        #     sampling_mode = str(experiment_cfg.model.sampling_mode).split(".")[-1]
        #     job_type = f"{sampling_mode}"

        wandb.init(
            dir=exp_wandb.logs_dir,
            project=exp_wandb.project_name,
            group=group_name,
            job_type=job_type,
        )

        # sweep_config = wandb.config

        # experiment_cfg_dict = {
        #     k: sweep_config[k] if k in sweep_config else v
        #     for k, v in experiment_cfg_dict.items()
        # }

    results = experiment(experiment_cfg_dict)
    wandb.finish()

    return results


def experiment(config: dict):
    """Run experiment for a given method and environment."""
    cfg_dict = config
    config = SimpleNamespace(**cfg_dict)

    if config.logging_wandb:
        wandb.config.update(cfg_dict)  # Log config to wandb

    """ Environment """
    env_args = {
        "action_cost": config.action_cost,
        "angle_tolerance": config.angle_tolerance,
        "stability_duration": config.stability_duration,
        "max_steps": config.max_steps,
        "max_speed": config.max_speed,
    }

    env = ConstrainedPendulum(**env_args)
    # env = ConstrainedPendulumState(**env_args)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    """ Optimizer """
    if config.optimizer == OptimizerType.CEM:
        optimizer = CrossEntropyMethod(
            action_dim=action_dim,
            horizon=config.horizon,
            num_iter=config.num_iter,
            num_elites=config.num_elites,
            num_samples=config.num_samples,
            lower_bound=-1,
            upper_bound=1,
        )
    elif config.optimizer == OptimizerType.ICEM:
        optimizer = ICEM(
            action_dim=action_dim,
            horizon=config.horizon,
            num_iter=config.num_iter,
            num_elites=config.num_elites,
            num_samples=config.num_samples,
            exponent=config.exponent,
            lower_bound=-1,
            upper_bound=1,
        )
    elif config.optimizer == OptimizerType.MinMax:

        action_exponent = {}
        hal_action_exponent = {}
        if hasattr(config, "exponent_x"):
            action_exponent = {"exponent": config.exponent_x}
        if hasattr(config, "exponent_y"):
            hal_action_exponent = {"exponent": config.exponent_y}

        if config.agent == AgentType.OptMinMax:
            ext_action_dim = state_dim[0] + action_dim[0]
            opt_action_dim = (config.horizon_x, ext_action_dim)
        else:
            opt_action_dim = (config.horizon_x, *action_dim)

        action_config = {
            "action_dim": opt_action_dim,  # Regular action dimension
            "num_fixed_elites": config.num_fixed_elites_x,
            "num_elites": config.num_elites_x,
            "num_iter": config.num_iter_x,
            "num_samples": config.num_samples_x,
            **action_exponent,
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
            **hal_action_exponent,
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

    # Common hyperparameters
    input_dim = state_dim[0] + action_dim[0]
    output_dim = state_dim[0]

    # Dynamics
    if config.dynamics == DynamicsType.TRUE:

        if isinstance(env, ConstrainedPendulumState):
            dynamics = PendulumStateDynamics()
        else:
            dynamics = PendulumTrueDynamics()

    elif config.dynamics == DynamicsType.BNN:
        config.bnn_type = eval(config.bnn_type)
        config.output_stds = jnp.array(config.output_stds)
        config.beta = jnp.array(config.beta)

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
            "num_calibration_ps": config.num_calibration_ps,
            "num_test_alphas": config.num_test_alphas,
            "num_particles": config.num_ensembles,
            "train_share": config.train_share,
            "batch_size": config.batch_size,
            "eval_frequency": config.eval_frequency,
            "eval_batch_size": config.eval_batch_size,
            "logging_wandb": config.logging_wandb,
            "return_best_model": config.return_best_model,
        }

        dynamics = BNNDynamicsModel(
            seed=config.seed,
            sampling_mode=config.sampling_mode,
            diff_states=config.diff_states,
            **model_config,
        )

    elif config.dynamics == DynamicsType.GP:
        config.output_stds = jnp.array(config.output_stds)
        config.beta = jnp.array(config.beta)

        model_config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            # "f_norm_bound": config.f_norm_bound,
            # "delta": config.delta,
            "num_training_steps": config.num_training_steps,
            "beta": config.beta,
            "output_stds": config.output_stds,
            "weight_decay": config.weight_decay,
            "lr_rate": config.lr_rate,
            "logging_wandb": config.logging_wandb,
        }

        dynamics = GPDynamicsModel(
            seed=config.seed,
            sampling_mode=config.sampling_mode,
            diff_states=config.diff_states,
            **model_config,
        )
    else:
        raise NotImplementedError

    reward = PendulumReward(control_cost=config.action_cost)

    # System
    if SAFE:
        if isinstance(env, ConstrainedPendulumState):
            cost = PendulumStateCost(max_speed=config.speed_threshold)
            dynamical_system = SafePendulumStateSystem(
                env=env, dynamics=dynamics, reward=reward, cost=cost
            )
        else:
            cost = PendulumCost(max_speed=config.speed_threshold)
            # dynamical_system = SafePendulumSystem(env=env, dynamics=dynamics, cost=cost)
            dynamical_system = SafePendulumSystem(
                dynamics=dynamics, reward=reward, cost=cost
            )
    else:
        if isinstance(env, ConstrainedPendulumState):
            dynamical_system = PendulumStateSystem(
                env=env, dynamics=dynamics, reward=reward
            )
        else:
            # dynamical_system = PendulumSystem(env=env, dynamics=dynamics)
            dynamical_system = PendulumSystem(dynamics=dynamics, reward=reward)

    # To produce validation data
    env.set_dynamical_sytem(PendulumSystem(reward=reward))

    """ Planner """
    assert (
        dynamical_system is not None
    ), "Could not initialize the Pendulum dynamical system."

    if config.dynamics == DynamicsType.TRUE:
        num_particles = 1
    else:
        num_particles = config.num_particles

    if config.agent == AgentType.CEM:
        planner = CEMPlanner(
            dynamical_system=dynamical_system,
            optimizer=optimizer,
            num_particles=num_particles,
        )
    elif config.agent == AgentType.SafeCEM:
        # assert SAFE, "SafeCEM only works with safe environments"
        planner = SafeCEMPlanner(
            safe_dynamical_system=dynamical_system,
            optimizer=optimizer,
            num_particles=num_particles,
        )
    elif config.agent == AgentType.PesTraj:
        # assert SAFE, "PesTraj only works with safe environments"
        planner = PesTrajectoryPlanner(
            safe_dynamical_system=dynamical_system,
            optimizer=optimizer,
            num_particles=num_particles,
        )

    elif config.agent == AgentType.MinMax:
        # assert SAFE, "MinMax only works with safe environments"
        planner = MinMaxPlanner(
            dynamical_system=dynamical_system,
            optimizer=optimizer,
            num_particles=num_particles,
            pes_alpha=config.pes_alpha,
            iterations=config.iterations,
        )
    elif config.agent == AgentType.OptMinMax:
        # assert SAFE, "MinMax only works with safe environments"
        planner = OptMinMaxPlanner(
            dynamical_system=dynamical_system,
            optimizer=optimizer,
            num_particles=num_particles,
            pes_alpha=config.pes_alpha,
            opt_alpha=config.opt_alpha,
            iterations=config.iterations,
        )
    else:
        raise NotImplementedError

    """ Agent """
    #######################################
    ### Optimization Objective Function ###
    #######################################

    if SAFE:
        assert config.barrierfn is not None, "Please provide a barrier function."
        function_cls = BarrierAugmentedLagragian(
            d=config.d, lmbda=config.lmbda, barrier_type=config.barrierfn
        )

        if config.agent == AgentType.PesTraj:
            optimize_fn = function_cls.get_pes_function()
        else:
            optimize_fn = function_cls.get_function()
    else:
        optimize_fn = mean_reward

    if (
        config.agent == AgentType.CEM
        or config.agent == AgentType.SafeCEM
        or config.agent == AgentType.PesTraj
    ):
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
    elif config.agent == AgentType.MinMax:
        assert isinstance(
            planner, MinMaxPlanner
        ), f"Expected planner to be an instance of MinMaxPlanner, but got {type(planner).__name__}."

        start_value = config.pes_alpha
        end_value = config.pes_alpha + 2

        scale = 0.25
        midpoint = 10

        # pes_alpha_scheduler = LinearScheduler(
        #     start_value=start_value,
        #     end_value=end_value,
        #     n_steps=config.num_model_updates,
        # )
        pes_alpha_scheduler = SigmoidScheduler(
            start_value=start_value,
            end_value=end_value,
            scale=scale,
            midpoint=midpoint,
            n_steps=config.num_model_updates,
        )

        agent = MinMaxAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            optimize_fn=optimize_fn,
            policy_optimizer=planner,
            dynamical_system=dynamical_system,
            pes_alpha_scheduler=pes_alpha_scheduler,
        )
    elif config.agent == AgentType.OptMinMax:
        assert isinstance(
            planner, OptMinMaxPlanner
        ), f"Expected planner to be an instance of OptMinMaxPlanner, but got {type(planner).__name__}."

        start_value = config.pes_alpha
        end_value = config.pes_alpha + 2

        scale = 0.25
        midpoint = 10

        # pes_alpha_scheduler = LinearScheduler(
        #     start_value=start_value,
        #     end_value=end_value,
        #     n_steps=config.num_model_updates,
        # )
        pes_alpha_scheduler = SigmoidScheduler(
            start_value=start_value,
            end_value=end_value,
            scale=scale,
            midpoint=midpoint,
            n_steps=config.num_model_updates,
        )

        start_value = config.opt_alpha
        end_value = config.opt_alpha + 2

        scale = 0.25
        midpoint = 10

        # opt_alpha_scheduler = LinearScheduler(
        #     start_value=start_value,
        #     end_value=end_value,
        #     n_steps=config.num_model_updates,
        # )
        opt_alpha_scheduler = SigmoidScheduler(
            start_value=start_value,
            end_value=end_value,
            scale=scale,
            midpoint=midpoint,
            n_steps=config.num_model_updates,
        )

        agent = OptMinMaxAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            optimize_fn=optimize_fn,
            policy_optimizer=planner,
            dynamical_system=dynamical_system,
            pes_alpha_scheduler=pes_alpha_scheduler,
            opt_alpha_scheduler=opt_alpha_scheduler,
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
        if type(env) == ConstrainedPendulumState:
            eval_env = set_eval_env(
                env=ConstrainedPendulumState,
                env_args=env_args,
                num_envs=num_envs,
                video_dir=video_dir,
                seed=config.seed,
                render=config.render,
            )
        else:
            eval_env = set_eval_env(
                env=ConstrainedPendulum,
                env_args=env_args,
                num_envs=num_envs,
                video_dir=video_dir,
                seed=config.seed,
                render=config.render,
            )

    if hasattr(config, "calibrate"):
        calibrate = config.calibrate
    else:
        calibrate = False

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
        plot_freq=config.plot_freq,
        diff_states=config.diff_states,
        logging_wandb=config.logging_wandb,
        out_dir=config.exp_result_folder,
        params_file=params_file,
        best_params_file=best_params_file,
        verbose=config.verbose,
        render=config.render,
        calibrate=calibrate,
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
    # exp_hash = create_log_dir(cfg)
    pprint(OmegaConf.to_yaml(cfg))

    """ Evaluate arguments """
    assert_proper_args(cfg)
    cfg = eval_string_args(cfg)

    cfg_dict = OmegaConf.to_container(cfg)
    experiment_cfg = cfg.pendulum
    # exp_wandb = experiment_cfg.wandb
    # exp_result_folder = cfg.exp_result_folder

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
    print(f"Total time: {total_time_hours:02.0f}:{total_time_minutes:02.0f}")

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
    cfg.pendulum.model.dynamics = eval("DynamicsType." + cfg.pendulum.model.dynamics)
    cfg.pendulum.model.optimizer = eval("OptimizerType." + cfg.pendulum.model.optimizer)
    cfg.pendulum.model.agent = eval("AgentType." + cfg.pendulum.model.agent)

    if cfg.pendulum.model.dynamics == DynamicsType.TRUE:
        return cfg

    else:
        cfg.pendulum.model.sampling_mode = eval(
            "SamplingMode." + cfg.pendulum.model.sampling_mode
        )
        # cfg.pendulum.model.output_stds = eval(cfg.pendulum.model.output_stds)
        # cfg.pendulum.model.beta = eval(cfg.pendulum.model.beta)
        # cfg.pendulum.model.features = eval(cfg.pendulum.model.features)

        return cfg


def assert_proper_args(cfg):
    """Assert that the arguments are valid."""

    assert cfg.pendulum is not None, "Please provide a pendulum configuration."
    assert (
        cfg.pendulum.model is not None
    ), "Please provide a model configuration for the pendulum."

    model_cfg = cfg.pendulum.model

    assert isinstance(
        eval("DynamicsType." + model_cfg.dynamics), DynamicsType
    ), "Invalid dynamics type."
    assert isinstance(
        eval("OptimizerType." + model_cfg.optimizer), OptimizerType
    ), "Invalid optimizer type."
    assert isinstance(
        eval("AgentType." + model_cfg.agent), AgentType
    ), "Invalid agent type."

    dynamics = eval("DynamicsType." + model_cfg.dynamics)
    optimizer = eval("OptimizerType." + model_cfg.optimizer)
    agent = eval("AgentType." + model_cfg.agent)

    if dynamics == DynamicsType.BNN:
        assert model_cfg.sampling_mode is not None, "Please provide a sampling mode."
        assert isinstance(
            eval("SamplingMode." + model_cfg.sampling_mode), SamplingMode
        ), "Invalid sampling mode."

        assert model_cfg.bnn_type is not None, "Please provide a BNN model type."
        assert issubclass(
            eval(model_cfg.bnn_type),
            BayesianNeuralNet,
        ), "Invalid bnn type"

    if optimizer == OptimizerType.MinMax:
        assert cfg.pendulum.minmax is not None, "Please provide a minmax configuration."

    elif optimizer == OptimizerType.CEM:
        assert cfg.pendulum.cem is not None, "Please provide a cem configuration."
    elif optimizer == OptimizerType.ICEM:
        assert cfg.pendulum.icem is not None, "Please provide a icem configuration."
        assert cfg.pendulum.icem.exponent is not None, "Please provide an exponent."
    else:
        raise NotImplementedError

    if agent == AgentType.CEM:
        assert (
            optimizer == OptimizerType.CEM or optimizer == OptimizerType.ICEM
        ), "Agent and optimizer must match."
    elif agent == AgentType.SafeCEM or agent == AgentType.PesTraj:
        assert (
            optimizer == OptimizerType.CEM or optimizer == OptimizerType.ICEM
        ), "Agent and optimizer must match."
        assert cfg.pendulum.constraint is not None, "Please provide a constraint."
    elif agent == AgentType.MinMax or agent == AgentType.OptMinMax:
        assert optimizer == OptimizerType.MinMax, "Agent and optimizer must match."
        assert cfg.pendulum.constraint is not None, "Please provide a constraint."
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
