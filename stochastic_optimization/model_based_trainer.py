import glob
import os
import pickle
import gym
from jax import vmap
from jax.scipy.stats import norm
import pandas as pd
from typing import Dict, List, Optional, Union
import jax
import jax.numpy as jnp
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt

from gym.wrappers.record_video import RecordVideo
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
import wandb
from stochastic_optimization.agent.abstract_agent import AbstractAgent
from stochastic_optimization.agent.min_max_agent import MinMaxAgent
from stochastic_optimization.agent.opt_min_max_agent import OptMinMaxAgent
from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    SafeDynamicalSystem,
)

from bsm.utils.type_aliases import (
    ModelState,
    StatisticalModelOutput,
    StatisticalModelState,
)
from bsm.bayesian_regression.bayesian_neural_networks.bnn import BNNState

from stochastic_optimization.environment.pendulum_env import ConstrainedPendulum
from stochastic_optimization.environment.pitch_control_env import PitchControlEnv

from stochastic_optimization.utils.trainer_utils import (
    generate_random_transitions,
    get_dummy_transition,
    get_total_std,
    rollout_trajectory,
    format_floats,
    generate_train_data,
    uniform_sampling,
)

from stochastic_optimization.utils.wandb_utils import (
    create_violation_plots,
    get_plot_dir,
    log_calibration_alpha,
    log_eval_video,
)

from copy import deepcopy

from mbse.utils.vec_env import VecEnv


class ModelBasedTrainer:
    def __init__(
        self,
        env: Union[gym.Env, VecEnv],
        eval_env: Optional[Union[gym.Env, VecEnv]],
        agent: AbstractAgent,
        seed: int = 0,
        sample_batch_size: int = 64,
        buffer_size: int = 1000,
        num_model_updates: int = 10,
        num_rollout_steps: int = 200,
        num_exploration_steps: int = 200,
        eval_episodes: int = 1,
        eval_frequency: int = 5,
        val_buffer_size: int = 0,
        val_batch_size: int = 0,
        plot_freq: int = 10,
        diff_states: bool = False,
        logging_wandb: bool = False,
        out_dir: str = "run/output/",
        params_file: str = "model_params.pkl",
        best_params_file: str = "best_model_params.pkl",
        verbose: bool = False,
        render: bool = False,
        calibrate: bool = True,
    ):
        self.agent = agent

        self.input_dim = self.agent.transition_model.input_dim
        self.output_dim = self.agent.transition_model.output_dim

        self.seed = seed
        self.sample_batch_size = sample_batch_size
        self.buffer_size = buffer_size
        self.num_model_updates = num_model_updates
        self.num_rollout_steps = num_rollout_steps
        self.num_exploration_steps = num_exploration_steps
        self.eval_episodes = eval_episodes
        self.eval_frequency = eval_frequency
        self.val_buffer_size = val_buffer_size
        self.val_batch_size = val_batch_size
        self.plot_freq = plot_freq
        self.diff_states = diff_states
        self.logging_wandb = logging_wandb
        self.out_dir = out_dir
        self.params_file = out_dir + params_file
        self.best_params_file = out_dir + best_params_file
        self.verbose = verbose
        self.calibrate = calibrate

        self.worst_params_file = out_dir + f"worst_model_params_{seed}.pkl"
        self.max_violations = 0

        self.input_dim = self.agent.transition_model.input_dim
        self.output_dim = self.agent.transition_model.output_dim

        self.render = render
        self.env = env
        self.eval_env = eval_env

        assert (
            buffer_size >= sample_batch_size
        ), "Train buffer size must be greater than or equal to the train batch size"
        assert (
            val_buffer_size >= val_batch_size
        ), "Validation buffer size must be greater than or equal to the validation batch size"

        # Set the random seed for the environment
        _, _ = self.env.reset(seed=seed)

        # Set the evaluation environment
        self.evaluation_enabled = eval_frequency > 0
        self.setup_evaluation()

        # Set the constraint
        # self.is_safe = isinstance(self.agent.dynamical_system, SafeDynamicalSystem)
        self.set_constraint()

        self.is_minmax = isinstance(self.agent, MinMaxAgent)
        self.is_optminmax = isinstance(self.agent, OptMinMaxAgent)

        self.validation_enabled = self.val_buffer_size > 0
        self.init_validation_data()

        # Metris to log
        if self.logging_wandb:
            self.set_wandb_metrics()

    #################################
    ### Initialization functions ####
    #################################

    def init_validation_data(self):
        if self.validation_enabled:
            self.val_data = generate_random_transitions(self.env, self.val_buffer_size)
            return

    def set_wandb_metrics(self):
        self.defined_metrics = []
        self.step_metric = "Episode"

        training_metrics = [
            "Train Reward",
            "Train Violations",
            "Max Violation",
        ]
        validation_metrics = [
            "Validation MSE",
            "Mean epistemic uncertainty",
            "Std epistemic uncertainty",
            "Mean aleatoric uncertainty",
        ]
        eval_metrics = [
            "Reward",
            "Planning MSE",
            "Average violations",
            "Average predicted violations",
            "Average NLL",
            "Iterations",
            "Simulation",  # Video
        ]
        metrics = training_metrics + validation_metrics + eval_metrics

        wandb.define_metric(self.step_metric)
        # define which metrics will be plotted against it
        for metric in metrics:
            if metric in self.defined_metrics:
                continue
            else:
                wandb.define_metric(metric, step_metric=self.step_metric)
                self.defined_metrics.append(metric)
        return

    def set_wandb_custom_axis(self, metrics: str, step_metric: str):
        for metric in metrics:
            if metric in self.defined_metrics:
                continue
            else:
                wandb.define_metric(metric, step_metric=step_metric)
                self.defined_metrics.append(metric)
        return

    def set_constraint(self):
        self.constraint = self.agent.dynamical_system.constraint
        self.constraint_deviation = self.agent.dynamical_system.constraint_deviation

        return

    def setup_evaluation(self):
        if self.evaluation_enabled:
            assert self.eval_episodes != 0, "Evaluation episodes must be greater than 0"
            assert self.eval_env is not None, "Evaluation environment must be defined"

            if self.render:
                assert isinstance(self.eval_env, VecEnv)
                assert isinstance(self.eval_env.envs[0], RecordVideo)
                self.video_folder = self.eval_env.envs[0].video_folder

            _, _ = self.eval_env.reset(seed=self.seed)

        else:
            self.eval_env = None

        return

    #############
    ### Train ###
    #############

    def train(self):
        self.curr_episode = 0
        self.train_violations = []
        eval_table = None
        calibration_alphas = []

        key = jax.random.PRNGKey(seed=self.seed)
        key, buffer_key = jax.random.split(key=key, num=2)

        # Initialize the training buffer with dummy data
        buffer, buffer_state = self.init_replay_buffer(
            buffer_size=self.buffer_size,
            batch_size=self.sample_batch_size,
            buffer_key=buffer_key,
        )

        # Setup and populate validation buffer
        if self.validation_enabled:
            key, val_buffer_key = jax.random.split(key=key, num=2)
            val_buffer, val_buffer_state = self.init_replay_buffer(
                buffer_size=self.val_buffer_size,
                batch_size=self.val_batch_size,
                buffer_key=val_buffer_key,
            )
            val_buffer_state = val_buffer.insert(
                buffer_state=val_buffer_state, samples=self.val_data
            )

        # Initialize the model
        key, model_key = jax.random.split(key=key, num=2)
        model_params, reward_params, cost_params = self.agent.init(key=model_key)

        # Log calibration alphas if the model is a Bayesian Neural Network
        # if self.logging_wandb and has_calibration_alpha(model_params):
        #     # self.set_wandb_custom_axis(["Max Calibration alpha"], self.step_metric)
        #     self.set_wandb_custom_axis(["Calibration alpha"], self.step_metric)
        #     episodic_alpha = model_params.model_state.calibration_alpha
        #     calibration_alphas.append(episodic_alpha)

        #     log_calibration_alpha(
        #         self.curr_episode, calibration_alphas, self.output_dim
        #     )

        ###################
        ### Exploration ###
        ###################
        obs, _ = self.env.reset()

        # exploration_policy = lambda obs, params, rng: self.env.action_space.sample()
        exploration_policy = lambda obs, params, rng: uniform_sampling(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            rng=rng,
        )

        key, exploration_rng = jax.random.split(key=key, num=2)
        exploration_data, obs = rollout_trajectory(
            self.env,
            exploration_policy,
            obs,
            model_params,
            exploration_rng,
            self.num_exploration_steps,
            reset_on_end=False,
        )
        buffer_state = buffer.insert(
            buffer_state=buffer_state, samples=exploration_data
        )

        # Scheduler
        if self.is_minmax and self.logging_wandb:
            self.set_wandb_custom_axis(["Pessimistic Alpha"], self.step_metric)
            if self.is_optminmax:
                self.set_wandb_custom_axis(["Optimistic Alpha"], self.step_metric)

        self.log_episode(exploration_data, model_params, in_loop=False)
        self.log_training_metrics(exploration_data, in_loop=True)

        #################
        ### Pre-Train ###
        #################
        model_params = self.agent.train_step(
            buffer, buffer_state, model_params, diff_states=self.diff_states
        )

        # Evaluate the model after exploration
        if self.evaluation_enabled:
            key, eval_key = jax.random.split(key=key, num=2)
            eval_metrics = self.evaluate_model(
                model_params=model_params, eval_key=eval_key
            )

            # Log the evaluation metrics before training
            self.log_evaluation_metrics(eval_metrics)

            best_params = model_params
            best_avg_reward = eval_metrics["Reward"]

        key, optimizer_rng = jax.random.split(key=key, num=2)

        # Train the agent with model-based planning
        optimization_policy = self.agent.select_best_action

        for i in tqdm(range(self.num_model_updates)):
            obs, _ = self.env.reset()
            self.curr_episode = i + 1

            if self.is_minmax:
                self.agent: MinMaxAgent
                self.agent.update_alpha(self.curr_episode)

            # Rollout the environment with the agent's policy
            transitions, obs = rollout_trajectory(
                self.env,
                optimization_policy,
                obs,
                model_params,
                optimizer_rng,
                self.num_rollout_steps,
                reset_on_end=False,
            )
            self.log_episode(transitions, model_params, in_loop=True)
            self.log_training_metrics(transitions, in_loop=True)

            self.agent.reset_optimizer()
            # transitions = generate_train_data(
            #     self.env, self.num_rollout_steps, num_episodes=1, rng=optimizer_rng
            # )
            buffer_state = buffer.insert(buffer_state=buffer_state, samples=transitions)

            # Update the model parameters
            model_params = self.agent.train_step(
                buffer, buffer_state, model_params, self.diff_states
            )

            # Log calibration alphas if the model is a Bayesian Neural Network
            if has_calibration_alpha(model_params):

                if not self.calibrate:
                    ###########################
                    ### Disable calibration ###
                    ###########################
                    model_params.model_state.calibration_alpha = jnp.ones(
                        shape=(self.output_dim,)
                    )

                # Log the calibration alphas
                episodic_alpha = model_params.model_state.calibration_alpha
                calibration_alphas.append(episodic_alpha)

                # if self.logging_wandb:
                #     log_calibration_alpha(
                #         self.curr_episode, calibration_alphas, self.output_dim
                #     )

            # Validate the model on fixed data
            if self.validation_enabled:
                key, val_key = jax.random.split(key=key, num=2)
                val_metrics, val_buffer_state = self.validate_model(
                    val_buffer, val_buffer_state, model_params, val_key
                )
                if self.logging_wandb:
                    wandb.log(val_metrics)
                else:
                    tabulated_metrics = tabulate(
                        [[x, y] for (x, y) in val_metrics.items()]
                    )
                    tqdm.write(tabulated_metrics)

            # Evaluate the model
            if self.evaluation_enabled:
                key, eval_key = jax.random.split(key=key, num=2)
                eval_metrics = self.evaluate_model(
                    model_params=model_params, eval_key=eval_key
                )
                self.log_evaluation_metrics(eval_metrics, in_loop=True)

                avg_reward = eval_metrics["Reward"]
                if avg_reward > best_avg_reward:
                    best_params = model_params
                    best_avg_reward = avg_reward
                    self.checkpoint(best_params, self.best_params_file)

        # Final Evaluation
        if self.evaluation_enabled:
            key, eval_key = jax.random.split(key=key, num=2)
            final_eval_metrics = self.evaluate_model(
                model_params=model_params, eval_key=eval_key
            )
            self.log_evaluation_metrics(final_eval_metrics)

            avg_reward = final_eval_metrics["Reward"]
            if avg_reward > best_avg_reward:
                best_params = model_params
                best_avg_reward = avg_reward
                self.checkpoint(best_params, self.params_file)

        if self.evaluation_enabled and self.render:
            self.eval_env.close()

        # Create additional constraint violation splots for visualization
        create_violation_plots(
            violations=self.train_violations,
            run_seed=self.seed,
            out_dir=self.out_dir,
            logging_wandb=self.logging_wandb,
        )

        return model_params

    @staticmethod
    def checkpoint(model_params, out_file):
        with open(out_file, "wb") as f:
            pickle.dump(model_params, f)

        return

    """ Validates the model on fixed data sampled at the start of the training."""

    def validate_model(self, val_buffer, val_buffer_state, model_params, val_key):
        val_buffer_state, val_transitions = val_buffer.sample(val_buffer_state)
        val_obs = val_transitions.observation
        val_next_obs = val_transitions.next_observation
        val_action = val_transitions.action
        val_rewards = val_transitions.reward
        model_out, next_obs_pred, rewards_pred, costs_pred = self.agent.validate(
            val_obs, val_action, model_params, val_key
        )

        mean_obs_pred = model_out.mean

        eps_uncertainty = model_out.epistemic_std
        mean_eps_uncertainty = jnp.mean(eps_uncertainty)
        std_eps_uncertainty = jnp.std(eps_uncertainty)
        max_eps_uncertainty = jnp.max(eps_uncertainty)

        al_uncertainty = model_out.aleatoric_std
        mean_al_uncertainty = jnp.mean(al_uncertainty)

        val_mse = jnp.mean((val_next_obs - mean_obs_pred) ** 2)
        reward_mse = jnp.mean((val_rewards - rewards_pred) ** 2)

        # Compute the state MSE for individual elements
        # val_next_state = vmap(self.env.get_state)(val_next_obs)
        # val_state_pred = vmap(self.env.get_state)(mean_obs_pred)
        # val_state_mse = jnp.mean((val_next_state - val_state_pred) ** 2, axis=0)
        # val_metric_names = [
        #     f"Validation {name} MSE" for name in self.env.state_var_names
        # ]
        # val_state_mse_dict = {k: v for k, v in zip(val_metric_names, val_state_mse)}

        # if self.logging_wandb:
        #     self.set_wandb_custom_axis(val_metric_names, self.step_metric)

        val_metrics = {
            "Mean epistemic uncertainty": mean_eps_uncertainty,
            "Std epistemic uncertainty": std_eps_uncertainty,
            # "Max epistemic uncertainty": max_eps_uncertainty,
            "Mean aleatoric uncertainty": mean_al_uncertainty,
            "Validation MSE": val_mse,
            # **val_state_mse_dict,
            # "Reward MSE": reward_mse,
            # "Episode": self.curr_episode,
        }

        return val_metrics, val_buffer_state

    """ Evaluates the performance of a trained model through planning on the environment."""

    def evaluate_model(self, model_params, eval_key):
        assert isinstance(self.eval_env, VecEnv)

        num_envs = self.eval_episodes
        env_instance = self.eval_env.envs[0]

        total_eval_reward = np.zeros(num_envs)
        eval_mse = np.zeros((num_envs, *self.env.dim_observation))
        eval_nll = np.zeros(num_envs)
        num_env_iters = np.zeros(num_envs)
        constraint_violations = np.zeros(num_envs)

        if self.is_minmax:
            predicted_violations = np.zeros(num_envs)

        columns = [
            "Iteration",
            "Obs",
            "Action",
            "Reward",
            "Next_obs",
            "Mean_next_obs",
            "Std_next_obs",
            "Obs_MSE",
            "NLL",
            "Total_reward",
            "Finished",
            "Truncate",
            "Violations",
        ]

        eval_episode_dfs = [pd.DataFrame(columns=columns) for _ in range(num_envs)]

        #######################
        ### Evaluation loop ###
        #######################

        obs, _ = self.eval_env.reset()
        init_obs = obs
        prev_done = np.array(num_envs * [False])

        # Store actions and hal_actions to compute predicted violations
        if self.is_minmax:
            episode_actions = []
            episode_hal_actions = []

        for i in range(self.num_rollout_steps):
            it = i + 1
            eval_key, episode_key = jax.random.split(key=eval_key, num=2)

            #########################
            ### Start the episode ###
            #########################

            episode_key, sample_key = jax.random.split(key=episode_key, num=2)
            env_keys = jax.random.split(sample_key, num=num_envs)

            if self.is_minmax:
                action, hal_action = vmap(
                    self.agent.select_best_action, in_axes=(0, None, 0, None)
                )(obs, model_params, env_keys, True)

                episode_actions.append(action)
                episode_hal_actions.append(hal_action)
            else:
                action = vmap(self.agent.select_best_action, in_axes=(0, None, 0))(
                    obs, model_params, env_keys
                )

            # Simulate the environment
            next_obs, reward, finished, truncate, info = self.eval_env.step(action)

            done = np.array([a or b for a, b in zip(finished, truncate)])
            num_env_iters += np.array([1 if not x else 0 for x in prev_done])

            # Mask which only updates unfinished environments
            indices = np.where(prev_done == False)[0]

            total_eval_reward[indices] += reward[indices]

            episode_key, pred_key = jax.random.split(key=episode_key, num=2)
            env_keys = jax.random.split(pred_key, num=num_envs)

            model_out, pred_next_obs, pred_reward, pred_cost = vmap(
                self.agent.validate, in_axes=(0, 0, None, 0)
            )(obs, action, model_params, env_keys)
            mean_next_obs = model_out.mean
            std_next_obs = get_total_std(model_out)

            # Observed violations of the constraints
            violated = vmap(self.constraint)(next_obs)
            constraint_violations[indices] += np.array(
                [1 if x else 0 for x in violated]
            )[indices]

            curr_eval_mse = (next_obs - mean_next_obs) ** 2
            eval_mse[indices] += curr_eval_mse[indices]

            # This only occurs when there is no uncertainty within the model (True Dynamics)
            if np.all(std_next_obs == 0):
                log_prob = 0.0
            else:
                log_prob = vmap(norm.logpdf, in_axes=(0, 0, 0))(
                    next_obs, mean_next_obs, std_next_obs
                )

            nll = -jnp.mean(log_prob, axis=-1)  # Take the mean wrt the observation dim
            eval_nll[indices] += nll[indices]

            for i in range(num_envs):
                if prev_done[i] == False:
                    # Log the evaluation episode of ONE environment in a dataframe
                    entry = {
                        "Iteration": [it],
                        "Obs": [obs[i].tolist()],
                        "Action": [action[i].tolist()],
                        "Reward": [reward[i]],
                        "Next_obs": [next_obs[i].tolist()],
                        "Mean_next_obs": [mean_next_obs[i].tolist()],
                        "Std_next_obs": [std_next_obs[i].tolist()],
                        "Obs_MSE": [curr_eval_mse[i].tolist()],
                        "NLL": [nll[i].tolist()],
                        "Total_reward": [total_eval_reward[i]],
                        "Finished": [finished[i]],
                        "Truncate": [truncate[i]],
                        "Violations": [constraint_violations[i]],
                    }

                    entry_df = pd.DataFrame(entry)

                    eval_episode_dfs[i] = pd.concat(
                        [eval_episode_dfs[i], entry_df], ignore_index=True
                    )

            # If the environment terminated, it should stay terminated
            prev_done = np.logical_or(prev_done, done)
            obs = next_obs

            ######################
            ### End of episode ###
            ######################

        self.eval_env.close()

        eval_mse = eval_mse / num_env_iters[:, np.newaxis]

        # Compute the predicted violations
        if self.is_minmax:
            episode_actions = np.array(episode_actions)
            episode_hal_actions = np.array(episode_hal_actions)

            episode_key, sample_key = jax.random.split(key=episode_key, num=2)
            env_keys = jax.random.split(sample_key, num=num_envs)

            for i in range(num_envs):
                predicted_violations[i] = self.agent.predict_violations(
                    self.constraint,
                    init_obs[i],
                    episode_actions[i],
                    episode_hal_actions[i],
                    model_params,
                    env_keys[i],
                )

        self.agent.reset_optimizer()

        # Log the evaluation episode
        if self.verbose:
            # if not isinstance(self.env, ConstrainedPendulum):
            #     drop_cols = [
            #         "State",
            #         "Next_state",
            #     ]
            #     eval_episode_dfs = [
            #         df.drop(columns=drop_cols, axis=1) for df in eval_episode_dfs
            #     ]

            pd.set_option("display.max_columns", None)
            log_eval_episode_dfs = [
                df.applymap(format_floats) for df in eval_episode_dfs
            ]

            for i in range(num_envs):
                print(log_eval_episode_dfs[i].to_string(index=False))
                print("-" * (log_eval_episode_dfs[i].shape[1] * 10))

        # Plot ONE evaluation episode
        self.plot_eval_episode(eval_episode_dfs[0])

        # Planning error
        avg_eval_reward = np.mean(total_eval_reward)

        # Model error
        total_eval_mse = np.mean(eval_mse)
        # Average number of environment steps per episode
        avg_iterations = np.mean(num_env_iters)
        # Average number of constraint violations per episode
        avg_violations = np.mean(constraint_violations)

        avg_eval_nll = np.mean(eval_nll)

        pred_violations_dict = {}
        if self.is_minmax:
            avg_pred_violations = np.mean(predicted_violations)
            pred_violations_dict = {"Average predicted violations": avg_pred_violations}

        eval_metrics = {
            "Reward": avg_eval_reward,
            "Average violations": avg_violations,
            **pred_violations_dict,
            "Iterations": avg_iterations,
            "Average NLL": avg_eval_nll,
            "Planning MSE": total_eval_mse,
            # "Episode": self.curr_episode,
        }

        return eval_metrics

    def plot_eval_episode(self, episode_data_df: pd.DataFrame):

        figure = self.env.plot_episode(episode_data_df)

        e = self.curr_episode
        if self.logging_wandb:
            wandb.log({f"Eval_episode_{e}": wandb.Image(figure)})

        # Save the plot
        plot_dir = get_plot_dir(self.out_dir, self.seed)
        plot_file = os.path.join(plot_dir, f"eval_episode_{e}.png")
        figure.savefig(plot_file)

        return

    def log_training_metrics(self, data: Transition, in_loop: bool = False):

        num_violations = vmap(self.constraint)(data.next_observation)
        num_violations = jnp.sum(num_violations)

        max_violation = jnp.max(vmap(self.constraint_deviation)(data.next_observation))
        max_violation = max(0, max_violation)

        self.train_violations.append(num_violations)

        alpha_dict = {}
        if self.is_minmax:
            pes_alpha = self.agent.policy_optimizer.get_pes_alpha()
            alpha_dict = {"Pessimistic Alpha": pes_alpha}
            if self.is_optminmax:
                opt_alpha = self.agent.policy_optimizer.get_opt_alpha()
                alpha_dict["Optimistic Alpha"] = opt_alpha

        metrics = {
            "Train Reward": jnp.sum(data.reward),
            "Train Violations": num_violations,
            "Max Violation": max_violation,
            **alpha_dict,
            "Episode": self.curr_episode,
        }

        tabulated_metrics = tabulate([[x, y] for (x, y) in metrics.items()])

        if self.logging_wandb:
            wandb.log(metrics)

        if in_loop:
            tqdm.write(tabulated_metrics)
        else:
            print(tabulated_metrics)

        return metrics

    def log_episode(self, transitions: Transition, model_params, in_loop: bool = False):
        columns = [
            "Iteration",
            "Obs",
            "Action",
            "Reward",
            "Next_obs",
            "Mean_next_obs",
            "Std_next_obs",
            "Obs_MSE",
            "NLL",
            "Total_reward",
            "Violations",
        ]

        iterations = list(range(len(transitions.observation)))

        num_data = len(transitions.observation)
        index = list(range(num_data))

        obs = transitions.observation
        action = transitions.action
        reward = transitions.reward
        next_obs = transitions.next_observation

        # Randomness is not used
        rng = jax.random.PRNGKey(seed=self.seed)
        model_out, _, _, _ = self.agent.validate(obs, action, model_params, rng)

        mean_next_obs = model_out.mean
        std_next_obs = get_total_std(model_out)

        obs_mse = (next_obs - mean_next_obs) ** 2

        # This only occurs when there is no uncertainty within the model (True Dynamics)
        if np.all(std_next_obs == 0):
            log_prob = std_next_obs
        else:
            log_prob = vmap(norm.logpdf, in_axes=(0, 0, 0))(
                next_obs, mean_next_obs, std_next_obs
            )

        nll = -jnp.mean(log_prob, axis=-1)  # Take the mean wrt the observation dim

        total_reward = jnp.cumsum(reward)

        # Observed violations of the constraints
        violated = vmap(self.constraint)(next_obs)
        num_violations = jnp.sum(violated)

        if num_violations > self.max_violations:
            self.max_violations = num_violations
            self.checkpoint(model_out, self.worst_params_file)

        state_data = {}
        if isinstance(self.env, ConstrainedPendulum):
            state = vmap(self.env.get_state)(obs)
            next_state = vmap(self.env.get_state)(next_obs)
            state_data = {
                "State": state.tolist(),
                "Next_state": next_state.tolist(),
            }

        entry = {
            "Iteration": iterations,
            **state_data,
            "Obs": obs.tolist(),
            "Action": action.tolist(),
            "Reward": reward,
            "Next_obs": next_obs.tolist(),
            "Mean_next_obs": mean_next_obs.tolist(),
            "Std_next_obs": std_next_obs.tolist(),
            "Obs_MSE": obs_mse.tolist(),
            "NLL": nll,
            "Total_reward": total_reward,
            "Violations": violated,
        }

        episode_data_df = pd.DataFrame(index=index, columns=columns)
        episode_data_df = pd.DataFrame(entry)

        # Plot the evaluation episode
        if self.plot_freq > 0 and self.curr_episode % self.plot_freq == 0:
            self.plot_eval_episode(episode_data_df)

        pd.set_option("display.max_columns", None)
        episode_data_df = episode_data_df.map(format_floats)
        episode_data_str = episode_data_df.to_string(index=False)

        if in_loop:
            tqdm.write(episode_data_str)
        else:
            print(episode_data_str)

        return

    def log_evaluation_metrics(self, eval_metrics, in_loop: bool = False):
        if self.logging_wandb:
            wandb.log(eval_metrics)
            if self.render:
                log_eval_video(
                    episode=self.curr_episode, video_folder=self.video_folder
                )

        # Print the evaluation metrics
        tabulated_metrics = tabulate([[x, y] for (x, y) in eval_metrics.items()])

        if in_loop:
            tqdm.write(tabulated_metrics)
        else:
            print(tabulated_metrics)

        return

    def init_replay_buffer(self, buffer_size, batch_size, buffer_key=None):
        """
        Initializes the replay buffer with dummy data and returns the buffer and its state.

        Args:
            buffer_key: The key used to initialize the buffer.

        Returns:
            buffer: The initialized replay buffer.
            buffer_state: The state of the initialized replay buffer.
        """

        if buffer_key is None:
            buffer_key = jax.random.PRNGKey(seed=self.seed)

        dummy_data = get_dummy_transition(self.env)

        buffer = UniformSamplingQueue(
            max_replay_size=buffer_size,
            dummy_data_sample=dummy_data,
            sample_batch_size=batch_size,
        )
        buffer_state = buffer.init(key=buffer_key)

        return buffer, buffer_state


def has_calibration_alpha(model_params: StatisticalModelState[BNNState]):
    return isinstance(model_params, StatisticalModelState) and isinstance(
        model_params.model_state, BNNState
    )
