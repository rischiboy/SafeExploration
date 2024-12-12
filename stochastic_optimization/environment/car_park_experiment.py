import time
from typing import Dict, List
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from stochastic_optimization.dynamical_system import abstract_dynamical_system
from stochastic_optimization.dynamical_system.car_park_system import CarParkSystem
from stochastic_optimization.dynamical_system.pendulum_system import PendulumSystem
from stochastic_optimization.environment.car_park_env import CarParkEnv
from stochastic_optimization.environment.env_consts import CarParkConsts
from stochastic_optimization.environment.pendulum_env import ConstrainedPendulum
from stochastic_optimization.environment.pitch_control_env import PitchControlEnv
from stochastic_optimization.optimizer.cem import CrossEntropyMethod
from stochastic_optimization.optimizer.cem_planner import CEMPlanner
from stochastic_optimization.dynamical_system.pitch_control_system import (
    PitchControlSystem,
)
from stochastic_optimization.optimizer.utils import mean_reward, plan


def plot_planning_results(
    env: CarParkEnv,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    save_fig: bool = True,
):
    x_1 = np.arange(len(states))
    x_2 = np.arange(len(actions))
    carPosition = states[:, 0]
    carSpeed = states[:, 1]
    max_idx = np.argmax(carPosition)
    maxCarPosition = np.max(carPosition)

    bottom_margin = env.bottom_margin
    top_margin = env.top_margin
    destination = env.destination
    max_action = env.max_action

    # Creating subplots in one row
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    # Plotting the data on each subplot
    axs[0].plot(x_1, carPosition, label="Observed Position")
    axs[0].plot(max_idx, maxCarPosition, "ro", label="Max Position")
    axs[0].annotate(
        f"{maxCarPosition:.2f}",
        (max_idx, maxCarPosition),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
    )
    axs[0].set_title("Car Position")

    # Draw a horizontal line at desired angle
    axs[0].axhline(destination, color="red", linestyle="--", label="Destination")
    # Shaded area around desired angle to show tolerance
    axs[0].axhspan(
        (destination + bottom_margin),
        (destination + top_margin),
        facecolor="orange",
        alpha=0.3,
        label="Tolerance",
    )
    # Setting the axis limits
    min_y, max_y = (0, CarParkConsts.MAX_POSITION)
    axs[0].set_ylabel("Position [m]", color="b")
    axs[0].set_ylim(min_y, max_y)
    # Add yticks
    yticks = np.arange(min_y, max_y + 0.01, 1)
    # yticks = np.append(yticks, [-tolerance, tolerance])  # Add ytick at tolerance
    axs[0].set_yticks(yticks)
    axs[0].legend()

    axs_dup = axs[0].twinx()
    axs_dup.plot(x_1, carSpeed, "g", label="Observed Speed")
    axs_dup.set_ylabel("Speed [m/s]", color="g")
    axs_dup.set_ylim(min_y, max_y)

    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = axs_dup.get_legend_handles_labels()
    axs[0].legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    axs[1].plot(x_2, actions)
    axs[1].set_title("Actions")
    # Setting the axis limits
    axs[1].set_ylim(-1, 1)
    yticks = np.arange(-1, 1 + 0.1, 0.2)
    yticks = np.append(yticks, [-1, 1])
    axs[1].set_yticks(yticks)

    axs[2].plot(x_2, rewards)
    axs[2].set_title("Rewards")

    # Adding labels, title, etc. for the whole figure
    plt.suptitle("Car Park planning results")
    plt.tight_layout()  # Adjusts subplots to fit into the figure area properly

    if save_fig:
        plt.savefig(
            f"./plots/car_park/Week-21/Full_stop/Short_SC_AC/planning_h{horizon}.png"
        )

    plt.show()
    return


def planning_test(ENV, config, seed=0, num_steps=200, plot=True):
    key = jax.random.PRNGKey(seed=seed)

    dynamical_system = true_dynamical_system
    model_params, reward_params = dynamical_system.init(key)

    cem_optimizer = CrossEntropyMethod(**config)
    cem_planner = CEMPlanner(dynamical_system=dynamical_system, optimizer=cem_optimizer)

    obs, _ = ENV.reset(seed=seed)
    key, opt_key = jax.random.split(key, 2)

    start_time = time.time()
    transition, done_list = plan(
        env=ENV,
        planner=cem_planner,
        optimize_fn=mean_reward,
        init_obs=obs,
        model_params=model_params,
        rng=opt_key,
        num_steps=num_steps,
        debug=DEBUG,
    )
    elapsed_time = time.time() - start_time
    print(f"Total planning time: {elapsed_time:.3f}s")

    last_obs = transition.next_observation[-1]

    eval_metrics = [
        ["Horizon", config["horizon"]],
        ["Average reward", jnp.mean(transition.reward)],
        ["Total episode reward", jnp.sum(transition.reward)],
        ["Max Position", np.max(transition.observation[:, 0])],
        ["Last state", last_obs],
        ["Planning successful", done_list[-1]],
    ]

    if done_list[-1]:
        false_ind = np.where(~np.array(done_list))[0]
        last_false_ind = false_ind[-1]
        stability_period = num_sim_steps - last_false_ind
        # print("Pendulum upright for last: ", stability_period, " steps")
        eval_metrics.append(["Stability period", stability_period])

    tabulated_metrics = tabulate(eval_metrics)
    print(tabulated_metrics)

    if plot:
        plot_planning_results(
            ENV, transition.observation, transition.action, transition.reward
        )


def horizon_test(
    env,
    dynamical_system: abstract_dynamical_system,
    config: Dict,
    horizon: List,
    num_steps: int = 200,
    num_runs: int = 5,
    seed: int = 0,
    plot: bool = True,
    save_fig: bool = True,
):
    sum_rewards = []
    stability_periods = []
    planning_times = []

    for h in tqdm(horizon, desc="Horizon"):
        config["horizon"] = h
        cem_optimizer = CrossEntropyMethod(**config)
        cem_planner = CEMPlanner(
            dynamical_system=dynamical_system, optimizer=cem_optimizer
        )

        key = jax.random.PRNGKey(seed=seed)
        model_params, reward_params = dynamical_system.init(key)

        obs, _ = env.reset(seed=seed)

        for i in tqdm(range(num_runs), desc="Runs", leave=False):
            obs, _ = env.reset()
            key, opt_key = jax.random.split(key, 2)
            start = time.time()
            transition, done_list = plan(
                env=env,
                planner=cem_planner,
                optimize_fn=mean_reward,
                rng=opt_key,
                init_obs=obs,
                model_params=model_params,
                num_steps=num_steps,
            )
            elapsed_time = time.time() - start
            planning_times.append(elapsed_time)

            stability_period = 0
            if done_list[-1]:
                false_ind = np.where(~np.array(done_list))[0]
                last_false_ind = false_ind[-1]
                stability_period = num_sim_steps - last_false_ind
            # else:
            #     print("Planning failed")
            #     print(f"Stability period: {env.stablized_steps}")
            #     print(f"Action: {transition.action}")
            #     print(f"Next state: {transition.next_observation}")

            stability_periods.append(stability_period)
            sum_rewards.append(jnp.sum(transition.reward))

    sum_rewards = np.array(sum_rewards)
    stability_periods = np.array(stability_periods)
    planning_times = np.array(planning_times)

    sum_rewards = np.mean(sum_rewards.reshape(-1, num_runs), axis=1)
    avg_stability_periods = np.mean(stability_periods.reshape(-1, num_runs), axis=1)
    num_finished = np.count_nonzero(stability_periods.reshape(-1, num_runs), axis=1)
    avg_planning_times = np.mean(planning_times.reshape(-1, num_runs), axis=1)

    for i in range(len(horizon)):
        eval_metrics = [
            ["Horizon", horizon[i]],
            ["Average reward", sum_rewards[i]],
            ["Planning successful", num_finished[i]],
        ]

        tabulated_metrics = tabulate(eval_metrics)
        print(tabulated_metrics)

    # ------------------- Plotting ------------------- #

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # Plot average rewards
        axs[0, 0].plot(horizon, sum_rewards)
        axs[0, 0].set_xlabel("Horizon")
        axs[0, 0].set_ylabel("Average Rewards")
        axs[0, 0].set_ylim(bottom=0)

        # Plot stability periods
        axs[0, 1].plot(horizon, avg_stability_periods)
        axs[0, 1].set_xlabel("Horizon")
        axs[0, 1].set_ylabel("Stability Periods")
        axs[0, 1].set_ylim(0, num_sim_steps)

        # Plot number of finished runs
        axs[1, 0].plot(horizon, num_finished)
        axs[1, 0].set_xlabel("Horizon")
        axs[1, 0].set_ylabel("Number of Finished Runs")
        axs[1, 0].set_ylim(0, num_runs)

        # Plot planning times
        axs[1, 1].plot(horizon, avg_planning_times)
        axs[1, 1].set_xlabel("Horizon")
        axs[1, 1].set_ylabel("Planning Times")
        axs[1, 1].set_ylim(bottom=0)

        if save_fig:
            plt.savefig(
                "./plots/car_park/Week-21/Full_stop/Short_SC_AC/horizon_test.png"
            )

        plt.tight_layout()
        plt.show()

    return


if __name__ == "__main__":
    DEBUG = False

    # CEM parameters for Planning
    horizon = 10  # default: 20
    num_elites = 50  # default: 50
    num_iter = 10  # default: 10
    num_samples = 500  # default: 500

    num_episodes = 1
    num_sim_steps = 200  # default: 200

    # Environment parameters
    max_action = CarParkConsts.MAX_ACTION
    ENV = CarParkEnv(max_action=max_action)
    true_dynamical_system = CarParkSystem()

    # ENV = ConstrainedPendulum()
    # true_dynamical_system = PendulumSystem()

    # ENV = PitchControlEnv()
    # true_dynamical_system = PitchControlSystem()

    state_dim = ENV.observation_space.shape
    action_dim = ENV.action_space.shape

    config = {
        "action_dim": action_dim,
        "horizon": horizon,
        "num_elites": num_elites,
        "num_iter": num_iter,
        "num_samples": num_samples,
        "lower_bound": -1,
        "upper_bound": 1,
    }

    # planning_test(ENV, config, seed=10230, num_steps=num_sim_steps, plot=True)

    horizon = [1, 2, 3, 5, 10, 20]
    horizon_test(
        env=ENV,
        dynamical_system=true_dynamical_system,
        config=config,
        horizon=horizon,
        num_steps=num_sim_steps,
        num_runs=100,
        seed=40415,
        plot=True,
        save_fig=True,
    )
