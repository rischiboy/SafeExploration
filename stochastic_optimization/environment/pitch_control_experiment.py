import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from stochastic_optimization.dynamical_system.rccar_system import (
    RCCarDynamics,
    RCCarSystem,
)
from stochastic_optimization.environment.rccar_env import RCCarSimEnv
from stochastic_optimization.optimizer.cem import CrossEntropyMethod
from stochastic_optimization.optimizer.cem_planner import CEMPlanner
from stochastic_optimization.dynamical_system.pitch_control_system import (
    PitchControlCost,
    PitchControlSystem,
    SafePitchControlSystem,
)
from stochastic_optimization.environment.pitch_control_env import PitchControlEnv
from stochastic_optimization.optimizer.safe_cem_planner import SafeCEMPlanner
from stochastic_optimization.optimizer.utils import (
    BarrierAugmentedLagragian,
    mean_reward,
    plan,
)


def plot_planning_results(env, states, actions, rewards, save_fig=True):
    x_1 = np.arange(len(states))
    x_2 = np.arange(len(actions))
    pitch_angle = states[:, -1]

    tolerance = ENV.angle_tolerance
    desired_angle = ENV.desired_angle
    max_action = ENV.max_action

    # Creating subplots in one row
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    # Plotting the data on each subplot
    axs[0].plot(x_1, pitch_angle, label="Observed Pitch Angle")
    axs[0].set_title("Pitch Angle")

    # Draw a horizontal line at desired angle
    axs[0].axhline(desired_angle, color="red", linestyle="--", label="Desired Angle")
    # Shaded area around desired angle to show tolerance
    axs[0].axhspan(
        -tolerance, tolerance, facecolor="orange", alpha=0.3, label="Tolerance"
    )
    # Setting the axis limits
    min_y, max_y = (-0.3, 0.3)
    axs[0].set_ylim(min_y, max_y)
    # Add yticks
    yticks = np.arange(min_y, max_y + 0.01, 0.1)
    # yticks = np.append(yticks, [-tolerance, tolerance])  # Add ytick at tolerance
    axs[0].set_yticks(yticks)
    axs[0].legend()

    axs[1].plot(x_2, actions)
    axs[1].set_title("Actions")
    # Setting the axis limits
    axs[1].set_ylim(-max_action, max_action)
    yticks = np.arange(-int(max_action), int(max_action) + 0.1, 0.5)
    yticks = np.append(yticks, [-max_action, max_action])
    axs[1].set_yticks(yticks)

    axs[2].plot(x_2, rewards)
    axs[2].set_title("Rewards")

    # Adding labels, title, etc. for the whole figure
    plt.suptitle("Pitch control planning results")
    plt.tight_layout()  # Adjusts subplots to fit into the figure area properly

    if save_fig:
        plt.savefig(f"./plots/pitch_control/planning_results_h{horizon}.png")

    plt.show()
    return


def planning_test(
    ENV, planner, optimize_fn, model_params, key, num_steps=200, plot=True
):

    obs, _ = ENV.reset()
    key, opt_key = jax.random.split(key, 2)

    start_time = time.time()
    transition, done_list = plan(
        env=ENV,
        planner=planner,
        optimize_fn=optimize_fn,
        init_obs=obs,
        model_params=model_params,
        rng=opt_key,
        num_steps=num_steps,
    )
    elapsed_time = time.time() - start_time
    print(f"Total planning time: {elapsed_time:.3f}s")

    last_obs = transition.next_observation[-1]

    # num_state_violations = np.sum(transition.next_observation[:, -1] > 0)
    # num_action_violations = np.sum(np.abs(transition.action) > 0.5)
    # num_attack_violations = np.sum(transition.next_observation[:, 0] > 0.125)

    eval_metrics = [
        ["Average reward", jnp.mean(transition.reward)],
        ["Total episode reward", jnp.sum(transition.reward)],
        ["Last state", last_obs],
        ["Planning successful", done_list[-1]],
        # ["Num State Violations", num_state_violations],
        # ["Num Action Violations", num_action_violations],
        # ["Num Attack Violations", num_attack_violations],
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
    ENV,
    config,
    horizon=[1, 5, 10, 20, 30, 40, 50],
    num_steps=300,
    num_runs=10,
    seed=0,
    plot=True,
    save_fig=True,
):
    sum_rewards = []
    avg_stability_periods = []
    avg_planning_times = []

    for h in tqdm(horizon, desc="Horizon"):
        config["horizon"] = h
        cem_optimizer = CrossEntropyMethod(**config)
        dynamical_system = PitchControlSystem()
        cem_planner = CEMPlanner(
            dynamical_system=dynamical_system, optimizer=cem_optimizer
        )

        key = jax.random.PRNGKey(seed=seed)
        model_params, reward_params = dynamical_system.init(key)

        for i in tqdm(range(num_runs), desc="Runs", leave=False):
            obs, _ = ENV.reset()
            key, opt_key = jax.random.split(key, 2)
            start = time.time()
            states, actions, rewards, done = plan(
                env=ENV,
                planner=cem_planner,
                rng=opt_key,
                init_obs=obs,
                model_params=model_params,
                num_steps=num_steps,
            )
            elapsed_time = time.time() - start
            avg_planning_times.append(elapsed_time)

            stability_period = 0
            if done[-1]:
                false_ind = np.where(~np.array(done))[0]
                last_false_ind = false_ind[-1]
                stability_period = num_sim_steps - last_false_ind

            avg_stability_periods.append(stability_period)
            sum_rewards.append(jnp.sum(rewards))

    sum_rewards = np.array(sum_rewards)
    avg_stability_periods = np.array(avg_stability_periods)
    avg_planning_times = np.array(avg_planning_times)

    sum_rewards = np.mean(sum_rewards.reshape(-1, num_runs), axis=1)
    avg_stability_periods = np.mean(avg_stability_periods.reshape(-1, num_runs), axis=1)
    avg_planning_times = np.mean(avg_planning_times.reshape(-1, num_runs), axis=1)

    # ------------------- Plotting ------------------- #

    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        # Plot average rewards
        axs[0].plot(horizon, sum_rewards)
        axs[0].set_ylabel("Average Rewards")

        # Plot stability periods
        axs[1].plot(horizon, avg_stability_periods)
        axs[1].set_ylabel("Stability Periods")

        # Plot planning times
        axs[2].plot(horizon, avg_planning_times)
        axs[2].set_ylabel("Planning Times")
        axs[2].set_xlabel("Horizon")

        if save_fig:
            plt.savefig("./plots/pitch_control/horizon_test.png")

        plt.tight_layout()
        plt.show()

    return


if __name__ == "__main__":
    # CEM parameters for Planning
    horizon = 20  # default: 20
    num_elites = 50  # default: 50
    num_iter = 10  # default: 10
    num_samples = 500  # default: 500

    num_episodes = 1
    num_sim_steps = 300  # default: 200

    seed = 129088
    rng = jax.random.PRNGKey(seed)

    # Environment parameters
    ENV = RCCarSimEnv(encode_angle=True)
    state_dim = ENV.observation_space.shape
    action_dim = ENV.action_space.shape

    config = {
        "action_dim": action_dim,
        "horizon": horizon,
        "num_elites": num_elites,
        "num_iter": num_iter,
        "num_samples": num_samples,
        "lower_bound": -1.0,
        "upper_bound": 1.0,
    }

    cem_optimizer = CrossEntropyMethod(**config)

    # print("Env step size:", ENV.step_size)

    # Initialize the dynamical system

    dynamics = RCCarDynamics(encode_angle=True)
    true_dynamical_system = RCCarSystem(dynamics=dynamics)
    cem_planner = CEMPlanner(
        dynamical_system=true_dynamical_system, optimizer=cem_optimizer
    )
    rng, init_key = jax.random.split(rng)
    params = true_dynamical_system.init(init_key)
    model_params = params[0]

    # print("Dynamics step size:", model_params.step_size)

    planning_test(
        ENV,
        cem_planner,
        mean_reward,
        model_params,
        rng,
        num_steps=num_sim_steps,
        plot=False,
    )

    # cost = PitchControlCost(max_angle=0.125)
    # true_safe_system = SafePitchControlSystem(cost=cost)
    # safe_cem_planner = SafeCEMPlanner(
    #     safe_dynamical_system=true_safe_system, optimizer=cem_optimizer, num_particles=1
    # )
    # rng, init_key = jax.random.split(rng)
    # params = true_dynamical_system.init(init_key)
    # model_params = params[0]

    # optimize_fn = BarrierAugmentedLagragian(
    #     d=0.0, lmbda=0.005, barrier_type="relu"
    # ).get_function()

    # print("Dynamics step size:", model_params.step_size)

    # planning_test(
    #     ENV,
    #     safe_cem_planner,
    #     optimize_fn,
    #     model_params,
    #     rng,
    #     num_steps=num_sim_steps,
    #     plot=False,
    # )
    # horizon_test(ENV, config, plot=False)
