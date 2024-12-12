import matplotlib.pyplot as plt
import numpy as np
import os
import wandb
import pandas as pd
import math
import pickle


api = wandb.Api()
entity = "safex"

# baselines = ["Unsafe", "Safe-MEAN", "Safe-DIST", "PesTraj", "MinMax"]
baselines = ["Unsafe", "Safe-MEAN", "Safe-DIST", "PesTraj", "MinMax", "OptMinMax"]
metrics = ["Train Reward", "Train Violations"]

violation_tolerance = 1e-4

"""Pendulum"""

pendulum = {
    "entity": entity,
    "project": "Final",
    "group": None,
    "jobs": [
        "LR-WD-FEAT128",
        "L1-MEAN-SafeCEM",
        "L1-SafeCEM",
        "L1-PesTrajCEM",
        "L1-MinMaxCEM",
        "L1-OptMinMaxCEM",
    ],
    "baselines": baselines,
    "metrics": metrics,
    # "num_episodes": 50,
}

pendulum_lambda = {
    "entity": entity,
    "project": "Final-Pendulum",
    "group": None,
    "jobs": [
        "L0.1",
        "L0.2",
        "L0.5",
        "L1-MinMaxCEM",
    ],
    "baselines": ["λ=0.1", "λ=0.2", "λ=0.5", "λ=1.0"],
    "metrics": metrics,
    # "num_episodes": 50,
}

"""Pitch Control"""

# pitch_control = {
#     "entity": entity,
#     "project": "Final-PC",
#     "group": "Final-Exp50",
#     "jobs": [
#         "UnsafeCEM",
#         "MEAN-SafeCEM",
#         "DIST-SafeCEM",
#         "PesTrajCEM",
#         "MinMax",
#         "OptMinMax-Scheduler",
#     ],
#     "baselines": baselines,
#     "metrics": metrics,
#     # "num_episodes": 10,
# }

pitch_control = {
    "entity": entity,
    "project": "Final-PC",
    "group": "LR-0.001",
    "jobs": [
        "UnsafeCEM",
        "MEAN-SafeCEM",
        "SafeCEM",
        "PesTraj",
        "MinMax-Constant",
        "OptMinMax-Constant",
    ],
    "baselines": baselines,
    "metrics": metrics,
    # "num_episodes": 10,
}

"""RCCar"""
rccar = {
    "entity": entity,
    "project": "Final-RCCar",
    "group": None,
    "jobs": [
        "Unsafe",
        "L1-MEAN-Safe",
        "L1-DIST-Safe",
        "L1-PesTrajCEM",
        "L1-DIST-MinMax",
        "L1-DIST-OptMinMax",
    ],
    "baselines": baselines,
    "metrics": metrics,
    # "num_episodes": 50,
}

pes_alpha_rccar = {
    "entity": entity,
    "project": "Final-RCCar",
    "group": "MinMax-CEM",
    "jobs": [
        "L1-DIST-MinMax",
        "L1-DIST-MinMax-Constant",
        "L1-DIST-MinMax-Constant2.0",
        "L1-DIST-MinMax-Constant3.0",
    ],
    "baselines": ["Scheduler", "Beta=1.0", "Beta=2.0", "Beta=3.0"],
    "metrics": ["Pessimistic Alpha", "Train Reward", "Train Violations"],
}


def fetch_data(entity, project, group, jobs, baselines, metrics):
    baseline_data = {}
    project_url = entity + "/" + project

    if group is not None:
        filters = {"group": group}
    else:
        filters = {}

    for job_name, base in zip(jobs, baselines):

        filters["jobType"] = job_name
        job_runs = api.runs(project_url, filters=filters)
        # print("Job Name: ", job_name, "\tBaseline: ", base)

        job_rewards = []
        job_violations = []

        print("Fetching data for ", job_name, "...")

        for run in job_runs:
            history = run.scan_history(keys=metrics, page_size=100)
            rewards = [row["Train Reward"] for row in history]
            violations = [row["Train Violations"] for row in history]

            # Remove Exploration metrics
            rewards = rewards[1:]
            violations = violations[1:]

            job_rewards.append(rewards)
            job_violations.append(violations)

            # print(run.name)

        baseline_data[base] = {"rewards": job_rewards, "violations": job_violations}

    return baseline_data


def fetch_pes_alpha_data(entity, project, group, jobs, baselines, metrics):
    baseline_data = {}
    project_url = entity + "/" + project

    if group is not None:
        filters = {"group": group}
    else:
        filters = {}

    for job_name, base in zip(jobs, baselines):

        filters["jobType"] = job_name
        job_runs = api.runs(project_url, filters=filters)
        # print("Job Name: ", job_name, "\tBaseline: ", base)

        job_alpha = []
        job_rewards = []
        job_violations = []

        print("Fetching data for ", job_name, "...")

        for run in job_runs:
            history = run.scan_history(keys=metrics, page_size=100)
            alpha = [row["Pessimistic Alpha"] for row in history]
            rewards = [row["Train Reward"] for row in history]
            violations = [row["Train Violations"] for row in history]

            # Remove Exploration metrics
            alpha = alpha[1:]
            rewards = rewards[1:]
            violations = violations[1:]

            job_alpha.append(alpha)
            job_rewards.append(rewards)
            job_violations.append(violations)

            # print(run.name)

        baseline_data[base] = {
            "alpha": job_alpha,
            "rewards": job_rewards,
            "violations": job_violations,
        }

    return baseline_data


def plot_baseline_data(experiments):

    num_envs = len(experiments)
    num_metrics = len(metrics)
    # fig, axs = plt.subplots(
    #     num_metrics,
    #     num_envs,
    #     figsize=(14, 10),
    #     gridspec_kw={
    #         "hspace": 0.3,
    #         "wspace": 0.2,
    #     },
    # )

    fig, axs = plt.subplots(
        num_envs,
        num_metrics,
        figsize=(8, 8),
        gridspec_kw={
            "hspace": 0.5,
            "wspace": 0.3,
        },
    )

    labels = []
    lines = []

    for i, (env, baseline_data) in enumerate(experiments.items()):

        box_plot_dict = {}
        labels = baseline_data.keys()

        for base, data in baseline_data.items():
            rewards = np.array(data["rewards"])
            violations = np.array(data["violations"])
            episodes = np.arange(len(rewards[0]))

            mean_reward = np.mean(rewards, axis=0)
            std_reward = np.std(rewards, axis=0)
            min_reward = np.min(rewards, axis=0)
            max_reward = np.max(rewards, axis=0)
            sum_violations = np.sum(violations, axis=1)

            box_plot_dict[base] = sum_violations

            # (line,) = axs[0, i].plot(episodes, mean_reward)

            # if i == 0:
            #     lines.append(line)

            # # Fill between the lines for the shaded region (mean +/- std)
            # # axs[0, i].fill_between(
            # #     episodes, mean_reward - std_reward, mean_reward + std_reward, alpha=0.3
            # # )

            # # Fill between the lines for the shaded region (min, max)
            # axs[0, i].fill_between(episodes, min_reward, max_reward, alpha=0.3)

            # axs[0, i].set_title(f"{env} Reward")
            # axs[0, i].set_xlabel("Episode")
            # axs[0, i].set_ylabel("Reward")

            # Fill between the lines for the shaded region (mean +/- std)
            # axs[0, i].fill_between(
            #     episodes, mean_reward - std_reward, mean_reward + std_reward, alpha=0.3
            # )

            # Fill between the lines for the shaded region (min, max)
            if num_envs == 1:
                (line,) = axs[0].plot(episodes, mean_reward)
                axs[0].fill_between(episodes, min_reward, max_reward, alpha=0.3)

                axs[0].set_title(f"{env} Reward")
                axs[0].set_xlabel("Episode")
                axs[0].set_ylabel("Reward")
            else:
                (line,) = axs[i, 0].plot(episodes, mean_reward)
                axs[i, 0].fill_between(episodes, min_reward, max_reward, alpha=0.3)

                axs[i, 0].set_title(f"{env} Reward")
                axs[i, 0].set_xlabel("Episode")
                axs[i, 0].set_ylabel("Reward")

            if i == 0:
                lines.append(line)

        box_plot_data = list(box_plot_dict.values())
        box_plot_labels = list(box_plot_dict.keys())

        # if i == 0:
        #     axs[1, i].boxplot(box_plot_data, labels=box_plot_labels, vert=False)
        # else:
        #     axs[1, i].boxplot(box_plot_data, vert=False)
        #     axs[1, i].set_yticks([])
        # axs[1, i].set_title(f"{env} Violations")
        # axs[1, i].set_xlabel("Total Violations")

        if num_envs == 1:
            axs[1].boxplot(box_plot_data, labels=box_plot_labels, vert=False)
            axs[1].set_title(f"{env} Violations")
            axs[1].set_xlabel("Total Violations")
        else:

            axs[i, 1].boxplot(box_plot_data, labels=box_plot_labels, vert=False)
            axs[i, 1].set_title(f"{env} Violations")
            axs[i, 1].set_xlabel("Total Violations")

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.legend(
        handles=lines,
        labels=labels,
        loc="upper center",
        ncol=4,
    )

    plt.show()

    return fig


def plot_pes_alpha(experiments):

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(16, 8),
        gridspec_kw={
            "wspace": 0.25,
        },
    )

    lines = []
    labels = experiments.keys()
    box_plot_dict = {}

    for i, (base, data) in enumerate(experiments.items()):
        alphas = np.array(data["alpha"])
        rewards = np.array(data["rewards"])
        violations = np.array(data["violations"])
        episodes = np.arange(len(rewards[0]))

        alphas = np.mean(alphas, axis=0)
        mean_reward = np.mean(rewards, axis=0)
        min_reward = np.min(rewards, axis=0)
        max_reward = np.max(rewards, axis=0)
        mean_violations = np.mean(violations, axis=0)
        min_violations = np.min(violations, axis=0)
        max_violations = np.max(violations, axis=0)
        sum_violations = np.sum(violations, axis=1)

        box_plot_dict[base] = sum_violations

        (line,) = axs[0].plot(episodes, alphas)
        lines.append(line)

        # Fill between the lines for the shaded region (min, max)
        axs[1].plot(episodes, mean_reward)
        axs[1].fill_between(episodes, min_reward, max_reward, alpha=0.3)

        # axs[2].plot(episodes, mean_violations)
        # axs[2].fill_between(episodes, min_violations, max_violations, alpha=0.3)

    axs[0].set_title(f"RCCar Pessimistic Beta")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Beta")

    axs[1].set_title(f"RCCar Reward")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Reward")

    box_plot_data = list(box_plot_dict.values())
    box_plot_labels = list(box_plot_dict.keys())
    axs[2].boxplot(box_plot_data, labels=box_plot_labels, vert=False)

    axs[2].set_title(f"RCCar Violations")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Violations")

    fig.legend(
        handles=lines,
        labels=labels,
        loc="upper center",
        ncol=4,
    )

    plt.show()

    return fig


def store_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_data(filename, remove_key, episodes):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    for key in remove_key:
        if key in data:
            data.pop(key)

    for key, value in data.items():
        for metric, datapoints in value.items():
            value[metric] = list(map(lambda d: d[:episodes], datapoints))

    return data


if __name__ == "__main__":
    pendulum_file = "pendulum_data.pkl"
    pitch_control_file = "pitch_control_data-lr.pkl"
    rc_car_file = "rccar_data.pkl"
    pes_alpha_file = "pes_alpha_rccar.pkl"
    pendulum_lambda_file = "pendulum_lambda.pkl"

    unwanted_baseline = ["Unsafe", "OptMinMax"]

    if os.path.exists(pendulum_file):
        pendulum_data = load_data(pendulum_file, unwanted_baseline, 30)
    else:
        pendulum_data = fetch_data(**pendulum)
        # print(pendulum_data)
        store_data(pendulum_data, pendulum_file)

    if os.path.exists(pitch_control_file):
        pitch_control_data = load_data(pitch_control_file, unwanted_baseline, 6)
    else:
        pitch_control_data = fetch_data(**pitch_control)
        # print(pitch_control_data)
        store_data(pitch_control_data, pitch_control_file)

    if os.path.exists(rc_car_file):
        rccar_data = load_data(rc_car_file, unwanted_baseline, 50)
    else:
        rccar_data = fetch_data(**rccar)
        # print(rccar_data)
        store_data(rccar_data, rc_car_file)

    if os.path.exists(pes_alpha_file):
        pes_alpha_data = load_data(pes_alpha_file, [], 50)
    else:
        pes_alpha_data = fetch_pes_alpha_data(**pes_alpha_rccar)
        # print(pes_alpha_data)
        store_data(pes_alpha_data, pes_alpha_file)

    if os.path.exists(pendulum_lambda_file):
        pendulum_lambda_data = load_data(pendulum_lambda_file, unwanted_baseline, 30)
    else:
        pendulum_lambda_data = fetch_data(**pendulum_lambda)
        # print(pendulum_lambda_data)
        store_data(pendulum_lambda_data, pendulum_lambda_file)

    data = {
        "Pendulum": pendulum_data,
        "Pitch Control": pitch_control_data,
        # "RCCar": rccar_data,
    }

    # fig = plot_baseline_data(data)
    # fig.savefig("final_plots.png")

    # fig = plot_pes_alpha(pes_alpha_data)
    # fig.savefig("final_pes_alpha.png")

    pendulum_lambda_data = {
        "Pendulum": pendulum_lambda_data,
    }
    fig = plot_baseline_data(pendulum_lambda_data)
    fig.savefig("final_lambda.png")

    # runs = api.runs(
    #     entity + "/Final-PC",
    #     filters={"group": "Final-Exp50", "jobType": "UnsafeCEM"},
    # )
    # test_run = runs[0]
    # print(test_run.name)
    # print(len(runs))
