import os
from typing import List, Optional
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Initialize wandb API
api = wandb.Api()


def get_runs(project: str, group: Optional[str] = None, job_type: Optional[str] = None):
    # Retrieve all runs from the specified group
    runs = api.runs(path=f"{project}")

    # Filter runs by group name
    if group is not None:
        group_runs = [run for run in runs if group in run.config["group_name"]]
        runs = group_runs

    # Filter runs by job name
    if job_type is not None:
        jobs = [run for run in group_runs if job_type in run.config["job_name"]]
        runs = jobs

    return runs


def aggregated_histogram_plot(
    project: str,
    group: str,
    job_types: List[str],
    metric: str,
    plot_title: str,
    plot_name: str,
    plot_dir: str,
    bin_size: int = 1,
):

    metric_data = []
    plt.figure()

    for i, job_type in enumerate(job_types):
        # Retrieve all runs from the specified group
        runs = get_runs(project, group, job_type)

        job_metric_data = []

        # Accumulate the metric data from the grouped runs
        for run in runs:
            # Retrieve the metrics
            history = run.scan_history(keys=[metric])
            metric_values = list(history)
            # Unpack values
            metric_values = list(map(lambda x: x[metric], metric_values))
            job_metric_data.extend(metric_values)

        metric_data.append(job_metric_data)

    max_value = max([max(data) for data in metric_data])
    min_value = min([min(data) for data in metric_data])

    bins = range(min_value, max_value + bin_size + 1, bin_size)

    for i, data in enumerate(metric_data):
        label = job_types[i]
        plt.hist(data, bins=bins, alpha=0.5, label=label)

    plt.xlabel(f"{metric}")
    plt.ylabel(f"Frequency")
    plt.xlim(left=min_value)
    plt.legend()
    plt.title(f"{plot_title}")
    plt.tight_layout()

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_name = os.path.join(plot_dir, f"{plot_name}_hist.png")

    plt.savefig(plot_name)

    return


def aggregated_box_plot(
    project: str,
    group: str,
    job_types: List[str],
    metric: str,
    plot_title: str,
    plot_name: str,
    plot_dir: str,
):

    metric_data = []
    plt.figure()

    for i, job_type in enumerate(job_types):
        # Retrieve all runs from the specified group
        runs = get_runs(project, group, job_type)

        job_metric_data = []

        # Accumulate the metric data from the grouped runs
        for run in runs:
            # Retrieve the metrics
            history = run.scan_history(keys=[metric])
            metric_values = list(history)
            # Unpack values
            metric_values = list(map(lambda x: x[metric], metric_values))
            job_metric_data.extend(metric_values)

        metric_data.append(job_metric_data)

    plt.boxplot(metric_data, labels=job_types, vert=False)
    plt.xlabel(f"{metric}")
    plt.ylabel(f"Frequency")
    plt.title(f"{plot_title}")
    plt.tight_layout()

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_name = os.path.join(plot_dir, f"{plot_name}_boxplot.png")

    plt.savefig(plot_name)

    return


if __name__ == "__main__":

    project_name = "Week-23"
    group_name = "DetFSVGD"
    job_types = ["Unsafe-Cal20-DIST", "Safe-Cal20-DIST"]
    metric = "Train Violations"
    plot_title = "Comparison of constraint violations during training"
    plot_name = "train_violations"
    plot_dir = (
        "/cluster/home/ghodkih/MasterThesis/safe_exploration/plots/pendulum/experiments"
    )

    aggregated_histogram_plot(
        project_name,
        group_name,
        job_types,
        metric,
        plot_title,
        plot_name,
        plot_dir,
        bin_size=1,
    )

    aggregated_box_plot(
        project_name,
        group_name,
        job_types,
        metric,
        plot_title,
        plot_name,
        plot_dir,
    )
