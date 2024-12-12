import os
from gym import Env
from jax import vmap
import wandb
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Union

""" Logs the calibration alpha of the trained model """


def log_calibration_alpha(episode: int, calibration_alphas: List, out_dim: int):
    # episodic_alpha = calibration_alphas[-1]
    # max_episodic_alpha = jnp.max(episodic_alpha)
    # wandb.log(
    #     {
    #         "Max Calibration alpha": max_episodic_alpha,
    #         "Episode": episode,
    #     }
    # )

    y_data = np.array(calibration_alphas).reshape(out_dim, -1)
    keys = [f"Alpha {i}" for i in range(1, out_dim + 1)]
    wandb.log(
        {
            "Calibration_Alpha": wandb.plot.line_series(
                xs=np.arange(0, episode + 1),
                ys=y_data,
                keys=keys,
                title="Calibration alpha",
                xname="Episode",
            )
        }
    )

    return


""" Logs the rendered evaluation video of the trained model """


def log_eval_video(episode: int, video_folder: str):
    assert video_folder is not None, "Video folder is not defined"
    mp4list = glob.glob(video_folder + "*.mp4")
    if len(mp4list) > 0:
        mp4 = mp4list[-1]
        # log gameplay video in wandb
        wandb.log(
            data={
                "Simulation": wandb.Video(
                    mp4,
                    caption="Episode: " + str(episode),
                    fps=5,
                    format="gif",
                ),
            },
        )
    return


########################
### Evaluation plots ###
########################

""" 
Plot the evaluation episode of the pendulum environment 
- True and predicted states
- Planned actions
"""


def get_plot_dir(out_dir: str, run_seed: int):
    plot_dir = os.path.join(out_dir, f"plot_{run_seed}")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    return plot_dir


def custom_box_plot(
    data: List,
    labels: List[str],
    x_label: str,
    title: str,
    run_seed: int,
    plot_name: str,
    out_dir: str,
    logging_wandb: bool = False,
):

    assert len(data) == len(labels), "Data and labels must have the same length"

    # Create a box plot
    plt.figure()
    plt.boxplot(data, labels=labels, vert=False)
    plt.xlabel(f"{x_label}")
    plt.title(f"{title}")
    plt.tight_layout()

    if logging_wandb:
        wandb.log({f"{plot_name}": wandb.Image(plt)})

    plot_dir = get_plot_dir(out_dir, run_seed)
    plot_file = os.path.join(plot_dir, f"{plot_name}.png")
    plt.savefig(plot_file)

    return


def custom_vertical_bar_plot(
    x_values: List,
    y_values: List,
    x_label: str,
    y_label: str,
    title: str,
    run_seed: int,
    plot_name: str,
    out_dir: str,
    logging_wandb: bool = False,
):

    assert len(x_values) == len(y_values), "Data and labels must have the same length"

    # Create a bar plot
    plt.figure()
    plt.bar(x_values, y_values)
    plt.xlabel(f"{x_label}")
    plt.ylabel(f"{y_label}")
    plt.title(f"{title}")
    plt.tight_layout()

    if logging_wandb:
        wandb.log({f"{plot_name}": wandb.Image(plt)})

    plot_dir = get_plot_dir(out_dir, run_seed)
    plot_file = os.path.join(plot_dir, f"{plot_name}.png")
    plt.savefig(plot_file)

    return


def custom_histogram_plot(
    bins: List,
    values: List,
    x_label: str,
    y_label: str,
    title: str,
    run_seed: int,
    plot_name: str,
    out_dir: str,
    logging_wandb: bool = False,
):

    # Create a histogram plot
    plt.figure()
    plt.hist(values, bins=bins)
    plt.xlabel(f"{x_label}")
    plt.ylabel(f"{y_label}")
    plt.xlim(left=min(0, min(values)))
    plt.title(f"{title}")
    plt.tight_layout()

    if logging_wandb:
        wandb.log({f"{plot_name}": wandb.Image(plt)})

    plot_dir = get_plot_dir(out_dir, run_seed)
    plot_file = os.path.join(plot_dir, f"{plot_name}.png")
    plt.savefig(plot_file)

    return


def wandb_histogram_plot(data: List, label: str, title: str):
    data = [[d] for d in data]
    table = wandb.Table(data=data, columns=[label])
    wandb.log({f"{label}_hist": wandb.plot.histogram(table, label, title=f"{title}")})
    return


def create_violation_plots(
    violations: List, run_seed: int, out_dir: str, logging_wandb: bool
):
    x_values = range(len(violations))

    title = "Violations during Training"

    if logging_wandb:
        group_name = wandb.config.group_name  # Gets the group name of the current run
    else:
        group_name = "Group"

    group_name = group_name + "_" + str(run_seed)

    custom_box_plot(
        data=[violations],
        labels=["Violations"],
        x_label="Number of Violations",
        title=title,
        run_seed=run_seed,
        plot_name="box_plot_violations",
        out_dir=out_dir,
        logging_wandb=logging_wandb,
    )

    custom_vertical_bar_plot(
        x_values=x_values,
        y_values=violations,
        x_label="Episodes",
        y_label="Violations",
        title=title,
        run_seed=run_seed,
        plot_name="bar_plot_violations",
        out_dir=out_dir,
        logging_wandb=logging_wandb,
    )

    bin_size = 1
    bins = range(0, max(violations) + bin_size + 1, bin_size)
    custom_histogram_plot(
        bins=bins,
        values=violations,
        x_label="Violations",
        y_label="Frequency of Violations",
        title=title,
        run_seed=run_seed,
        plot_name="histogram_violations",
        out_dir=out_dir,
        logging_wandb=logging_wandb,
    )

    if logging_wandb:
        wandb_histogram_plot(data=violations, label="Violations", title=title)

    return


##############
### UNUSED ###
##############

""" Populates the evaluation table with the evaluation metrics. """


def log_eval_data(eval_table, episode: int, data: Dict):
    data = {"Episode": episode, **data}
    table_data = pd.DataFrame(data=data, index=[0])

    if eval_table is None:
        columns = list(data.keys())
        eval_table = wandb.Table(data=table_data)
    else:
        data = list(data.values())
        eval_table.add_data(*data)

    # Create a copy of the evaluation table to log the updated version
    table_copy = wandb.Table(columns=eval_table.columns, data=eval_table.data)
    wandb.log({"Eval_table": table_copy})

    return eval_table


""" Plots the evaluation table. """


def plot_eval_data(eval_table):
    columns = eval_table.columns
    x_label = columns[0]
    assert (
        x_label == "Episode"
    ), "The first column of the table must be called <<Episode>>"

    # Log the evaluation table
    table_copy = wandb.Table(columns=eval_table.columns, data=eval_table.data)
    wandb.log({"Eval_table": table_copy})

    # Create a plot for each column/metric
    for y_label in columns[1:]:
        plot_id = f"{y_label} plot"
        wandb.log(
            {
                plot_id: wandb.plot.line(
                    eval_table,
                    x=x_label,
                    y=y_label,
                    title=f"{y_label}",
                )
            }
        )
        return
