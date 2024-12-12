from dataclasses import dataclass
import sys
from typing import Dict, List
from omegaconf import OmegaConf
import wandb
from config.sweep.sweep_dataclass import SweepParams
from config.sweeper_cell.cell_dataclass import SweeperCellParams
from type_utils import OptimizerType, DynamicsType
from util import (
    generate_base_command,
    generate_run_commands,
    hash_dict,
    sample_flag,
    RESULT_DIR,
)
import yaml
import numpy as np
import os
import hydra
from hydra.core.config_store import ConfigStore
import datetime


@dataclass
class PendulumSweepConfig:
    sweep: SweepParams
    sweeper_cell: SweeperCellParams
    num_agents: int
    seed: int


def launch_agents_cli(
    log_dir: str, sweeper_cell_cfg: SweeperCellParams, sweep_id: str, num_agents: int
):
    cmd = "wandb agent " + sweep_id
    command_list = [cmd] * num_agents

    num_cpus = sweeper_cell_cfg.num_cpus
    num_gpus = sweeper_cell_cfg.num_gpus
    time = sweeper_cell_cfg.time
    mem = sweeper_cell_cfg.mem

    generate_run_commands(
        script_dir=log_dir,
        command_list=command_list,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        mem=mem,
        mode="euler",
        time=time,
    )

    return command_list


def launch_agents_script(
    log_dir: str, cfg: PendulumSweepConfig, sweep_id: str, num_agents: int
):
    import cem_pendulum_sweep as experiment

    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(experiment.__file__)

    base_cmd = interpreter_script + " -u " + base_exp_script
    base_cmd += f" +exp_result_folder={log_dir}"
    base_cmd += f" +sweep_id={sweep_id}"
    base_cmd += f" pendulum.train.seed={cfg.seed}"
    base_cmd += f" pendulum.wandb.project_name={cfg.sweep.project}"
    base_cmd += f" pendulum.wandb.group_name={cfg.sweep.name}"

    command_list = [base_cmd] * num_agents

    num_cpus = cfg.sweeper_cell.num_cpus
    num_gpus = cfg.sweeper_cell.num_gpus
    time = cfg.sweeper_cell.time
    mem = cfg.sweeper_cell.mem

    generate_run_commands(
        script_dir=log_dir,
        command_list=command_list,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        mem=mem,
        mode="euler",
        time=time,
    )

    return command_list


# Global variables
config_path = "config"
config_name = "default_sweep.yaml"
cs = ConfigStore.instance()
cs.store(name=config_name, node=PendulumSweepConfig)


@hydra.main(
    config_path=config_path,
    config_name=config_name,
    version_base=None,
)
def main(cfg: PendulumSweepConfig):
    exp_base_path = os.path.join(RESULT_DIR, "pendulum")
    exp_path = os.path.join(exp_base_path, "CEM-Pendulum")

    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    print("Log folder: ", time_stamp)
    exp_result_folder = os.path.join(exp_path, time_stamp)

    if exp_result_folder[-1] != "/":
        exp_result_folder += "/"

    seed = cfg.seed
    num_agents = cfg.num_agents
    sweep_cfg = OmegaConf.to_container(cfg.sweep)
    sweeper_cell_cfg = cfg.sweeper_cell

    # Add additional arguments to the base command
    sweep_cfg["command"] = sweep_cfg["command"] + [
        "+exp_result_folder=" + exp_result_folder,
    ]
    sweep_cfg["command"] = sweep_cfg["command"] + [
        "pendulum.train.seed=" + str(seed),
    ]
    sweep_cfg["command"] = sweep_cfg["command"] + [
        "pendulum.wandb.project_name=" + cfg.sweep.project,
    ]
    sweep_cfg["command"] = sweep_cfg["command"] + [
        "pendulum.wandb.group_name=" + cfg.sweep.name,
    ]

    sweep_id = wandb.sweep(sweep_cfg)
    full_sweep_id = cfg.sweep.entity + "/" + cfg.sweep.project + "/" + sweep_id
    print("Sweep ID: ", full_sweep_id)

    # launch_agents_cli(
    #     log_dir=exp_result_folder,
    #     sweeper_cell_cfg=sweeper_cell_cfg,
    #     sweep_id=full_sweep_id,
    #     num_agents=num_agents,
    # )

    launch_agents_script(
        log_dir=exp_result_folder,
        cfg=cfg,
        sweep_id=full_sweep_id,
        num_agents=num_agents,
    )

    print(f"Launched {num_agents} agents.")


if __name__ == "__main__":
    main()
