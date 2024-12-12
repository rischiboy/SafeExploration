from config.job.job_dataclass import JobConfig
from type_utils import Objective, OptimizerType, DynamicsType
from util import (
    generate_base_command,
    generate_run_commands,
    hash_dict,
    sample_flag,
    RESULT_DIR,
)
import yaml
import argparse
import numpy as np
import copy
import os
import itertools
import hydra
from hydra.core.config_store import ConfigStore
import datetime

# Global variables
config_path = "config/job"
config_name = "pitch_control_job.yaml"
cs = ConfigStore.instance()
cs.store(name=config_name, node=JobConfig)

objective = Objective.MinMax  # ["UNSAFE", "SAFE", "MinMax", "OptMinMax"]
dynamics_model = DynamicsType.BNN  # ["TRUE", "BNN", "GP"]

job_out_dir = f"{objective.name}/{dynamics_model.name}"
job_config_file = (
    f"{objective.name.lower()}_{dynamics_model.name.lower()}_pitch_control"
)

launch_cmd_args = {
    "pitch_control": job_config_file,  # Experiment configuration file
}


@hydra.main(
    config_path=config_path,
    config_name=config_name,
    version_base=None,
)
def main(cfg: JobConfig):
    import cem_pitch_control_test as experiment

    params = cfg.params
    sbatch = cfg.sbatch

    rds = np.random.RandomState(params.seed)
    assert params.num_seeds_per_hparam < 100
    init_seeds = list(rds.randint(0, 10**6, size=(100,)))

    command_list = []
    for _ in range(params.num_hparam_samples):
        if params.launch_mode == "euler":
            logs_dir = "/cluster/scratch/"
            logs_dir += params.user_name + "/"
        else:
            logs_dir = "experiments/results/pitch_control/"

        exp_base_path = os.path.join(RESULT_DIR, "pitch_control")
        exp_path = os.path.join(exp_base_path, job_out_dir)

        # Wandb logging directory
        flags = {"logs_dir": logs_dir}

        time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
        print("Log folder: ", time_stamp)
        exp_result_folder = os.path.join(exp_path, time_stamp)

        if exp_result_folder[-1] != "/":
            exp_result_folder += "/"

        # Experiment logs and results directory
        flags["exp_result_folder"] = exp_result_folder

        # Add the necessary launch configs to the flags
        flags.update(launch_cmd_args)

        for j in range(params.num_seeds_per_hparam):
            seed = init_seeds[j]
            # cmd = generate_base_command(
            #     experiment, flags=dict(**flags, **{"seed": seed}), is_hydra=True
            # )
            cmd = generate_base_command(
                experiment,
                flags=dict(**flags, **{"pitch_control.train.seed": seed}),
                is_hydra=True,
            )
            command_list.append(cmd)

    # submit jobs
    generate_run_commands(
        exp_result_folder,
        command_list,
        num_cpus=sbatch.num_cpus,
        num_gpus=sbatch.num_gpus,
        mode=params.launch_mode,
        long=sbatch.long_run,
        prompt=params.prompt,
        mem=sbatch.mem,
        time=sbatch.time,
    )


if __name__ == "__main__":
    main()
