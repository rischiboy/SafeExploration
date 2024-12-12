import os
import argparse
from util import AsyncExecutor


def generate_commands(sweep_id, num_agents, script_dir):
    commands = []
    for i in range(num_agents):
        command = f"CUDA_VISIBLE_DEVICES={i} wandb agent {sweep_id} > {script_dir}/sweeper_cell_{i}.log 2>&1 &"
        commands.append(command)
    return commands


def main(args):
    # Parse arguments
    sweep_id = args.sweep_id
    num_agents = args.num_agents
    script_dir = args.script_dir

    # Generate commands
    cmd_list = generate_commands(sweep_id, num_agents, script_dir)

    executor = AsyncExecutor(n_jobs=num_agents)
    cmd_exec_fun = lambda cmd: os.system(cmd)
    executor.run(cmd_exec_fun, cmd_list)

    print(f"Started {num_agents} agents for sweep {sweep_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweeper Cell Launcher")

    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--num_agents", type=int, default=None)
    parser.add_argument("--script_dir", type=str, default=None)

    args = parser.parse_args()
    main(args)
