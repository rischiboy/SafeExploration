from dataclasses import dataclass
from dataclasses import dataclass


@dataclass
class JobParams:
    seed: int
    launch_mode: str
    user_name: str
    prompt: bool
    num_hparam_samples: int
    num_seeds_per_hparam: int


@dataclass
class SbatchParams:
    num_cpus: int
    num_gpus: int
    time: str
    long_run: bool
    mem: int


@dataclass
class JobConfig:
    params: JobParams
    sbatch: SbatchParams
