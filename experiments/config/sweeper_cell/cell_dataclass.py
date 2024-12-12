from dataclasses import dataclass


@dataclass
class SweeperCellParams:
    num_cpus: int
    num_gpus: int
    time: str
    mem: int
