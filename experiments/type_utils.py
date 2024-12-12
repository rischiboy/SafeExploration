from enum import Enum
from pydantic import BaseModel


class DynamicsType(Enum):
    TRUE = 0
    BNN = 1
    GP = 2


class OptimizerType(Enum):
    CEM = 0
    ICEM = 1
    MinMax = 2


class AgentType(Enum):
    CEM = 0
    SafeCEM = 1
    PesTraj = 2
    MinMax = 3
    OptMinMax = 4


class Objective(Enum):
    UNSAFE = 0
    SAFE = 1
    MinMax = 2
    OptMinMax = 3


class SweeperParams(BaseModel):
    num_cpus: int
    num_gpus: int
    time: str
    mem: int
