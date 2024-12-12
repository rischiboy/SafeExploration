from dataclasses import dataclass
from typing import Optional

from sweep.sweep_config import (
    WandbParams,
    CEMParams,
    MinMaxParams,
    ModelParams,
    TrainParams,
)


@dataclass
class EnvParams:
    angle_tolerance: float
    stability_duration: int
    max_steps: int
    max_speed: float


@dataclass
class ConstraintParams:
    speed_threshold: float
    lmbda: float
    d: float


@dataclass
class PendulumTrainer:
    wandb: WandbParams
    env: EnvParams
    cem: CEMParams
    model: ModelParams
    train: TrainParams


@dataclass
class SafePendulumTrainer(PendulumTrainer):
    constraint: ConstraintParams


@dataclass
class MinMaxPendulumTrainer:
    wandb: WandbParams
    env: EnvParams
    constraint: ConstraintParams
    minmax: MinMaxParams
    model: ModelParams
    train: TrainParams


@dataclass
class DefaultPendulumTrainer:
    # defaults: list[dict]
    pendulum: PendulumTrainer
    exp_result_folder: str
    sweep_id: Optional[str] = None


@dataclass
class DefaultSafePendulumTrainer:
    # defaults: list[dict]
    pendulum: SafePendulumTrainer
    exp_result_folder: str
    sweep_id: Optional[str] = None
