from dataclasses import dataclass
from typing import Optional


@dataclass
class WandbParams:
    project_name: str
    group_name: str
    job_name: str
    logs_dir: str
    logging_wandb: bool


@dataclass
class CEMParams:
    horizon: int
    num_iter: int
    num_elites: int
    num_samples: int


@dataclass
class MinMaxParams:
    horizon_x: int
    num_iter_x: int
    num_fixed_elites_x: int
    num_elites_x: int
    num_samples_x: int
    horizon_y: int
    num_iter_y: int
    num_fixed_elites_y: int
    num_elites_y: int
    num_samples_y: int
    alpha: float


@dataclass
class ModelParams:
    dynamics_type: str
    agent_type: str
    bnn_type: Optional[str] = None
    sampling_mode: Optional[str] = None
    output_stds: Optional[str] = None
    num_training_steps: Optional[int] = None
    beta: Optional[str] = None
    features: Optional[str] = None
    lr_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    num_particles: Optional[int] = None
    train_share: Optional[float] = None
    batch_size: Optional[int] = None
    eval_frequency: Optional[int] = None
    eval_batch_size: Optional[int] = None
    return_best_model: Optional[bool] = None


@dataclass
class TrainParams:
    seed: int
    sample_batch_size: int
    buffer_size: int
    num_model_updates: int
    num_rollout_steps: int
    num_exploration_steps: int
    val_buffer_size: int
    val_batch_size: int
    eval_episodes: int
    eval_model_freq: int
    diff_states: bool
    verbose: bool
    render: bool
