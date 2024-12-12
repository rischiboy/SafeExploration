from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MetricParams:
    name: str
    goal: str


@dataclass
class SweepParams:
    name: str
    entity: str
    project: str
    run_cap: int
    program: str
    method: str
    metric: MetricParams
    parameters: Dict
    command: List[str]
