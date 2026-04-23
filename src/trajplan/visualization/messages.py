from __future__ import annotations

from dataclasses import dataclass

from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.quadrotor.state import QuadrotorState
from trajplan.shared_types import Matrix, Vector


@dataclass(slots=True)
class AgentVisualizationSnapshot:
    agent_id: int
    timestamp: float
    state: QuadrotorState
    active_command: QuadrotorCommand | None


@dataclass(slots=True)
class PlannerVisualizationSnapshot:
    agent_id: int
    timestamp: float
    global_traj_points: Matrix | None
    global_progress_position: Vector | None
