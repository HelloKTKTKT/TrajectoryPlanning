from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.quadrotor.state import QuadrotorState
from trajplan.shared_types import Vector
from trajplan.trajectory.bspline import UniformBSpline


@dataclass(slots=True)
class PlannerTickInput:
    """
    One planning-cycle input snapshot.

    Fields
    ------
    state
        Latest available quadrotor state.
        None means the planner currently has no state estimate.
    timestamp
        Wall-clock or simulation time in seconds for this planning tick.
    active_command
        Command currently being executed by the agent, if any.
        The planner can use this for trajectory extension / continuity.
    """

    state: QuadrotorState
    timestamp: float
    active_command: QuadrotorCommand | None = None

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)


@dataclass(slots=True)
class PlannerTickOutput:
    """
    One planning-cycle output.

    For now the stable planner contract is intentionally small:
    the planner receives a snapshot and returns the next command.
    Extra debug/status information should be added separately later
    instead of bloating the runtime-facing interface.
    """

    command: QuadrotorCommand


@dataclass(slots=True)
class NeighborTrajectoryMessage:
    agent_id: int
    traj_id: int
    start_time: float
    trajectory: UniformBSpline
    local_target_pva: Vector | None = None

    def __post_init__(self) -> None:
        self.agent_id = int(self.agent_id)
        self.traj_id = int(self.traj_id)
        self.start_time = float(self.start_time)
        if self.local_target_pva is not None:
            self.local_target_pva = np.asarray(
                self.local_target_pva,
                dtype=np.float64,
            ).reshape(-1)

    def copy(self) -> NeighborTrajectoryMessage:
        local_target_pva_copy = (
            None
            if self.local_target_pva is None
            else np.asarray(self.local_target_pva, dtype=np.float64).copy()
        )
        return NeighborTrajectoryMessage(
            agent_id=self.agent_id,
            traj_id=self.traj_id,
            start_time=self.start_time,
            trajectory=self.trajectory.copy(),
            local_target_pva=local_target_pva_copy,
        )


def is_newer_neighbor_message(
    candidate: NeighborTrajectoryMessage,
    reference: NeighborTrajectoryMessage,
) -> bool:
    if candidate.traj_id != reference.traj_id:
        return candidate.traj_id > reference.traj_id
    return candidate.start_time > reference.start_time
