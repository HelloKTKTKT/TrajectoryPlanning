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
class TargetStateMessage:
    timestamp: float
    position: Vector
    velocity: Vector
    yaw: float | None = None

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)
        if self.yaw is not None:
            self.yaw = float(self.yaw)

    def copy(self) -> TargetStateMessage:
        return TargetStateMessage(
            timestamp=self.timestamp,
            position=np.asarray(self.position, dtype=np.float64).copy(),
            velocity=np.asarray(self.velocity, dtype=np.float64).copy(),
            yaw=self.yaw,
        )


@dataclass(slots=True)
class TrackingReference:
    target_timestamp: float
    target_position: Vector
    target_velocity: Vector
    formation_offset: Vector
    prediction_lead_time: float

    def __post_init__(self) -> None:
        self.target_timestamp = float(self.target_timestamp)
        self.target_position = np.asarray(
            self.target_position,
            dtype=np.float64,
        ).reshape(3)
        self.target_velocity = np.asarray(
            self.target_velocity,
            dtype=np.float64,
        ).reshape(3)
        self.formation_offset = np.asarray(
            self.formation_offset,
            dtype=np.float64,
        ).reshape(3)
        self.prediction_lead_time = float(self.prediction_lead_time)

    def copy(self) -> TrackingReference:
        return TrackingReference(
            target_timestamp=self.target_timestamp,
            target_position=np.asarray(self.target_position, dtype=np.float64).copy(),
            target_velocity=np.asarray(self.target_velocity, dtype=np.float64).copy(),
            formation_offset=np.asarray(self.formation_offset, dtype=np.float64).copy(),
            prediction_lead_time=self.prediction_lead_time,
        )


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
