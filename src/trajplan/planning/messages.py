from __future__ import annotations

from dataclasses import dataclass

from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.quadrotor.state import QuadrotorState


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

    state: QuadrotorState | None
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
