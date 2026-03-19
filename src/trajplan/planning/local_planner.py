from __future__ import annotations

from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.shared_types import Vector
from trajplan.trajectory.bspline import UniformBSpline


class LocalPlanner:
    def plan(
        self,
        active_command: QuadrotorCommand | None,
        local_start_pva: Vector,
        local_target_pva: Vector,
        current_time: float,
    ) -> UniformBSpline | None:
        raise NotImplementedError
