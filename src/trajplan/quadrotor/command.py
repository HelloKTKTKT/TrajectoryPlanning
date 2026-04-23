from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from typing import Any

import numpy as np

from trajplan.shared_types import Vector
from trajplan.trajectory.bspline import UniformBSpline
from trajplan.trajectory.polynomial import MinicostTraj


class QuadrotorMode(str, Enum):
    TRACK = "track"
    HOVER = "hover"
    EMERGENCY_STOP = "emergency_stop"
    FINISH = "finish"


@dataclass(slots=True)
class QuadrotorCommand:
    """
    Command committed by the planner manager to the quadrotor agent.

    Design notes
    ------------
    - TRACK requires a valid UniformBSpline trajectory.
    - EMERGENCY_STOP requires a valid stop trajectory.
    - HOVER and FINISH do not carry a trajectory.
    - start_time is the real timestamp when this command becomes active.
    - message is intentionally left generic for future extension.
    """

    mode: QuadrotorMode
    start_time: float
    traj_id: int | None = None
    local_target_pva: Vector | None = None
    trajectory: UniformBSpline | MinicostTraj | None = None
    message: Any | None = None

    def __post_init__(self) -> None:
        self.start_time = float(self.start_time)
        if self.traj_id is not None:
            self.traj_id = int(self.traj_id)

        if self.mode == QuadrotorMode.TRACK and self.trajectory is None:
            raise ValueError("trajectory must not be None when mode is TRACK.")

        if (
            self.mode == QuadrotorMode.TRACK
            and not isinstance(self.trajectory, UniformBSpline)
        ):
            raise TypeError(
                "TRACK command must carry a UniformBSpline trajectory."
            )

        if self.mode == QuadrotorMode.TRACK and self.traj_id is None:
            raise ValueError("traj_id must not be None when mode is TRACK.")

        if self.mode == QuadrotorMode.EMERGENCY_STOP and self.trajectory is None:
            raise ValueError(
                "trajectory must not be None when mode is EMERGENCY_STOP."
            )

        if self.mode == QuadrotorMode.EMERGENCY_STOP and not isinstance(
            self.trajectory, (UniformBSpline, MinicostTraj)
        ):
            raise TypeError(
                "EMERGENCY_STOP command must carry a UniformBSpline or MinicostTraj trajectory."
            )

        if (
            self.mode != QuadrotorMode.TRACK
            and self.mode != QuadrotorMode.EMERGENCY_STOP
            and self.trajectory is not None
        ):
            raise ValueError(
                "trajectory must be None unless mode is TRACK or EMERGENCY_STOP."
            )

        if self.mode != QuadrotorMode.TRACK and self.traj_id is not None:
            raise ValueError("traj_id must be None unless mode is TRACK.")

    @classmethod
    def track(
        cls,
        trajectory: UniformBSpline,
        start_time: float,
        traj_id: int,
        message: Any | None = None,
        local_target_pva: Vector | None = None,
    ) -> QuadrotorCommand:
        return cls(
            mode=QuadrotorMode.TRACK,
            start_time=start_time,
            traj_id=traj_id,
            local_target_pva=local_target_pva,
            trajectory=trajectory,
            message=message,
        )

    @classmethod
    def hover(
        cls,
        start_time: float,
        message: Any | None = None,
    ) -> QuadrotorCommand:
        return cls(
            mode=QuadrotorMode.HOVER,
            start_time=start_time,
            traj_id=None,
            local_target_pva=None,
            trajectory=None,
            message=message,
        )

    @classmethod
    def emergency_stop(
        cls,
        trajectory: UniformBSpline | MinicostTraj,
        start_time: float,
        message: Any | None = None,
        local_target_pva: Vector | None = None,
    ) -> QuadrotorCommand:
        return cls(
            mode=QuadrotorMode.EMERGENCY_STOP,
            start_time=start_time,
            traj_id=None,
            local_target_pva=local_target_pva,
            trajectory=trajectory,
            message=message,
        )

    @classmethod
    def finish(
        cls,
        start_time: float,
        message: Any | None = None,
    ) -> QuadrotorCommand:
        return cls(
            mode=QuadrotorMode.FINISH,
            start_time=start_time,
            traj_id=None,
            local_target_pva=None,
            trajectory=None,
            message=message,
        )

    @property
    def is_track(self) -> bool:
        return self.mode == QuadrotorMode.TRACK

    @property
    def is_hover(self) -> bool:
        return self.mode == QuadrotorMode.HOVER

    @property
    def is_emergency_stop(self) -> bool:
        return self.mode == QuadrotorMode.EMERGENCY_STOP

    @property
    def is_finish(self) -> bool:
        return self.mode == QuadrotorMode.FINISH

    def get_elapsed_time(self, now_time: float) -> float:
        return max(0.0, float(now_time - self.start_time))

    def is_track_expired(
        self,
        now_time: float,
        tolerance: float = 1e-3,
    ) -> bool:
        if not self.is_track or not isinstance(self.trajectory, UniformBSpline):
            return False

        return self.get_elapsed_time(now_time) >= float(self.trajectory.duration) - float(
            tolerance
        )

    def sample_pva(self, now_time: float) -> Vector:
        if self.trajectory is None:
            raise ValueError(
                "sample_pva is only valid when the command carries a trajectory."
            )

        elapsed_time = self.get_elapsed_time(now_time)
        if self.is_track and isinstance(self.trajectory, UniformBSpline):
            traj_duration = float(self.trajectory.duration)
            if elapsed_time >= traj_duration:
                terminal_pva = np.asarray(
                    self.trajectory.evaluate_pva(traj_duration),
                    dtype=np.float64,
                ).reshape(-1)
                terminal_pva[3:9] = 0.0
                return terminal_pva

        return self.trajectory.evaluate_pva(elapsed_time)

    def copy(self) -> QuadrotorCommand:
        local_target_pva_copy = (
            None
            if self.local_target_pva is None
            else np.asarray(self.local_target_pva, dtype=np.float64).copy()
        )
        trajectory_copy = None if self.trajectory is None else self.trajectory.copy()

        return QuadrotorCommand(
            mode=self.mode,
            start_time=self.start_time,
            traj_id=self.traj_id,
            local_target_pva=local_target_pva_copy,
            trajectory=trajectory_copy,
            message=self.message,
        )
