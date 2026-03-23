from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from trajplan.shared_types import Vector
from trajplan.trajectory.bspline import UniformBSpline


class QuadrotorMode(str, Enum):
    TRACK = "track"
    HOVER = "hover"
    FINISH = "finish"


@dataclass(slots=True)
class QuadrotorCommand:
    """
    Command committed by the planner manager to the quadrotor agent.

    Design notes
    ------------
    - TRACK requires a valid UniformBSpline trajectory.
    - HOVER and FINISH do not carry a trajectory.
    - start_time is the real timestamp when this command becomes active.
    - message is intentionally left generic for future extension.
    """

    mode: QuadrotorMode
    start_time: float
    trajectory: UniformBSpline | None = None
    message: Any | None = None

    def __post_init__(self) -> None:
        self.start_time = float(self.start_time)

        if self.mode == QuadrotorMode.TRACK and self.trajectory is None:
            raise ValueError("trajectory must not be None when mode is TRACK.")

        if self.mode != QuadrotorMode.TRACK and self.trajectory is not None:
            raise ValueError("trajectory must be None unless mode is TRACK.")

    @classmethod
    def track(
        cls,
        trajectory: UniformBSpline,
        start_time: float,
        message: Any | None = None,
    ) -> QuadrotorCommand:
        return cls(
            mode=QuadrotorMode.TRACK,
            start_time=start_time,
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
            trajectory=None,
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
    def is_finish(self) -> bool:
        return self.mode == QuadrotorMode.FINISH

    def get_elapsed_time(self, now_time: float) -> float:
        return max(0.0, float(now_time - self.start_time))

    def sample_pva(self, now_time: float) -> Vector:
        if not self.is_track:
            raise ValueError("sample_pva is only valid when mode is TRACK.")
        if self.trajectory is None:
            raise ValueError("trajectory must not be None when mode is TRACK.")

        elapsed_time = self.get_elapsed_time(now_time)
        return self.trajectory.evaluate_pva(elapsed_time)
