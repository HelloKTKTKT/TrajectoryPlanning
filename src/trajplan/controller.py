from __future__ import annotations

from typing import Protocol

from trajplan.quadrotor.state import QuadrotorState
from trajplan.shared_types import Vector


class CommonController(Protocol):
    def compute_control(
        self,
        current_state: QuadrotorState,
        reference_pva: Vector,
    ) -> Vector: ...
