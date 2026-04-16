from __future__ import annotations

import numpy as np

from trajplan.controller import CommonController
from trajplan.quadrotor.state import QuadrotorState
from trajplan.shared_types import Vector


class CrazyflieController(CommonController):
    """
    Lightweight controller for Crazyflie full-state execution.

    The planner already produces a translational p/v/a reference. For the
    Crazyflie path we forward that reference downstream and let the backend
    derive the additional cmdFullState terms (yaw, omega).
    """

    def compute_control(
        self,
        current_state: QuadrotorState,
        reference_pva: Vector,
    ) -> Vector:
        _ = current_state
        return np.asarray(reference_pva, dtype=np.float64).reshape(-1)
