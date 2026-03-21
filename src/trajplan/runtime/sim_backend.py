from __future__ import annotations

import numpy as np

from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.sim_state_provider import SimStateProvider
from trajplan.shared_types import Vector


class SimBackend:
    """
    Minimal simulation backend.

    The control input is interpreted directly as commanded acceleration.
    """

    def __init__(
        self,
        state_provider: SimStateProvider,
        max_acceleration_norm: float = 5.0,
    ) -> None:
        self.state_provider = state_provider
        self.max_acceleration_norm = float(max_acceleration_norm)

        if self.max_acceleration_norm <= 0.0:
            raise ValueError(
                "Expected max_acceleration_norm > 0, "
                f"but got {self.max_acceleration_norm}."
            )

    def apply_control(
        self,
        control: Vector,
        dt: float | None = None,
    ) -> None:
        if dt is None:
            raise ValueError("SimBackend requires a valid dt.")

        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"Expected dt > 0, but got {dt}.")

        control = np.asarray(control, dtype=np.float64).reshape(-1)
        if control.shape != (3,):
            raise ValueError(f"Expected control shape (3,), but got {control.shape}.")

        acc_cmd = control.copy()
        acc_norm = np.linalg.norm(acc_cmd)
        if acc_norm > self.max_acceleration_norm:
            acc_cmd = acc_cmd / acc_norm * self.max_acceleration_norm

        current_state = self.state_provider.get_state()

        position = current_state.pva[:3]
        velocity = current_state.pva[3:6]

        next_position = position + velocity * dt + 0.5 * acc_cmd * dt**2
        next_velocity = velocity + acc_cmd * dt
        next_acceleration = acc_cmd

        next_pva = np.hstack((next_position, next_velocity, next_acceleration))
        next_state = QuadrotorState(pva=next_pva)

        self.state_provider.set_state(next_state)
