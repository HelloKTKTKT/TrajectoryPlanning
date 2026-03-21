from __future__ import annotations

import numpy as np

from trajplan.quadrotor.state import QuadrotorState
from trajplan.shared_types import Vector


class Controller:
    """
    Minimal acceleration-level tracking controller.

    Input
    -----
    - current quadrotor state pva
    - reference pva

    Output
    ------
    - commanded acceleration, shape (3,)
    """

    def __init__(
        self,
        kp_position: float = 2.0,
        kv_velocity: float = 1.5,
        max_acceleration_norm: float = 5.0,
    ) -> None:
        self.kp_position = float(kp_position)
        self.kv_velocity = float(kv_velocity)
        self.max_acceleration_norm = float(max_acceleration_norm)

        if self.kp_position < 0.0:
            raise ValueError(f"Expected kp_position >= 0, but got {self.kp_position}.")
        if self.kv_velocity < 0.0:
            raise ValueError(f"Expected kv_velocity >= 0, but got {self.kv_velocity}.")
        if self.max_acceleration_norm <= 0.0:
            raise ValueError(
                "Expected max_acceleration_norm > 0, "
                f"but got {self.max_acceleration_norm}."
            )

    def compute_control(
        self,
        current_state: QuadrotorState,
        reference_pva: Vector,
    ) -> Vector:
        reference_pva = np.asarray(reference_pva, dtype=np.float64).reshape(-1)

        if reference_pva.shape != (9,):
            raise ValueError(
                f"Expected reference_pva shape (9,), but got {reference_pva.shape}."
            )

        position = current_state.pva[:3]
        velocity = current_state.pva[3:6]

        reference_position = reference_pva[:3]
        reference_velocity = reference_pva[3:6]
        reference_acceleration = reference_pva[6:9]

        position_error = reference_position - position
        velocity_error = reference_velocity - velocity

        acceleration_command = (
            reference_acceleration
            + self.kp_position * position_error
            + self.kv_velocity * velocity_error
        )

        command_norm = np.linalg.norm(acceleration_command)
        if command_norm > self.max_acceleration_norm:
            acceleration_command = (
                acceleration_command / command_norm * self.max_acceleration_norm
            )

        return acceleration_command
