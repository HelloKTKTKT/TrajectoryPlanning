from __future__ import annotations

import numpy as np

from trajplan.quadrotor.dynamics import (
    dynamics_vector_to_state,
    first_euler_step,
    rk4_step,
    state_to_dynamics_vector,
)
from trajplan.quadrotor.model import QuadrotorPhysicalConfig
from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.backend import CommonBackend
from trajplan.shared_types import Vector


class SimBackend(CommonBackend):
    def __init__(
        self,
        physical_config: QuadrotorPhysicalConfig,
        integration_dt: float,
        integration_method: str = "rk4",
        min_rotor_thrust: float = 0.0,
    ) -> None:
        self.physical_config = physical_config
        self.integration_dt = float(integration_dt)
        self.integration_method = str(integration_method).lower()
        self.min_rotor_thrust = float(min_rotor_thrust)

        self.is_stopped = False

        if self.integration_dt <= 0.0:
            raise ValueError(
                f"Expected integration_dt > 0, but got {self.integration_dt}."
            )
        if self.integration_method not in {"rk4", "euler"}:
            raise ValueError(
                "Expected integration_method to be 'rk4' or 'euler', "
                f"but got {self.integration_method}."
            )
        if self.min_rotor_thrust < 0.0:
            raise ValueError(
                f"Expected min_rotor_thrust >= 0, but got {self.min_rotor_thrust}."
            )
        if self.min_rotor_thrust > self.physical_config.max_force:
            raise ValueError(
                "Expected min_rotor_thrust <= physical_config.max_force, "
                f"but got {self.min_rotor_thrust} > {self.physical_config.max_force}."
            )

    def apply_control(
        self,
        control: object,
        current_state: QuadrotorState,
        dt: float,
    ) -> QuadrotorState | None:
        if self.is_stopped:
            return None

        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"Expected dt > 0, but got {dt}.")

        rotor_thrust = self._clip_rotor_thrust(control)
        xk = state_to_dynamics_vector(current_state)

        if self.integration_method == "rk4":
            x_next = rk4_step(
                xk=xk,
                uk=rotor_thrust,
                ts=dt,
                physical_config=self.physical_config,
            )
        else:
            x_next = first_euler_step(
                xk=xk,
                uk=rotor_thrust,
                ts=dt,
                physical_config=self.physical_config,
            )

        next_state = dynamics_vector_to_state(
            xin=x_next,
            rotor_thrust=rotor_thrust,
            physical_config=self.physical_config,
        )
        return next_state

    def stop(self) -> None:
        self.is_stopped = True

    def _clip_rotor_thrust(
        self,
        control: object,
    ) -> Vector:
        rotor_thrust = np.asarray(control, dtype=np.float64).reshape(-1)

        if rotor_thrust.shape != (self.physical_config.rotor_count,):
            raise ValueError(
                "Expected control shape "
                f"({self.physical_config.rotor_count},), "
                f"but got {rotor_thrust.shape}."
            )

        return np.clip(
            rotor_thrust,
            self.min_rotor_thrust,
            self.physical_config.max_force,
        )
