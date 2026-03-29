from __future__ import annotations

import numpy as np

from trajplan.quadrotor.dynamics import (
    dynamics_vector_to_state,
    first_euler_step,
    rk4_step,
    state_to_dynamics_vector,
)
from trajplan.quadrotor.model import QuadrotorPhysicalConfig
from trajplan.runtime.backend import CommonBackend
from trajplan.runtime.sim_state_provider import SimStateProvider
from trajplan.shared_types import Vector


class SimBackend(CommonBackend):
    """
    Simulation backend using rotor-thrust input and quadrotor dynamics.

    Responsibilities
    ----------------
    - accept rotor thrust command
    - propagate the simulated quadrotor state by one step
    - write the updated state back into SimStateProvider

    Design note
    -----------
    In the current architecture, the simulation path is driven by:
        current_state + reference_pva
            -> controller.compute_control(...)
            -> rotor_thrust
            -> sim_backend.apply_control(rotor_thrust, dt)

    Therefore `apply_control()` is the main execution interface for simulation.
    """

    def __init__(
        self,
        state_provider: SimStateProvider,
        physical_config: QuadrotorPhysicalConfig,
        integration_method: str = "rk4",
        min_rotor_thrust: float = 0.0,
    ) -> None:
        self.state_provider = state_provider
        self.physical_config = physical_config
        self.integration_method = str(integration_method).lower()
        self.min_rotor_thrust = float(min_rotor_thrust)

        self._stopped = False

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

    def send_reference_pva(
        self,
        pva: Vector,
    ) -> None:
        raise NotImplementedError(
            "SimBackend does not use send_reference_pva() in the current design. "
            "Use apply_control(rotor_thrust, dt) instead."
        )

    def stop(self) -> None:
        self._stopped = True

    def apply_control(
        self,
        control: Vector,
        dt: float,
    ) -> None:
        """
        Apply one rotor-thrust control command and advance the simulation.

        Parameters
        ----------
        control
            Rotor thrust vector, shape (rotor_count,).
        dt
            Simulation time step in seconds.
        """
        if self._stopped:
            return

        if not self.state_provider.is_initialized():
            raise RuntimeError("SimStateProvider is not initialized.")

        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"Expected dt > 0, but got {dt}.")

        rotor_thrust = self._clip_rotor_thrust(control)

        current_state = self.state_provider.get_state()
        if current_state is None:
            raise RuntimeError("SimStateProvider returned None state.")

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
        self.state_provider.set_state(next_state)

    def _clip_rotor_thrust(
        self,
        control: Vector,
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
