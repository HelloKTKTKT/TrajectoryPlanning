from __future__ import annotations

import numpy as np
from trajplan.controller import DifferentialFlatnessController
from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.backend import CommonBackend
from trajplan.runtime.sim_backend import SimBackend
from trajplan.runtime.state_provider import StateProvider
from trajplan.shared_types import Vector


class QuadrotorAgent:
    def __init__(
        self,
        state_provider: StateProvider,
        backend: CommonBackend,
        controller: DifferentialFlatnessController | None = None,
    ) -> None:
        self.state_provider = state_provider
        self.backend = backend
        self.controller = controller

        self.active_command: QuadrotorCommand | None = None
        self.current_state: QuadrotorState | None = None
        self.current_reference_pvaj: Vector | None = None
        self.is_finished = False

        self._hover_reference_pvaj: Vector | None = None
        self._hover_reference_key: tuple[str, float] | None = None

    def set_active_command(
        self,
        command: QuadrotorCommand,
    ) -> None:
        self.active_command = command

    def clear_active_command(self) -> None:
        self.active_command = None

    def get_current_state(self) -> QuadrotorState | None:
        if self.current_state is None:
            return None
        return self.current_state.copy()

    def get_current_reference_pvaj(self) -> Vector | None:
        if self.current_reference_pvaj is None:
            return None
        return self.current_reference_pvaj.copy()

    def step(
        self,
        dt: float,
        now_time: float,
    ) -> None:
        if self.is_finished:
            return

        if not self.state_provider.is_initialized():
            return

        current_state = self.state_provider.get_state()
        if current_state is None:
            return

        self.current_state = current_state
        command = self.active_command

        if command is None:
            reference_pvaj = self._get_hover_reference(
                current_state=current_state,
                hover_key=("no_command", -1.0),
            )

        elif command.is_finish:
            self.is_finished = True
            self._clear_hover_reference()
            self.current_reference_pvaj = None
            self.backend.stop()
            return

        elif command.is_hover:
            reference_pvaj = self._get_hover_reference(
                current_state=current_state,
                hover_key=("hover", command.start_time),
            )

        elif command.is_track:
            self._clear_hover_reference()
            reference_pvaj = command.sample_pvaj(now_time)

        else:
            raise ValueError(f"Unsupported command mode: {command.mode}")

        self.current_reference_pvaj = np.asarray(
            reference_pvaj,
            dtype=np.float64,
        ).reshape(-1)
        self._execute_reference(
            current_state=current_state,
            reference_pvaj=reference_pvaj,
            dt=dt,
        )

    def _execute_reference(
        self,
        current_state: QuadrotorState,
        reference_pvaj: Vector,
        dt: float,
    ) -> None:
        if isinstance(self.backend, SimBackend):
            if self.controller is None:
                raise RuntimeError(
                    "Simulation path requires a controller, but controller is None."
                )

            rotor_thrust = self.controller.compute_control(
                current_state=current_state,
                reference_pva=np.asarray(reference_pvaj, dtype=np.float64).reshape(-1)[
                    :9
                ],
            )
            self.backend.apply_control(
                control=np.asarray(rotor_thrust, dtype=np.float64).reshape(-1),
                dt=dt,
            )
            return

        self.backend.send_reference_pvaj(reference_pvaj)

    def _get_hover_reference(
        self,
        current_state: QuadrotorState,
        hover_key: tuple[str, float],
    ) -> Vector:
        if self._hover_reference_pvaj is None or self._hover_reference_key != hover_key:
            self._hover_reference_pvaj = self._build_hover_reference(current_state)
            self._hover_reference_key = hover_key

        return self._hover_reference_pvaj.copy()

    def _clear_hover_reference(self) -> None:
        self._hover_reference_pvaj = None
        self._hover_reference_key = None

    @staticmethod
    def _build_hover_reference(
        current_state: QuadrotorState,
    ) -> Vector:
        position = np.asarray(current_state.position, dtype=np.float64).reshape(-1)
        zeros = np.zeros(9, dtype=np.float64)
        return np.hstack((position, zeros))
