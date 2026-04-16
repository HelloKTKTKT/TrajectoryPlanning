from __future__ import annotations
import numpy as np

from trajplan.runtime.backend import CommonBackend
from trajplan.runtime.sim_state_provider import SimStateProvider
from trajplan.runtime.state_provider import StateProvider
from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.quadrotor.state import QuadrotorState
from trajplan.shared_types import Vector
from trajplan.controller import CommonController


class QuadrotorAgent:
    def __init__(
        self,
        state_provider: StateProvider,
        backend: CommonBackend,
        controller: CommonController,
    ) -> None:
        self.state_provider = state_provider
        self.backend = backend
        self.controller = controller

        self.active_command: QuadrotorCommand | None = None
        self.current_reference_pva: Vector | None = None
        self.is_finished = False

        self._hover_reference_pva: Vector | None = None
        self._hover_reference_key: tuple[str, float] | None = None

    def set_active_command(
        self,
        command: QuadrotorCommand,
    ) -> None:
        self.try_commit_command(command)

    def clear_active_command(self) -> None:
        self.active_command = None
        self._clear_hover_reference()

    def try_commit_command(
        self,
        command: QuadrotorCommand,
    ) -> bool:
        current_command = self.active_command
        if (
            current_command is not None
            and current_command.is_track
            and command.is_track
            and current_command.traj_id == command.traj_id
        ):
            return False

        self.active_command = command.copy()
        self._clear_hover_reference()
        return True

    def get_state(self) -> QuadrotorState:
        return self.state_provider.get_state()

    def get_current_reference_pva(self) -> Vector | None:
        if self.current_reference_pva is None:
            return None
        return self.current_reference_pva.copy()

    def get_active_command(self) -> QuadrotorCommand | None:
        if self.active_command is None:
            return None
        return self.active_command.copy()

    def step(
        self,
        now_time: float,
        dt: float,
    ) -> None:
        if self.is_finished:
            return

        current_state = self.state_provider.get_state()

        command = self.active_command

        if command is None:
            # No active command, just hover at the current position.
            reference_pva = self._get_hover_reference(
                current_state=current_state,
                hover_key=("no_command", -1.0),
            )
        elif command.is_finish:
            self.is_finished = True
            self._clear_hover_reference()
            self.current_reference_pva = None
            self.backend.stop()
            return

        elif command.is_hover:
            reference_pva = self._get_hover_reference(
                current_state=current_state,
                hover_key=("hover", command.start_time),
            )

        elif command.is_track or command.is_emergency_stop:
            self._clear_hover_reference()
            reference_pva = command.sample_pva(now_time=now_time)

        else:
            raise ValueError(f"Unsupported command mode: {command.mode}")

        self.current_reference_pva = np.asarray(
            reference_pva,
            dtype=np.float64,
        ).reshape(-1)
        self._execute_reference(
            current_state=current_state,
            reference_pva=reference_pva,
            dt=dt,
        )

    def _get_hover_reference(
        self,
        current_state: QuadrotorState,
        hover_key: tuple[str, float],
    ) -> Vector:
        if self._hover_reference_pva is None or self._hover_reference_key != hover_key:
            self._hover_reference_pva = self._build_hover_reference(current_state)
            self._hover_reference_key = hover_key

        return self._hover_reference_pva.copy()

    def _clear_hover_reference(self) -> None:
        self._hover_reference_pva = None
        self._hover_reference_key = None

    def _execute_reference(
        self,
        current_state: QuadrotorState,
        reference_pva: Vector,
        dt: float,
    ) -> None:
        cal_control = self.controller.compute_control(
            current_state=current_state,
            reference_pva=reference_pva,
        )
        next_state = self.backend.apply_control(
            control=np.asarray(cal_control, dtype=np.float64).reshape(-1),
            current_state=current_state,
            dt=dt,
        )

        if next_state is not None:
            if not isinstance(self.state_provider, SimStateProvider):
                raise RuntimeError(
                    "Backend returned a simulated state, but state_provider is not SimStateProvider."
                )
            self.state_provider.set_state(next_state)

    @staticmethod
    def _build_hover_reference(
        current_state: QuadrotorState,
    ) -> Vector:
        position = np.asarray(current_state.position, dtype=np.float64).reshape(-1)
        zeros = np.zeros(6, dtype=np.float64)
        return np.hstack((position, zeros))
