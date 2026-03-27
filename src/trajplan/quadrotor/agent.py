from __future__ import annotations

import numpy as np

from trajplan.controller import Controller
from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.backend import Backend
from trajplan.runtime.state_provider import StateProvider
from trajplan.shared_types import Vector


class QuadrotorAgent:
    """
    Minimal quadrotor agent execution loop.

    Responsibilities
    ----------------
    - get the latest quadrotor state
    - read the latest committed command
    - convert the command into a reference pva
    - call controller to compute control
    - send control to backend

    Important design note
    ---------------------
    This class does not call the planner manager directly.
    Planning should run asynchronously outside this class and update the
    active command through `set_active_command()`.
    """

    def __init__(
        self,
        controller: Controller,
        state_provider: StateProvider,
        backend: Backend,
    ) -> None:
        self.controller = controller
        self.state_provider = state_provider
        self.backend = backend

        self.active_command: QuadrotorCommand | None = None
        self.current_state: QuadrotorState | None = None

        self.is_finished: bool = False

    def set_active_command(self, command: QuadrotorCommand) -> None:
        """
        Replace the current active command.

        This method is intended to be called by an external planning loop.
        """
        self.active_command = command

    def clear_active_command(self) -> None:
        """
        Clear the current command.

        The agent will fall back to hover behavior.
        """
        self.active_command = None

    def get_current_state(self) -> QuadrotorState | None:
        """
        Return the latest cached state.
        """
        return self.current_state

    def step(
        self,
        dt: float,
        now_time: float,
    ) -> None:
        """
        Run one execution step.

        Parameters
        ----------
        dt
            Optional control interval.
            In simulation it can be passed to the backend.
            In real flight it can be ignored by the backend if not needed.
        """
        if self.is_finished:
            return

        current_state = self.state_provider.get_state()
        self.current_state = current_state

        command = self.active_command

        if command is None:
            reference_pva = self._build_hover_reference(current_state)

        elif command.is_finish:
            self.is_finished = True
            return

        elif command.is_hover:
            reference_pva = self._build_hover_reference(current_state)

        else:
            reference_pva = command.sample_pva(now_time)

        control = self.controller.compute_control(
            current_state=current_state,
            reference_pva=reference_pva,
        )
        # self.backend.apply_control(control=control, dt=dt)
        self.backend.apply_control_direct_set_pva(pva=reference_pva)

    @staticmethod
    def _build_hover_reference(current_state: QuadrotorState) -> Vector:
        """
        Build a hover reference using current position with zero velocity
        and zero acceleration.
        """
        position = current_state.pva[:3]
        zeros = np.zeros(6, dtype=np.float64)
        return np.hstack((position, zeros))
