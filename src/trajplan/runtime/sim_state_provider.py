from __future__ import annotations

import numpy as np

from trajplan.quadrotor.state import QuadrotorState


class SimStateProvider:
    """
    Minimal simulation state provider.

    It stores the current simulated quadrotor state and returns a copy when queried.
    """

    def __init__(
        self,
        initial_state: QuadrotorState | None = None,
    ) -> None:
        if initial_state is None:
            initial_state = QuadrotorState()

        self._state = QuadrotorState(
            pva=np.asarray(initial_state.pva, dtype=np.float64).copy()
        )

    def get_state(self) -> QuadrotorState:
        return QuadrotorState(pva=self._state.pva.copy())

    def set_state(self, state: QuadrotorState) -> None:
        self._state = QuadrotorState(pva=np.asarray(state.pva, dtype=np.float64).copy())
