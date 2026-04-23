from __future__ import annotations

from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.state_provider import StateProvider


class SimStateProvider(StateProvider):
    def __init__(self) -> None:
        self._state: QuadrotorState | None = None

    def is_initialized(self) -> bool:
        return self._state is not None

    def get_state(self) -> QuadrotorState:
        if self._state is None:
            raise RuntimeError("SimStateProvider is not initialized.")
        return self._state.copy()

    def set_state(self, state: QuadrotorState) -> None:
        self._state = state.copy()
