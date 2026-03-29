from __future__ import annotations

from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.state_provider import StateProvider


class SimStateProvider(StateProvider):
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

        self._state = initial_state.copy()

        self._initialized = True

    def get_state(self) -> QuadrotorState:
        return self._state.copy()

    def set_state(self, state: QuadrotorState) -> None:
        self._state = state.copy()

    def is_initialized(self) -> bool:
        return self._initialized
