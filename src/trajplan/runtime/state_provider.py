from __future__ import annotations
from trajplan.quadrotor.state import QuadrotorState


class StateProvider:
    def get_state(self) -> QuadrotorState | None:
        raise NotImplementedError

    def is_initialized(self) -> bool:
        raise NotImplementedError
