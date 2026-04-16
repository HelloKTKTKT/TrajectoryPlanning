from __future__ import annotations
from typing import Protocol

from trajplan.quadrotor.state import QuadrotorState


class StateProvider(Protocol):
    def is_initialized(self) -> bool: ...

    def get_state(self) -> QuadrotorState: ...
