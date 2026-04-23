from __future__ import annotations

from typing import Any, Protocol

from trajplan.quadrotor.state import QuadrotorState


class CommonBackend(Protocol):
    def apply_control(
        self,
        control: Any,
        current_state: QuadrotorState,
        dt: float,
    ) -> QuadrotorState | None: ...

    def stop(self) -> None: ...
