from __future__ import annotations

from trajplan.shared_types import Vector


class CommonBackend:
    def send_reference_pva(
        self,
        pva: Vector,
    ) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError
