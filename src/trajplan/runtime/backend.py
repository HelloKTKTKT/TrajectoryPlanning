from __future__ import annotations

from trajplan.shared_types import Vector


class CommonBackend:
    def send_reference_pvaj(
        self,
        pvaj: Vector,
    ) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError
