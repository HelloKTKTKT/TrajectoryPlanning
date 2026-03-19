from __future__ import annotations


class Backend:
    def apply_control(
        self,
        control,
        dt: float | None = None,
    ) -> None:
        raise NotImplementedError
