from __future__ import annotations


class Controller:
    def compute_control(
        self,
        current_state,
        reference_pva,
    ):
        raise NotImplementedError
