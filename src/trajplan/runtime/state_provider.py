from __future__ import annotations


class StateProvider:
    def get_state(self):
        raise NotImplementedError
