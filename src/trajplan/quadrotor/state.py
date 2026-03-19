from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trajplan.shared_types import Vector


@dataclass(slots=True)
class QuadrotorState:
    pva: Vector = field(default_factory=lambda: np.zeros(9, dtype=np.float64))

    @property
    def position(self) -> Vector:
        return self.pva[:3]

    @property
    def velocity(self) -> Vector:
        return self.pva[3:6]

    @property
    def acceleration(self) -> Vector:
        return self.pva[6:9]
