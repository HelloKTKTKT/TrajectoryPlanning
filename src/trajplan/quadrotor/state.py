from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trajplan.shared_types import Vector


@dataclass(slots=True)
class QuadrotorState:
    pva: Vector = field(
        default_factory=lambda: np.zeros(9, dtype=np.float64)
    )  # 用field和dafautt_factory来初始化,避免可变默认值共享
    euler: Vector = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    angular_rate: Vector = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        self.pva = np.asarray(self.pva, dtype=np.float64).reshape(-1)
        self.euler = np.asarray(self.euler, dtype=np.float64).reshape(-1)
        self.angular_rate = np.asarray(self.angular_rate, dtype=np.float64).reshape(-1)

        if self.pva.shape != (9,):
            raise ValueError(f"Expected pva shape (9,), but got {self.pva.shape}.")
        if self.euler.shape != (3,):
            raise ValueError(f"Expected euler shape (3,), but got {self.euler.shape}.")
        if self.angular_rate.shape != (3,):
            raise ValueError(
                f"Expected angular_rate shape (3,), but got {self.angular_rate.shape}."
            )

    @property
    def position(self) -> Vector:
        return self.pva[:3]

    @property
    def velocity(self) -> Vector:
        return self.pva[3:6]

    @property
    def acceleration(self) -> Vector:
        return self.pva[6:9]

    @property
    def roll_pitch_yaw(self) -> Vector:
        return self.euler

    @property
    def body_rate(self) -> Vector:
        return self.angular_rate

    def copy(self) -> QuadrotorState:
        return QuadrotorState(
            pva=self.pva.copy(),
            euler=self.euler.copy(),
            angular_rate=self.angular_rate.copy(),
        )
