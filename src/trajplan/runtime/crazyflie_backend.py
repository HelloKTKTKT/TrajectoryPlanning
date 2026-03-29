from __future__ import annotations

import threading

import numpy as np

from trajplan.runtime.backend import CommonBackend
from trajplan.shared_types import Vector


class CrazyflieBackend(CommonBackend):
    """
    Backend adapter for a Crazyflie drone driven by full-state references.

    The desktop trajectory stack produces a 9D pva reference:
        [px, py, pz, vx, vy, vz, ax, ay, az]

    This backend forwards that reference to Crazyflie's onboard controller via
    ``cmdFullState``. Yaw and body-rate are currently fixed because the current
    planning stack only generates translational references.
    """

    def __init__(
        self,
        cf,
        default_yaw: float = 0.0,
        default_omega: Vector | None = None,
        landing_height: float = 0.04,
        landing_duration: float = 2.0,
    ) -> None:
        self._cf = cf
        self._default_yaw = float(default_yaw)
        self._default_omega = (
            np.zeros(3, dtype=np.float64)
            if default_omega is None
            else np.asarray(default_omega, dtype=np.float64).reshape(3)
        )
        self._landing_height = float(landing_height)
        self._landing_duration = float(landing_duration)
        self._emergency_land_event = threading.Event()

        if self._landing_height < 0.0:
            raise ValueError(
                f"Expected landing_height >= 0, but got {self._landing_height}."
            )
        if self._landing_duration <= 0.0:
            raise ValueError(
                f"Expected landing_duration > 0, but got {self._landing_duration}."
            )

    @property
    def emergency_triggered(self) -> bool:
        return self._emergency_land_event.is_set()

    def send_reference_pva(
        self,
        pva: Vector,
    ) -> None:
        pva = np.asarray(pva, dtype=np.float64).reshape(-1)
        if pva.shape != (9,):
            raise ValueError(f"Expected pva shape (9,), but got {pva.shape}.")

        pos = pva[0:3]
        vel = pva[3:6]
        acc = pva[6:9]

        self._cf.cmdFullState(
            pos,
            vel,
            acc,
            self._default_yaw,
            self._default_omega,
        )

    def stop(self) -> None:
        self._cf.notifySetpointsStop()

    def arm(
        self,
        enabled: bool,
    ) -> None:
        self._cf.arm(bool(enabled))

    def takeoff(
        self,
        target_height: float,
        duration: float,
    ) -> None:
        self._cf.takeoff(
            targetHeight=float(target_height),
            duration=float(duration),
        )

    def land(
        self,
        target_height: float | None = None,
        duration: float | None = None,
    ) -> None:
        self._cf.notifySetpointsStop()
        self._cf.land(
            targetHeight=self._landing_height if target_height is None else float(target_height),
            duration=self._landing_duration if duration is None else float(duration),
        )

    def emergency_land(
        self,
        target_height: float | None = None,
        duration: float | None = None,
    ) -> None:
        if self._emergency_land_event.is_set():
            return

        self._emergency_land_event.set()
        try:
            self.land(
                target_height=target_height,
                duration=duration,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[CrazyflieBackend] emergency_land() failed "
                f"(ignored): {exc}"
            )
