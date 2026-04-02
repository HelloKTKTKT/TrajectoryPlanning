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
        landing_height: float = 0.04,
        landing_duration: float = 2.0,
    ) -> None:
        self._cf = cf
        self._default_yaw = float(default_yaw)
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

    def _calculate_omega(
        self,
        acc: np.ndarray,
        jerk: np.ndarray,
    ) -> np.ndarray:
        thrust = acc + np.array([0.0, 0.0, 9.81], dtype=np.float64)
        thrust_norm = np.linalg.norm(thrust)
        if thrust_norm < 1e-6:
            return np.zeros(3, dtype=np.float64)

        z_body = thrust / thrust_norm
        yaw = self._default_yaw
        x_world = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)

        y_body_unnorm = np.cross(z_body, x_world)
        y_body_norm = np.linalg.norm(y_body_unnorm)
        if y_body_norm < 1e-6:
            y_body = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            y_body = y_body_unnorm / y_body_norm

        x_body = np.cross(y_body, z_body)

        jerk_orth = jerk - np.dot(jerk, z_body) * z_body
        h_w = jerk_orth / thrust_norm

        return np.array(
            [-np.dot(h_w, y_body), np.dot(h_w, x_body), 0.0],
            dtype=np.float64,
        )

    def send_reference_pvaj(
        self,
        pvaj: Vector,
    ) -> None:
        pvaj = np.asarray(pvaj, dtype=np.float64).reshape(-1)
        if pvaj.shape != (12,):
            raise ValueError(f"Expected pvaj shape (12,), but got {pvaj.shape}.")

        pos = pvaj[0:3]
        vel = pvaj[3:6]
        acc = pvaj[6:9]
        jerk = pvaj[9:12]

        omega = self._calculate_omega(acc, jerk)

        self._cf.cmdFullState(
            pos,
            vel,
            acc,
            self._default_yaw,
            omega,
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
            targetHeight=(
                self._landing_height if target_height is None else float(target_height)
            ),
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
            print("[CrazyflieBackend] emergency_land() failed " f"(ignored): {exc}")
