from __future__ import annotations

import threading

import numpy as np

from trajplan.runtime.backend import Backend
from trajplan.shared_types import Vector


class CfBackend(Backend):
    """
    Backend adapter for a real Crazyflie drone.

    Converts a reference pva (9D: position, velocity, acceleration) into a
    ``cmdFullState`` call on the provided ``crazyflie_py.Crazyflie`` object.
    The Crazyflie's onboard Mellinger controller handles thrust allocation from
    the full-state reference — no additional PC-side controller is needed.

    Yaw and angular-rate are fixed to zero because the TrajectoryPlanning
    pipeline generates 3-D translational trajectories without a yaw component.
    Override ``default_yaw`` / ``default_omega`` if yaw tracking is required.

    Parameters
    ----------
    cf
        A ``crazyflie_py.crazyflie.Crazyflie`` instance (already initialised
        and armed via ``Crazyswarm``).
    default_yaw
        Yaw angle [rad] sent with every setpoint (default: 0.0).
    default_omega
        Body-frame angular velocity [rad/s] sent with every setpoint
        (default: zeros).
    """

    def __init__(
        self,
        cf,
        default_yaw: float = 0.0,
        default_omega: Vector | None = None,
    ) -> None:
        self._cf = cf
        self._default_yaw = float(default_yaw)
        self._default_omega: np.ndarray = (
            np.zeros(3, dtype=np.float64)
            if default_omega is None
            else np.asarray(default_omega, dtype=np.float64).reshape(3)
        )
        self._land_event = threading.Event()

    # ------------------------------------------------------------------
    # Emergency landing
    # ------------------------------------------------------------------

    @property
    def emergency_triggered(self) -> bool:
        """True once emergency_land() has been called."""
        return self._land_event.is_set()

    def emergency_land(self, takeoff_height: float = 0.5) -> None:
        """
        Immediately stop setpoint streaming and command the drone to land.

        Safe to call from any thread.  Sets ``emergency_triggered`` so the
        main loop can detect the condition and stop stepping the agent.
        ``takeoff_height`` is used to set an appropriate landing duration.
        """
        if self._land_event.is_set():
            return  # already triggered — avoid duplicate calls
        self._land_event.set()
        try:
            self._cf.notifySetpointsStop()
            self._cf.land(targetHeight=0.04, duration=takeoff_height + 1.5)
        except Exception as exc:  # noqa: BLE001
            print(f"[CfBackend] emergency_land() command failed (ignored): {exc}")

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def apply_control(self, control, dt: float | None = None) -> None:
        raise NotImplementedError(
            "CfBackend does not support apply_control(). "
            "Use apply_control_direct_set_pva() instead."
        )

    def apply_control_direct_set_pva(self, pva: Vector) -> None:
        """
        Send a full-state setpoint to the Crazyflie.

        Parameters
        ----------
        pva
            9-element array [px, py, pz, vx, vy, vz, ax, ay, az].
        """
        pva = np.asarray(pva, dtype=np.float64).reshape(-1)
        if pva.shape != (9,):
            raise ValueError(
                f"Expected pva shape (9,), but got {pva.shape}."
            )

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
