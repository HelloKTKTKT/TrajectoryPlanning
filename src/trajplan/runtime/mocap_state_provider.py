from __future__ import annotations

import threading

import numpy as np

from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.state_provider import StateProvider


class MocapStateProvider(StateProvider):
    """
    State provider for a real Crazyflie using external motion capture.

    It subscribes to ``/{cf_name}/pose`` (``geometry_msgs/PoseStamped``) and
    estimates velocity and acceleration from position differences with EMA
    smoothing. Orientation and angular rate are currently set to zero because
    the Crazyflie execution path only requires pva.
    """

    def __init__(
        self,
        node,
        cf_name: str,
        ema_alpha: float = 0.3,
    ) -> None:
        try:
            from geometry_msgs.msg import PoseStamped
        except ImportError as exc:
            raise ImportError(
                "geometry_msgs is not available. Make sure the ROS2 workspace is sourced."
            ) from exc

        self._node = node
        self._cf_name = str(cf_name)
        self._ema_alpha = float(ema_alpha)

        if not (0.0 < self._ema_alpha <= 1.0):
            raise ValueError(
                f"Expected ema_alpha in (0, 1], but got {self._ema_alpha}."
            )

        self._state = QuadrotorState()
        self._initialized = False

        self._prev_pos: np.ndarray | None = None
        self._prev_vel = np.zeros(3, dtype=np.float64)
        self._prev_acc = np.zeros(3, dtype=np.float64)
        self._prev_stamp: float | None = None

        self._lock = threading.Lock()
        self._sub = node.create_subscription(
            PoseStamped,
            f"/{self._cf_name}/pose",
            self._pose_callback,
            10,
        )

    def get_state(self) -> QuadrotorState | None:
        if not self._initialized:
            return None

        with self._lock:
            return self._state.copy()

    def is_initialized(self) -> bool:
        return self._initialized

    def _pose_callback(self, msg) -> None:
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64,
        )

        with self._lock:
            if self._prev_pos is None or self._prev_stamp is None:
                self._prev_pos = pos.copy()
                self._prev_stamp = stamp_sec
                self._state = QuadrotorState(
                    pva=np.hstack((pos, np.zeros(6, dtype=np.float64))),
                    euler=np.zeros(3, dtype=np.float64),
                    angular_rate=np.zeros(3, dtype=np.float64),
                )
                self._initialized = True
                return

            dt = stamp_sec - self._prev_stamp
            if dt <= 1e-9:
                return

            alpha = self._ema_alpha

            vel_raw = (pos - self._prev_pos) / dt
            vel_filtered = alpha * vel_raw + (1.0 - alpha) * self._prev_vel

            acc_raw = (vel_filtered - self._prev_vel) / dt
            acc_filtered = alpha * acc_raw + (1.0 - alpha) * self._prev_acc

            self._state = QuadrotorState(
                pva=np.hstack((pos, vel_filtered, acc_filtered)),
                euler=np.zeros(3, dtype=np.float64),
                angular_rate=np.zeros(3, dtype=np.float64),
            )

            self._prev_pos = pos.copy()
            self._prev_vel = vel_filtered.copy()
            self._prev_acc = acc_filtered.copy()
            self._prev_stamp = stamp_sec
            self._initialized = True
