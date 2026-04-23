from __future__ import annotations

import numpy as np

from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.state_provider import StateProvider


class MocapStateProvider(StateProvider):
    """
    Crazyflie state provider backed by motion-capture pose and onboard odometry.

    The caller is responsible for spinning the ROS node so that callbacks are
    delivered continuously.
    """

    def __init__(
        self,
        node,
        cf_name: str,
        ema_alpha: float = 0.3,
    ) -> None:
        try:
            from geometry_msgs.msg import PoseStamped
            from nav_msgs.msg import Odometry
        except ImportError as exc:
            raise ImportError(
                "geometry_msgs/nav_msgs are not available. "
                "Make sure the ROS2 workspace is sourced."
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
        self._ekf_vel: np.ndarray | None = None

        self._pose_sub = node.create_subscription(
            PoseStamped,
            f"/{self._cf_name}/pose",
            self._pose_callback,
            10,
        )
        self._odom_sub = node.create_subscription(
            Odometry,
            f"/{self._cf_name}/odom",
            self._odom_callback,
            10,
        )

    def is_initialized(self) -> bool:
        return self._initialized

    def get_state(self) -> QuadrotorState:
        if not self._initialized:
            raise RuntimeError("MocapStateProvider is not initialized.")
        return self._state.copy()

    def _odom_callback(self, msg) -> None:
        self._ekf_vel = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            dtype=np.float64,
        )

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

        if self._ekf_vel is not None:
            vel_raw = self._ekf_vel
        else:
            vel_raw = (pos - self._prev_pos) / dt

        vel = alpha * vel_raw + (1.0 - alpha) * self._prev_vel
        acc_raw = (vel - self._prev_vel) / dt
        acc = alpha * acc_raw + (1.0 - alpha) * self._prev_acc

        self._state = QuadrotorState(
            pva=np.hstack((pos, vel, acc)),
            euler=np.zeros(3, dtype=np.float64),
            angular_rate=np.zeros(3, dtype=np.float64),
        )

        self._prev_pos = pos.copy()
        self._prev_vel = vel.copy()
        self._prev_acc = acc.copy()
        self._prev_stamp = stamp_sec
        self._initialized = True
