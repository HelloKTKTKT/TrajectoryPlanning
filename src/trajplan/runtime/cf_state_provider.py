from __future__ import annotations

import numpy as np

from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.state_provider import StateProvider


class CfStateProvider(StateProvider):
    """
    State provider for a real Crazyflie using external motion capture (MoCap).

    The MoCap system publishes pose (position + orientation) via crazyswarm2 on
    the /{cf_name}/pose topic (geometry_msgs/PoseStamped).  Velocity and
    acceleration are not published directly, so they are estimated online using
    finite differences with exponential moving-average (EMA) smoothing.

    Finite-difference + EMA scheme
    --------------------------------
    vel_raw      = (pos_new - pos_prev) / dt
    vel_filtered = alpha * vel_raw  + (1 - alpha) * vel_prev

    acc_raw      = (vel_filtered - vel_prev_filtered) / dt
    acc_filtered = alpha * acc_raw  + (1 - alpha) * acc_prev

    Parameters
    ----------
    node
        An rclpy Node used to create the subscription.
    cf_name
        Crazyflie name as configured in crazyswarm2 (e.g. "cf1").
    ema_alpha
        Smoothing factor for the EMA filter (0 < alpha <= 1).
        Smaller values give heavier smoothing; larger values track faster changes.
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
                "geometry_msgs is not available. "
                "Make sure the ROS2 workspace is sourced."
            ) from exc

        self._node = node
        self._ema_alpha = float(ema_alpha)

        # Last-known state (initialised to zeros; valid flag set on first message)
        self._state: QuadrotorState = QuadrotorState(
            pva=np.zeros(9, dtype=np.float64)
        )
        self._initialized: bool = False

        # Bookkeeping for finite-difference estimation
        self._prev_pos: np.ndarray | None = None
        self._prev_vel: np.ndarray = np.zeros(3, dtype=np.float64)
        self._prev_acc: np.ndarray = np.zeros(3, dtype=np.float64)
        self._prev_stamp: float | None = None  # seconds

        self._sub = node.create_subscription(
            PoseStamped,
            f"/{cf_name}/pose",
            self._pose_callback,
            10,
        )

    # ------------------------------------------------------------------
    # StateProvider interface
    # ------------------------------------------------------------------

    def get_state(self) -> QuadrotorState:
        """Return the latest estimated QuadrotorState (position, velocity, acceleration)."""
        return self._state

    def is_initialized(self) -> bool:
        """Return True once at least one pose message has been received."""
        return self._initialized

    # ------------------------------------------------------------------
    # Internal callback
    # ------------------------------------------------------------------

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
            # First message: cannot compute derivatives yet.
            self._prev_pos = pos.copy()
            self._prev_stamp = stamp_sec
            self._state = QuadrotorState(
                pva=np.hstack((pos, np.zeros(6, dtype=np.float64)))
            )
            self._initialized = True
            return

        dt = stamp_sec - self._prev_stamp
        if dt <= 1e-9:
            # Duplicate or out-of-order message; skip.
            return

        alpha = self._ema_alpha

        # --- velocity estimate ---
        vel_raw = (pos - self._prev_pos) / dt
        vel_filtered = alpha * vel_raw + (1.0 - alpha) * self._prev_vel

        # --- acceleration estimate ---
        acc_raw = (vel_filtered - self._prev_vel) / dt
        acc_filtered = alpha * acc_raw + (1.0 - alpha) * self._prev_acc

        pva = np.hstack((pos, vel_filtered, acc_filtered))
        self._state = QuadrotorState(pva=pva)

        # Update bookkeeping
        self._prev_pos = pos.copy()
        self._prev_vel = vel_filtered.copy()
        self._prev_acc = acc_filtered.copy()
        self._prev_stamp = stamp_sec
