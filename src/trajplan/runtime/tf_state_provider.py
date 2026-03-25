from __future__ import annotations

import numpy as np

from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.state_provider import StateProvider


class TfStateProvider(StateProvider):
    """
    State provider that reads drone position from TF2 transforms.

    Used with the Crazyswarm2 simulation backend (``backend:=sim``), which
    publishes TF2 transforms instead of ``/cfX/pose`` PoseStamped topics.
    Can also be used with real hardware if TF2 is being broadcast.

    The sim server broadcasts: world → {cf_name} via TransformBroadcaster
    (see crazyflie_sim/visualization/rviz.py).

    Velocity and acceleration are estimated online using finite differences
    with exponential moving-average smoothing, identical to CfStateProvider.

    Parameters
    ----------
    node
        An rclpy Node used to create the TF listener and timer.
        Use ``cf.node`` from the ``crazyflie_py.Crazyflie`` object.
    cf_name
        Crazyflie frame name as broadcast in TF2 (e.g. "cf1").
    reference_frame
        Parent TF frame (default: "world").
    ema_alpha
        EMA smoothing factor (0 < alpha <= 1). Smaller = more smoothing.
    """

    def __init__(
        self,
        node,
        cf_name: str,
        reference_frame: str = "world",
        ema_alpha: float = 0.3,
    ) -> None:
        try:
            from tf2_ros import Buffer, TransformListener
            from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
        except ImportError as exc:
            raise ImportError(
                "tf2_ros is not available. "
                "Make sure the ROS2 workspace is sourced."
            ) from exc

        self._node = node
        self._cf_name = cf_name
        self._reference_frame = reference_frame
        self._ema_alpha = float(ema_alpha)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

        # Keep exception types for use in the timer callback
        self._LookupException = LookupException
        self._ConnectivityException = ConnectivityException
        self._ExtrapolationException = ExtrapolationException

        # State initialised to zeros; valid after first successful TF lookup
        self._state: QuadrotorState = QuadrotorState(
            pva=np.zeros(9, dtype=np.float64)
        )
        self._initialized: bool = False

        # Finite-difference bookkeeping
        self._prev_pos: np.ndarray | None = None
        self._prev_vel: np.ndarray = np.zeros(3, dtype=np.float64)
        self._prev_acc: np.ndarray = np.zeros(3, dtype=np.float64)
        self._prev_time: float | None = None  # seconds (node clock)

        # Poll TF at ~100 Hz
        self._timer = node.create_timer(0.01, self._timer_callback)

    # ------------------------------------------------------------------
    # StateProvider interface
    # ------------------------------------------------------------------

    def get_state(self) -> QuadrotorState:
        """Return the latest estimated QuadrotorState (position, velocity, acceleration)."""
        return self._state

    def is_initialized(self) -> bool:
        """Return True once at least one TF lookup has succeeded."""
        return self._initialized

    # ------------------------------------------------------------------
    # Internal timer callback
    # ------------------------------------------------------------------

    def _timer_callback(self) -> None:
        try:
            # Use Time() (= latest available) to avoid clock sync issues
            from rclpy.time import Time
            transform = self._tf_buffer.lookup_transform(
                self._reference_frame,
                self._cf_name,
                Time(),
            )
        except (
            self._LookupException,
            self._ConnectivityException,
            self._ExtrapolationException,
        ):
            # TF not yet available — keep previous state
            return

        t = transform.header.stamp
        now = t.sec + t.nanosec * 1e-9

        tr = transform.transform.translation
        pos = np.array([tr.x, tr.y, tr.z], dtype=np.float64)

        if self._prev_pos is None or self._prev_time is None:
            self._prev_pos = pos.copy()
            self._prev_time = now
            self._state = QuadrotorState(
                pva=np.hstack((pos, np.zeros(6, dtype=np.float64)))
            )
            self._initialized = True
            return

        dt = now - self._prev_time
        if dt <= 1e-9:
            return

        alpha = self._ema_alpha

        vel_raw = (pos - self._prev_pos) / dt
        vel_filtered = alpha * vel_raw + (1.0 - alpha) * self._prev_vel

        acc_raw = (vel_filtered - self._prev_vel) / dt
        acc_filtered = alpha * acc_raw + (1.0 - alpha) * self._prev_acc

        self._state = QuadrotorState(
            pva=np.hstack((pos, vel_filtered, acc_filtered))
        )

        self._prev_pos = pos.copy()
        self._prev_vel = vel_filtered.copy()
        self._prev_acc = acc_filtered.copy()
        self._prev_time = now
