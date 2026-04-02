from __future__ import annotations

import numpy as np
from trajplan.shared_types import Matrix, Vector
from trajplan.utils import as_matrix, clamp


class UniformBSpline:
    """
    Uniform cubic B-spline in 3D.

    Design assumptions
    ------------------
    - fixed dimension: 3
    - fixed degree: 3
    - fixed uniform time interval

    Array shape convention
    ----------------------
    - control_points: shape (N, 3)
    - point_set: shape (M, 3)
    - boundary_derivatives: shape (4, 3)
      ordered as [start_velocity, end_velocity, start_acceleration, end_acceleration]
    - position / velocity / acceleration / jerk: shape (3,)
    - pva: shape (9,)
    - pvaj: shape (12,)
    """

    # Fixed cubic uniform B-spline evaluation matrices.
    P_MATRIX = (
        np.array(
            [
                [1.0, 4.0, 1.0, 0.0],
                [-3.0, 0.0, 3.0, 0.0],
                [3.0, -6.0, 3.0, 0.0],
                [-1.0, 3.0, -3.0, 1.0],
            ],
            dtype=np.float64,
        )
        / 6.0
    )

    V_MATRIX_BASE = np.array(
        [
            [-0.5, 0.0, 0.5, 0.0],
            [1.0, -2.0, 1.0, 0.0],
            [-0.5, 1.5, -1.5, 0.5],
        ],
        dtype=np.float64,
    )

    A_MATRIX_BASE = np.array(
        [
            [1.0, -2.0, 1.0, 0.0],
            [-1.0, 3.0, -3.0, 1.0],
        ],
        dtype=np.float64,
    )

    J_MATRIX_BASE = np.array(
        [[-1.0, 3.0, -3.0, 1.0]],
        dtype=np.float64,
    )

    def __init__(self, control_points: Matrix, delta_t: float) -> None:
        """
        Initialize a uniform cubic B-spline.

        Parameters
        ----------
        control_points
            Control point array with shape (N, 3).
        delta_t
            Uniform time interval between consecutive spline segments.
        """
        control_points = as_matrix(control_points)
        delta_t = float(delta_t)

        if control_points.shape[1] != 3:
            raise ValueError(
                f"Expected control_points to have shape (N, 3), but got {control_points.shape}."
            )

        if control_points.shape[0] < 4:
            raise ValueError(
                f"A cubic B-spline needs at least 4 control points, but got {control_points.shape[0]}."
            )

        if delta_t <= 0.0:
            raise ValueError(f"Expected delta_t > 0, but got {delta_t}.")

        self.control_points: Matrix = control_points
        self.delta_t: float = delta_t

    @classmethod
    def from_point_set(
        cls,
        point_set: Matrix,
        delta_t: float,
        boundary_derivatives: Matrix,
    ) -> UniformBSpline:
        """
        Construct a spline from point constraints and boundary derivatives.

        Parameters
        ----------
        point_set
            Point constraint array with shape (M, 3).
        delta_t
            Uniform time interval.
        boundary_derivatives
            Array with shape (4, 3), ordered as
            [start_velocity, end_velocity, start_acceleration, end_acceleration].

        Returns
        -------
        UniformBSpline
            A new spline object constructed from the solved control points.

        Important design note
        ---------------------
        This function intentionally follows your draft design, not the more
        generic over-constrained formulation.

        If there are M visible waypoint constraints, then this implementation
        introduces one hidden segment near the start and one hidden segment near
        the end. As a result

        - number of segments = M + 1
        - number of control points = (M + 1) + 3 = M + 4

        The constraint count is also M + 4

        - M position constraints from the visible waypoint set
        - 2 boundary velocity constraints
        - 2 boundary acceleration constraints

        Therefore, the linear system becomes square and typically has a unique
        solution, instead of requiring an approximate projection / mapping step.

        The trade-off is that the first and last hidden segment are not directly
        constrained by visible waypoints. In practice this is acceptable when
        delta_t is chosen sufficiently small.
        """
        point_set = as_matrix(point_set)
        boundary_derivatives = as_matrix(boundary_derivatives)
        delta_t = float(delta_t)

        if point_set.shape[1] != 3:
            raise ValueError(
                f"Expected point_set to have shape (M, 3), but got {point_set.shape}."
            )

        if point_set.shape[0] < 2:
            raise ValueError(
                f"Expected at least 2 point constraints, but got {point_set.shape[0]}."
            )

        if boundary_derivatives.shape != (4, 3):
            raise ValueError(
                "Expected boundary_derivatives to have shape (4, 3), ordered as "
                "[start_velocity, end_velocity, start_acceleration, end_acceleration]."
            )

        if delta_t <= 0.0:
            raise ValueError(f"Expected delta_t > 0, but got {delta_t}.")

        waypoint_count = point_set.shape[0]

        # Hidden-segment construction.
        segment_count = waypoint_count + 1
        control_point_count = segment_count + 3

        b_matrix = np.vstack((point_set, boundary_derivatives))
        a_matrix = np.zeros(
            (control_point_count, control_point_count), dtype=np.float64
        )

        p_row = np.array([1.0, 4.0, 1.0], dtype=np.float64) / 6.0
        v_row = np.array([-1.0, 0.0, 1.0], dtype=np.float64) / (2.0 * delta_t)
        a_row = np.array([1.0, -2.0, 1.0], dtype=np.float64) / (delta_t**2)

        # Position constraints for visible waypoints.
        # Because of the hidden first and last segment, the visible waypoint set
        # is not aligned with a plain "one waypoint per segment end" interpretation.
        a_matrix[0, :3] = p_row
        for i in range(2, segment_count - 1):
            a_matrix[i - 1, i : i + 3] = p_row
        a_matrix[segment_count - 2, -3:] = p_row

        # Boundary velocity constraints.
        a_matrix[waypoint_count, :3] = v_row
        a_matrix[waypoint_count + 1, -3:] = v_row

        # Boundary acceleration constraints.
        a_matrix[waypoint_count + 2, :3] = a_row
        a_matrix[waypoint_count + 3, -3:] = a_row

        q_matrix, r_matrix = np.linalg.qr(a_matrix)
        control_points = np.linalg.solve(r_matrix, q_matrix.T @ b_matrix)

        return cls(control_points=control_points, delta_t=delta_t)

    @property
    def num_control_points(self) -> int:
        """Return the number of control points."""
        return int(self.control_points.shape[0])

    @property
    def num_segments(self) -> int:
        """Return the number of spline segments."""
        return self.num_control_points - 3

    @property
    def duration(self) -> float:
        """Return the physical duration of the spline."""
        return self.num_segments * self.delta_t

    def copy(self) -> UniformBSpline:
        """Return a copied spline object."""
        return UniformBSpline(
            control_points=self.control_points.copy(),
            delta_t=self.delta_t,
        )

    def evaluate_pva(self, t: float) -> Vector:
        """
        Evaluate and concatenate position, velocity, acceleration.

        Returns
        -------
        Vector
            Concatenated vector [p, v, a] with shape (9,).
        """
        segment_index, u = self._get_segment_index_and_u(t)
        local_cp = self.control_points[segment_index : segment_index + 4]

        basis_p = np.array([[1.0, u, u**2, u**3]], dtype=np.float64)
        basis_v = np.array([[1.0, u, u**2]], dtype=np.float64)
        basis_a = np.array([[1.0, u]], dtype=np.float64)

        position = basis_p @ self.P_MATRIX @ local_cp
        velocity = basis_v @ (self.V_MATRIX_BASE / self.delta_t) @ local_cp
        acceleration = basis_a @ (self.A_MATRIX_BASE / (self.delta_t**2)) @ local_cp

        return np.hstack((position, velocity, acceleration)).reshape(-1)

    def evaluate_pvaj(self, t: float) -> Vector:
        """
        Evaluate and concatenate position, velocity, acceleration, jerk.

        Returns
        -------
        Vector
            Concatenated vector [p, v, a, j] with shape (12,).
        """
        segment_index, u = self._get_segment_index_and_u(t)
        local_cp = self.control_points[segment_index : segment_index + 4]

        basis_p = np.array([[1.0, u, u**2, u**3]], dtype=np.float64)
        basis_v = np.array([[1.0, u, u**2]], dtype=np.float64)
        basis_a = np.array([[1.0, u]], dtype=np.float64)
        basis_j = np.array([[1.0]], dtype=np.float64)

        position = basis_p @ self.P_MATRIX @ local_cp
        velocity = basis_v @ (self.V_MATRIX_BASE / self.delta_t) @ local_cp
        acceleration = basis_a @ (self.A_MATRIX_BASE / (self.delta_t**2)) @ local_cp
        jerk = basis_j @ (self.J_MATRIX_BASE / (self.delta_t**3)) @ local_cp

        return np.hstack((position, velocity, acceleration, jerk)).reshape(-1)

    def check_feasibility(
        self,
        max_velocity: float,
        max_acceleration: float,
        tolerance: float = 0.0,
    ) -> tuple[bool, float]:
        """
        Check velocity and acceleration feasibility using derivative control points.

        Parameters
        ----------
        max_velocity
            Per-axis velocity limit.
        max_acceleration
            Per-axis acceleration limit.
        tolerance
            Relative feasibility tolerance.

        Returns
        -------
        tuple[bool, float]
            feasible_flag, required_ratio

        Notes
        -----
        The returned ratio is the suggested time-stretch ratio.
        If ratio > 1, the spline should be lengthened in time.
        """
        max_velocity = float(max_velocity)
        max_acceleration = float(max_acceleration)
        tolerance = float(tolerance)

        if max_velocity <= 0.0:
            raise ValueError(f"Expected max_velocity > 0, but got {max_velocity}.")
        if max_acceleration <= 0.0:
            raise ValueError(
                f"Expected max_acceleration > 0, but got {max_acceleration}."
            )
        if tolerance < 0.0:
            raise ValueError(f"Expected tolerance >= 0, but got {tolerance}.")

        control_points_num = self.control_points.shape[0]
        feasible_flag = True

        enlarged_velocity_limit = max_velocity * (1.0 + tolerance) + 1e-4
        enlarged_acceleration_limit = max_acceleration * (1.0 + tolerance) + 1e-4

        max_abs_velocity = 0.0
        for i in range(control_points_num - 1):
            velocity_cp = (
                self.control_points[i + 1] - self.control_points[i]
            ) / self.delta_t
            max_abs_velocity = max(max_abs_velocity, float(np.max(np.abs(velocity_cp))))

            if np.any(np.abs(velocity_cp) > enlarged_velocity_limit):
                feasible_flag = False

        max_abs_acceleration = 0.0
        for i in range(control_points_num - 2):
            acceleration_cp = (
                self.control_points[i + 2]
                - 2.0 * self.control_points[i + 1]
                + self.control_points[i]
            ) / (self.delta_t**2)

            max_abs_acceleration = max(
                max_abs_acceleration,
                float(np.max(np.abs(acceleration_cp))),
            )

            if np.any(np.abs(acceleration_cp) > enlarged_acceleration_limit):
                feasible_flag = False

        ratio_velocity = max_abs_velocity / max_velocity
        ratio_acceleration = np.sqrt(max_abs_acceleration / max_acceleration)
        required_ratio = max(ratio_velocity, ratio_acceleration, 1.0)

        return feasible_flag, float(required_ratio)

    def _get_segment_index_and_u(self, t: float) -> tuple[int, float]:
        """
        Convert physical time t into segment index and local normalized time u.

        Returns
        -------
        tuple[int, float]
            segment_index, u
            where segment_index is in [0, num_segments - 1]
            and u is in [0, 1].
        """
        t = clamp(t, 0.0, self.duration)

        if np.isclose(t, self.duration):
            return self.num_segments - 1, 1.0

        quotient, remainder = divmod(t, self.delta_t)
        segment_index = int(quotient)
        u = remainder / self.delta_t

        return segment_index, float(u)
