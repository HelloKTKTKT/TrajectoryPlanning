from __future__ import annotations

import math

import numpy as np

from trajplan.shared_types import Matrix, Vector
from trajplan.utils import as_matrix, as_vector, clamp


class MinicostTraj:
    """
    Piecewise minimum-cost polynomial trajectory in 3D.

    Supported costs
    ---------------
    - s = 3: minimum jerk, polynomial degree = 5
    - s = 4: minimum snap, polynomial degree = 7

    Notes
    -----
    - `waypoints` has shape (N, 3), where N >= 2
    - `time_idx` has shape (N,), strictly ascending in local trajectory time
    - `ho_start` / `ho_end` store higher-order boundary conditions ordered as
      velocity, acceleration, jerk, ...
    - internally the trajectory always starts at local time 0
    - each segment uses polynomial form:
      c_0 + c_1 t + c_2 t^2 + ...
    """

    def __init__(
        self,
        waypoints: Matrix,
        ho_start: Matrix,
        ho_end: Matrix,
        s: int,
        time_idx: Vector,
    ) -> None:
        self.waypoints = as_matrix(waypoints)
        self.s = int(s)
        self.ho_start = as_matrix(ho_start)
        self.ho_end = as_matrix(ho_end)
        self.time_idx = as_vector(time_idx).copy()

        self._feasibility_check_and_normalize()
        self.coefficient_list = self._solve_coefficients()

    @classmethod
    def from_boundary_pva(
        cls,
        start_position: Vector,
        end_position: Vector,
        start_velocity: Vector,
        end_velocity: Vector,
        start_acceleration: Vector,
        end_acceleration: Vector,
        duration: float,
    ) -> MinicostTraj:
        """
        Compatibility constructor for the current one-segment minimum-jerk use.
        """
        duration = float(duration)
        if duration <= 0.0:
            raise ValueError(f"Expected duration > 0, but got {duration}.")

        waypoints = np.vstack(
            (
                as_vector(start_position),
                as_vector(end_position),
            )
        )
        ho_start = np.vstack(
            (
                as_vector(start_velocity),
                as_vector(start_acceleration),
            )
        )
        ho_end = np.vstack(
            (
                as_vector(end_velocity),
                as_vector(end_acceleration),
            )
        )
        time_idx = np.array([0.0, duration], dtype=np.float64)
        return cls(
            waypoints=waypoints,
            ho_start=ho_start,
            ho_end=ho_end,
            s=3,
            time_idx=time_idx,
        )

    def _feasibility_check_and_normalize(self) -> None:
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError(
                f"Expected waypoints to have shape (N, 3), but got {self.waypoints.shape}."
            )
        if self.waypoints.shape[0] < 2:
            raise ValueError("Waypoints not enough, at least 2 are required.")
        if self.s not in (3, 4):
            raise ValueError(f"Only s=3 or s=4 is supported, but got {self.s}.")
        if self.time_idx.ndim != 1:
            raise ValueError(
                f"Expected time_idx to have shape (N,), but got {self.time_idx.shape}."
            )
        if self.time_idx.size != self.waypoints.shape[0]:
            raise ValueError("time_idx does not match the number of waypoints.")
        if np.any(np.diff(self.time_idx) <= 0.0):
            raise ValueError("time_idx must be strictly ascending.")

        self.ho_start = self._normalize_tip_cond(self.ho_start)
        self.ho_end = self._normalize_tip_cond(self.ho_end)
        self.time_idx = self.time_idx - float(self.time_idx[0])

    def _normalize_tip_cond(self, tip_cond: Matrix) -> Matrix:
        if tip_cond.ndim != 2 or tip_cond.shape[1] != 3:
            raise ValueError(
                f"Expected boundary derivatives to have shape (K, 3), but got {tip_cond.shape}."
            )
        required_rows = self.s - 1
        if tip_cond.shape[0] >= required_rows:
            return tip_cond[:required_rows, :].copy()
        padding = np.zeros((required_rows - tip_cond.shape[0], 3), dtype=np.float64)
        return np.vstack((tip_cond, padding))

    def _solve_coefficients(self) -> list[Matrix]:
        segment_num = self.waypoints.shape[0] - 1
        coeff_dim = self.s * 2

        f0 = np.zeros((self.s, coeff_dim), dtype=np.float64)
        f0[:, : self.s] = np.diag(
            [math.factorial(i) for i in range(self.s)]
        ).astype(np.float64)

        t_end = float(self.time_idx[-1] - self.time_idx[-2])
        em = self._poly_matrix(t_end)[: self.s, :]

        row_mid: list[Matrix] = []
        for i in range(1, self.waypoints.shape[0] - 1):
            ti = float(self.time_idx[i] - self.time_idx[i - 1])

            ei = self._poly_matrix(ti)[: coeff_dim - 1, :]
            ei = np.vstack((ei[0, :], ei))

            fi = -self._poly_matrix(0.0)[: coeff_dim - 1, :]
            fi = np.vstack((np.zeros((1, coeff_dim), dtype=np.float64), fi))

            row_i = np.zeros((coeff_dim, segment_num * coeff_dim), dtype=np.float64)
            row_i[:, (i - 1) * coeff_dim : (i + 1) * coeff_dim] = np.hstack((ei, fi))
            row_mid.append(row_i)

        row_0 = np.zeros((f0.shape[0], segment_num * coeff_dim), dtype=np.float64)
        row_0[:, :coeff_dim] = f0

        row_end = np.zeros((em.shape[0], segment_num * coeff_dim), dtype=np.float64)
        row_end[:, -coeff_dim:] = em

        if row_mid:
            m_mat = np.vstack((row_0, np.vstack(row_mid), row_end))
        else:
            m_mat = np.vstack((row_0, row_end))

        rhs_blocks: list[Matrix] = []
        for i in range(self.waypoints.shape[0]):
            if i == 0:
                block = np.vstack((self.waypoints[i : i + 1, :], self.ho_start))
            elif i == self.waypoints.shape[0] - 1:
                block = np.vstack((self.waypoints[i : i + 1, :], self.ho_end))
            else:
                block = np.vstack(
                    (
                        self.waypoints[i : i + 1, :],
                        np.zeros((coeff_dim - 1, 3), dtype=np.float64),
                    )
                )
            rhs_blocks.append(block)
        b_mat = np.vstack(rhs_blocks)

        coefficients = np.asarray(np.linalg.solve(m_mat, b_mat), dtype=np.float64)

        coefficient_list: list[Matrix] = []
        for i in range(segment_num):
            start_row = i * coeff_dim
            end_row = (i + 1) * coeff_dim
            coefficient_list.append(coefficients[start_row:end_row, :])
        return coefficient_list

    def _poly_matrix(self, t: float) -> Matrix:
        coeff_dim = self.s * 2
        e_mat = np.zeros((coeff_dim, coeff_dim), dtype=np.float64)
        for derivative_order in range(coeff_dim):
            for poly_order in range(coeff_dim):
                if poly_order >= derivative_order:
                    coefficient = math.factorial(poly_order) / math.factorial(
                        poly_order - derivative_order
                    )
                    e_mat[derivative_order, poly_order] = coefficient * (
                        float(t) ** (poly_order - derivative_order)
                    )
        return e_mat

    @property
    def degree(self) -> int:
        return 2 * self.s - 1

    @property
    def num_segments(self) -> int:
        return self.waypoints.shape[0] - 1

    @property
    def duration(self) -> float:
        return float(self.time_idx[-1])

    @property
    def start_position(self) -> Vector:
        return self.waypoints[0].copy()

    @property
    def end_position(self) -> Vector:
        return self.waypoints[-1].copy()

    @property
    def start_velocity(self) -> Vector:
        return self.ho_start[0].copy()

    @property
    def end_velocity(self) -> Vector:
        return self.ho_end[0].copy()

    @property
    def start_acceleration(self) -> Vector:
        return self.ho_start[1].copy()

    @property
    def end_acceleration(self) -> Vector:
        return self.ho_end[1].copy()

    def copy(self) -> MinicostTraj:
        return MinicostTraj(
            waypoints=self.waypoints.copy(),
            ho_start=self.ho_start.copy(),
            ho_end=self.ho_end.copy(),
            s=self.s,
            time_idx=self.time_idx.copy(),
        )

    def _clamp_time(self, t: float) -> float:
        return clamp(float(t), 0.0, self.duration)

    def _locate_segment(self, t: float) -> tuple[int, float]:
        t = self._clamp_time(t)
        if t >= self.duration:
            segment_idx = self.num_segments - 1
            return segment_idx, float(t - self.time_idx[segment_idx])

        segment_idx = int(np.searchsorted(self.time_idx[1:], t, side="right"))
        local_time = float(t - self.time_idx[segment_idx])
        return segment_idx, local_time

    def _evaluate_derivative(self, t: float, derivative_order: int) -> Vector:
        segment_idx, local_time = self._locate_segment(t)
        coeff = self.coefficient_list[segment_idx]
        basis = self._poly_matrix(local_time)[derivative_order, :].reshape(1, -1)
        return (basis @ coeff).reshape(-1)

    def evaluate_position(self, t: float) -> Vector:
        return self._evaluate_derivative(t, 0)

    def evaluate_velocity(self, t: float) -> Vector:
        return self._evaluate_derivative(t, 1)

    def evaluate_acceleration(self, t: float) -> Vector:
        return self._evaluate_derivative(t, 2)

    def evaluate_jerk(self, t: float) -> Vector:
        return self._evaluate_derivative(t, 3)

    def evaluate_pva(self, t: float) -> Vector:
        return np.concatenate(
            (
                self.evaluate_position(t),
                self.evaluate_velocity(t),
                self.evaluate_acceleration(t),
            ),
            axis=0,
        )
