from __future__ import annotations

import numpy as np

from trajplan.shared_types import Matrix, Vector
from trajplan.utils import as_vector, clamp


class OneSegmentMinimumJerkTrajectory:
    """
    One-segment 3D quintic polynomial trajectory.

    Boundary conditions
    -------------------
    - position at start / end
    - velocity at start / end
    - acceleration at start / end

    This is the practical trajectory primitive currently needed for local
    trajectory initialization before converting to a uniform cubic B-spline.
    """

    def __init__(
        self,
        start_position: Vector,
        end_position: Vector,
        start_velocity: Vector,
        end_velocity: Vector,
        start_acceleration: Vector,
        end_acceleration: Vector,
        duration: float,
    ) -> None:
        self.start_position = as_vector(start_position)
        self.end_position = as_vector(end_position)
        self.start_velocity = as_vector(start_velocity)
        self.end_velocity = as_vector(end_velocity)
        self.start_acceleration = as_vector(start_acceleration)
        self.end_acceleration = as_vector(end_acceleration)
        self.duration = float(duration)

        for name, vec in (
            ("start_position", self.start_position),
            ("end_position", self.end_position),
            ("start_velocity", self.start_velocity),
            ("end_velocity", self.end_velocity),
            ("start_acceleration", self.start_acceleration),
            ("end_acceleration", self.end_acceleration),
        ):
            if vec.shape != (3,):
                raise ValueError(
                    f"Expected {name} to have shape (3,), but got {vec.shape}."
                )

        if self.duration <= 0.0:
            raise ValueError(f"Expected duration > 0, but got {self.duration}.")

        self.coefficients = self._solve_coefficients()

    def _solve_coefficients(self) -> Matrix:
        """
        Solve quintic coefficients for each spatial axis.

        Polynomial form
        ---------------
        p(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5
        """
        T = self.duration

        a_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [1.0, T, T**2, T**3, T**4, T**5],
                [0.0, 1.0, 2.0 * T, 3.0 * T**2, 4.0 * T**3, 5.0 * T**4],
                [0.0, 0.0, 2.0, 6.0 * T, 12.0 * T**2, 20.0 * T**3],
            ],
            dtype=np.float64,
        )

        b_matrix = np.vstack(
            (
                self.start_position,
                self.start_velocity,
                self.start_acceleration,
                self.end_position,
                self.end_velocity,
                self.end_acceleration,
            )
        )

        # Solve for each axis together. Result shape: (6, 3)
        coefficients = np.linalg.solve(a_matrix, b_matrix)
        return coefficients

    def _clamp_time(self, t: float) -> float:
        return clamp(float(t), 0.0, self.duration)

    def evaluate_position(self, t: float) -> Vector:
        t = self._clamp_time(t)
        basis = np.array([1.0, t, t**2, t**3, t**4, t**5], dtype=np.float64)
        return (basis @ self.coefficients).reshape(-1)

    def evaluate_velocity(self, t: float) -> Vector:
        t = self._clamp_time(t)
        basis = np.array(
            [0.0, 1.0, 2.0 * t, 3.0 * t**2, 4.0 * t**3, 5.0 * t**4],
            dtype=np.float64,
        )
        return (basis @ self.coefficients).reshape(-1)

    def evaluate_acceleration(self, t: float) -> Vector:
        t = self._clamp_time(t)
        basis = np.array(
            [0.0, 0.0, 2.0, 6.0 * t, 12.0 * t**2, 20.0 * t**3],
            dtype=np.float64,
        )
        return (basis @ self.coefficients).reshape(-1)

    def evaluate_pva(self, t: float) -> Vector:
        return np.concatenate(
            (
                self.evaluate_position(t),
                self.evaluate_velocity(t),
                self.evaluate_acceleration(t),
            ),
            axis=0,
        )
