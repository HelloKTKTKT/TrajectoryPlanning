from __future__ import annotations

from dataclasses import dataclass

# import numpy as np

from trajplan.shared_types import Matrix, Vector
from trajplan.trajectory.bspline import UniformBSpline


@dataclass(slots=True)
class BsplineOptimizerConfig:
    """
    Configuration for collision-aware B-spline optimization.
    """

    max_iteration_num: int = 50
    max_restart_num: int = 5

    lambda_smooth: float = 1.0
    lambda_collision: float = 1.0
    lambda_feasibility: float = 1.0
    lambda_endpoint: float = 1.0

    collision_weight_increase_factor: float = 2.0

    def __post_init__(self) -> None:
        if self.max_iteration_num <= 0:
            raise ValueError(
                f"Expected max_iteration_num > 0, but got {self.max_iteration_num}."
            )
        if self.max_restart_num <= 0:
            raise ValueError(
                f"Expected max_restart_num > 0, but got {self.max_restart_num}."
            )
        if self.lambda_smooth < 0.0:
            raise ValueError(
                f"Expected lambda_smooth >= 0, but got {self.lambda_smooth}."
            )
        if self.lambda_collision < 0.0:
            raise ValueError(
                f"Expected lambda_collision >= 0, but got {self.lambda_collision}."
            )
        if self.lambda_feasibility < 0.0:
            raise ValueError(
                f"Expected lambda_feasibility >= 0, but got {self.lambda_feasibility}."
            )
        if self.lambda_endpoint < 0.0:
            raise ValueError(
                f"Expected lambda_endpoint >= 0, but got {self.lambda_endpoint}."
            )
        if self.collision_weight_increase_factor <= 1.0:
            raise ValueError(
                "Expected collision_weight_increase_factor > 1, "
                f"but got {self.collision_weight_increase_factor}."
            )


@dataclass(slots=True)
class BsplineOptimizationResult:
    """
    Result of one B-spline optimization call.
    """

    success: bool
    optimized_bspline: UniformBSpline | None
    final_cost: float
    message: str


class BsplineOptimizer:
    """
    Collision-aware B-spline optimizer.

    Current role
    ------------
    This class will later hold the real optimization logic, including
    cost construction, gradients, and restart strategy.

    At the current skeleton stage, the full API is defined, but the
    internals are still placeholders.
    """

    def __init__(
        self,
        config: BsplineOptimizerConfig | None = None,
    ) -> None:
        self.config = BsplineOptimizerConfig() if config is None else config

    def optimize(
        self,
        initial_bspline: UniformBSpline,
        pv_pair_info: list[dict],
        target_endpoint_pva: Vector,
    ) -> BsplineOptimizationResult:
        """
        Optimize a B-spline using collision guidance and endpoint targets.

        Parameters
        ----------
        initial_bspline
            Initial local B-spline before collision-aware optimization.
        pv_pair_info
            Collision guidance information generated from A* paths.
            Current placeholder type is list[dict].
        target_endpoint_pva
            Endpoint target pva of the local trajectory.

        Returns
        -------
        BsplineOptimizationResult
            Optimization result.
        """
        _ = pv_pair_info
        _ = target_endpoint_pva

        # Placeholder: currently return the input spline directly.
        return BsplineOptimizationResult(
            success=True,
            optimized_bspline=initial_bspline.copy(),
            final_cost=0.0,
            message="Optimizer placeholder returned the input spline.",
        )

    # ------------------------------------------------------------------
    # future internal optimization utilities
    # ------------------------------------------------------------------

    def _compute_total_cost(
        self,
        control_points: Matrix,
        pv_pair_info: list[dict],
        target_endpoint_pva: Vector,
    ) -> float:
        raise NotImplementedError

    def _compute_total_gradient(
        self,
        control_points: Matrix,
        pv_pair_info: list[dict],
        target_endpoint_pva: Vector,
    ) -> Matrix:
        raise NotImplementedError

    def _compute_smoothness_cost(self, control_points: Matrix) -> float:
        raise NotImplementedError

    def _compute_collision_cost(
        self,
        control_points: Matrix,
        pv_pair_info: list[dict],
    ) -> float:
        raise NotImplementedError

    def _compute_feasibility_cost(self, control_points: Matrix) -> float:
        raise NotImplementedError

    def _compute_endpoint_cost(
        self,
        control_points: Matrix,
        target_endpoint_pva: Vector,
    ) -> float:
        raise NotImplementedError
