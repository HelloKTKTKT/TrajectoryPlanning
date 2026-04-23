from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, TypedDict, cast

import numpy as np
from scipy.optimize import minimize  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]

from trajplan.planning.messages import TrackingReference
from trajplan.shared_types import Matrix, PvPair, Vector
from trajplan.trajectory.bspline import UniformBSpline


@dataclass(slots=True)
class BsplineOptimizerConfig:
    lambda_smooth: float
    lambda_dist: float
    lambda_feasibility: float
    lambda_fitness: float
    lambda_swarm: float
    lambda_tracking: float
    tol: float
    safe_dist: float
    swarm_clearance: float
    swarm_sample_divisor: int
    tracking_sample_divisor: int
    max_vel: float
    max_acc: float


class _OptimizeResultLike(Protocol):
    x: Vector


class _NeighborTrajectoryLike(Protocol):
    agent_id: int
    start_time: float
    trajectory: UniformBSpline


class _ReboundParam(TypedDict):
    ctps_opt: Matrix
    pv_pair_info: list[list[PvPair]]
    lambda_dist_apply: float
    lambda_swarm_apply: float
    t_ubs: float
    traj_start_time: float
    neighbor_trajectories: Sequence[_NeighborTrajectoryLike]
    tracking_reference: TrackingReference | None


class _RefineParam(TypedDict):
    ctps_opt: Matrix
    t_ubs: float
    ref_point: Matrix


class BsplineOptimizer:
    """
    Collision-aware B-spline optimizer.

    This is the same optimizer structure as the earlier draft, adapted to the
    VodafoneDroneDemoSingle codebase and type definitions.
    """

    def __init__(
        self,
        config: BsplineOptimizerConfig,
    ) -> None:
        self.config = config

    def optimize(
        self,
        initial_bspline: UniformBSpline,
        pv_pair_info: list[list[PvPair]],
        lambda_dist_apply: float,
        lambda_swarm_apply: float,
        traj_start_time: float,
        neighbor_trajectories: Sequence[_NeighborTrajectoryLike] = (),
        tracking_reference: TrackingReference | None = None,
    ) -> UniformBSpline:
        start_index = 3
        end_index = initial_bspline.num_control_points - 3
        if end_index <= start_index:
            return initial_bspline

        control_points_opt = initial_bspline.control_points.copy()
        x0 = control_points_opt[start_index:end_index].flatten()
        param: _ReboundParam = {
            "ctps_opt": control_points_opt,
            "pv_pair_info": pv_pair_info,
            "lambda_dist_apply": lambda_dist_apply,
            "lambda_swarm_apply": lambda_swarm_apply,
            "t_ubs": initial_bspline.delta_t,
            "traj_start_time": float(traj_start_time),
            "neighbor_trajectories": neighbor_trajectories,
            "tracking_reference": (
                None if tracking_reference is None else tracking_reference.copy()
            ),
        }

        opt_res = cast(
            _OptimizeResultLike,
            minimize(
                self._objective_gradient_rebound,
                x0,
                method="L-BFGS-B",
                jac=True,
                args=(param,),
                tol=self.config.tol,
            ),
        )
        initial_bspline.control_points[start_index:end_index] = opt_res.x.reshape(-1, 3)
        return initial_bspline

    def refine_feasibility(
        self,
        optimized_bspline: UniformBSpline,
        ratio: float,
        local_start_pva: Vector,
        local_target_pva: Vector,
    ) -> None:
        reparam_bspline = UniformBSpline(
            control_points=optimized_bspline.control_points.copy(),
            delta_t=optimized_bspline.delta_t * ratio,
        )
        reparam_dt = reparam_bspline.delta_t

        point_set_list: list[Vector] = []
        for t in np.arange(
            0.0,
            reparam_bspline.duration + 1e-3,
            reparam_dt,
            dtype=np.float64,
        ):
            point_set_list.append(reparam_bspline.evaluate_pva(float(t))[:3])
        del point_set_list[1]
        del point_set_list[-2]
        point_set = np.vstack(point_set_list)

        boundary_derivatives = np.vstack(
            (
                local_start_pva[3:6],
                local_target_pva[3:6],
                local_start_pva[6:9],
                local_target_pva[6:9],
            )
        )
        refine_init_bspline = UniformBSpline.from_point_set(
            point_set=point_set,
            delta_t=reparam_dt,
            boundary_derivatives=boundary_derivatives,
        )

        point_set_ref_list: list[Vector] = []
        for t in np.arange(
            0.0,
            refine_init_bspline.duration + 1e-3,
            refine_init_bspline.delta_t,
            dtype=np.float64,
        ):
            point_set_ref_list.append(refine_init_bspline.evaluate_pva(float(t))[:3])
        point_set_ref = np.vstack(point_set_ref_list)

        optimized_bspline.control_points = refine_init_bspline.control_points.copy()
        optimized_bspline.delta_t = refine_init_bspline.delta_t
        param: _RefineParam = {
            "ctps_opt": optimized_bspline.control_points.copy(),
            "t_ubs": optimized_bspline.delta_t,
            "ref_point": point_set_ref,
        }

        start_index = 3
        end_index = param["ctps_opt"].shape[0] - 3
        if end_index <= start_index:
            return

        x0 = param["ctps_opt"][start_index:end_index].flatten()
        opt_res = cast(
            _OptimizeResultLike,
            minimize(
                self._objective_gradient_refine,
                x0,
                method="L-BFGS-B",
                jac=True,
                args=(param,),
                tol=self.config.tol,
            ),
        )
        optimized_bspline.control_points[start_index:end_index] = opt_res.x.reshape(
            -1, 3
        )

    def _objective_gradient_rebound(
        self,
        x: Vector,
        param: _ReboundParam,
    ) -> tuple[float, Vector]:
        x = x.reshape(-1, 3)
        control_points_opt = param["ctps_opt"]
        pv_pair_info = param["pv_pair_info"]
        lambda_dist_apply = param["lambda_dist_apply"]
        lambda_swarm_apply = param["lambda_swarm_apply"]
        traj_start_time = param["traj_start_time"]
        neighbor_trajectories = param["neighbor_trajectories"]
        tracking_reference = param["tracking_reference"]
        x = np.vstack((control_points_opt[:3, :], x, control_points_opt[-3:, :]))
        qnum, _ = x.shape

        delta_t = 1.0
        inv_dt3 = (1.0 / delta_t) ** 3
        inv_dt2 = (1.0 / delta_t) ** 2
        jerk_array = np.array([-1.0, 3.0, -3.0, 1.0], dtype=np.float64)
        grad_smoothness = np.zeros_like(x)
        cost_smoothness = 0.0
        for i in range(qnum - 3):
            jerk_i = inv_dt3 * jerk_array @ x[i : i + 4, :]
            cost_smoothness += jerk_i @ jerk_i
            grad_smoothness[i : i + 4, :] += (
                2.0
                * inv_dt3
                * np.outer(
                    jerk_array,
                    jerk_i,
                )
            )
        delta_t = param["t_ubs"]
        inv_dt3 = (1.0 / delta_t) ** 3
        inv_dt2 = (1.0 / delta_t) ** 2

        grad_feasibility = np.zeros_like(x)
        cost_feasibility = 0.0

        for i in range(qnum - 1):
            vi = (x[i + 1] - x[i]) / delta_t
            for j in range(3):
                if vi[j] > self.config.max_vel:
                    cost_feasibility += (vi[j] - self.config.max_vel) ** 2 * inv_dt2
                    grad_contrib = (
                        2.0 * (vi[j] - self.config.max_vel) / delta_t * inv_dt2
                    )
                elif vi[j] < -self.config.max_vel:
                    cost_feasibility += (vi[j] + self.config.max_vel) ** 2 * inv_dt2
                    grad_contrib = (
                        2.0 * (vi[j] + self.config.max_vel) / delta_t * inv_dt2
                    )
                else:
                    grad_contrib = 0.0
                grad_feasibility[i + 1, j] += grad_contrib
                grad_feasibility[i, j] -= grad_contrib

        for i in range(qnum - 2):
            ai = (x[i] - 2.0 * x[i + 1] + x[i + 2]) * inv_dt2
            for j in range(3):
                if ai[j] > self.config.max_acc:
                    cost_feasibility += (ai[j] - self.config.max_acc) ** 2
                    grad_contrib = 2.0 * (ai[j] - self.config.max_acc) * inv_dt2
                elif ai[j] < -self.config.max_acc:
                    cost_feasibility += (ai[j] + self.config.max_acc) ** 2
                    grad_contrib = 2.0 * (ai[j] + self.config.max_acc) * inv_dt2
                else:
                    grad_contrib = 0.0
                grad_feasibility[i, j] += grad_contrib
                grad_feasibility[i + 1, j] -= 2.0 * grad_contrib
                grad_feasibility[i + 2, j] += grad_contrib

        grad_dist = np.zeros_like(x)
        cost_dist = 0.0
        d_safe = self.config.safe_dist
        for i in range(qnum):
            for pv_pair in pv_pair_info[i]:
                anchor = pv_pair.surface_anchor
                unit_dir = pv_pair.unit_dir
                dist = (x[i] - anchor) @ unit_dir
                c_dist = d_safe - dist
                if c_dist <= 0.0:
                    continue
                if c_dist <= d_safe:
                    cost_dist += c_dist**3
                    grad_dist[i] += -3.0 * c_dist**2 * unit_dir
                else:
                    cost_dist += (
                        3.0 * d_safe * c_dist**2 - 3.0 * d_safe**2 * c_dist + d_safe**3
                    )
                    grad_dist[i] += (
                        -(6.0 * d_safe * c_dist - 3.0 * d_safe**2) * unit_dir
                    )

        grad_swarm = np.zeros_like(x)
        cost_swarm = 0.0
        if len(neighbor_trajectories) > 0:
            order = 3
            end_idx = qnum - order - int((qnum - 2 * order) / 3.0)
            clearance = 2.0 * float(self.config.swarm_clearance)
            ellipsoid_a = 2.0
            ellipsoid_b = 1.0
            inv_a2 = 1.0 / (ellipsoid_a * ellipsoid_a)
            inv_b2 = 1.0 / (ellipsoid_b * ellipsoid_b)

            for i in range(order, end_idx):
                global_sample_time = traj_start_time + (
                    ((order - 1) / 2.0 + (i - order + 1)) * delta_t
                )

                for neighbor in neighbor_trajectories:
                    neighbor_t = global_sample_time - neighbor.start_time
                    neighbor_duration = float(neighbor.trajectory.duration)
                    if neighbor_t < 0.0 or neighbor_t >= neighbor_duration - 0.1:
                        continue

                    neighbor_position = np.asarray(
                        neighbor.trajectory.evaluate_pva(neighbor_t)[:3],
                        dtype=np.float64,
                    ).reshape(3)
                    position_diff = np.asarray(
                        x[i] - neighbor_position,
                        dtype=np.float64,
                    ).reshape(3)
                    ellip_dist = np.sqrt(
                        position_diff[2] * position_diff[2] * inv_a2
                        + (position_diff[0] * position_diff[0] + position_diff[1] * position_diff[1])
                        * inv_b2
                    )
                    dist_err = clearance - ellip_dist
                    if dist_err <= 0.0:
                        continue

                    safe_ellip_dist = max(float(ellip_dist), 1e-6)
                    coeff_xy = -2.0 * (clearance / safe_ellip_dist - 1.0) * inv_b2
                    coeff_z = -2.0 * (clearance / safe_ellip_dist - 1.0) * inv_a2
                    coeff = np.array([coeff_xy, coeff_xy, coeff_z], dtype=np.float64)

                    cost_swarm += dist_err**2
                    grad_swarm[i] += coeff * position_diff

        grad_tracking = np.zeros_like(x)
        cost_tracking = 0.0
        if tracking_reference is not None:
            tracking_sample_divisor = max(self.config.tracking_sample_divisor, 1)
            sample_dt = delta_t / float(tracking_sample_divisor)
            duration = max((qnum - 3) * delta_t, 0.0)
            sample_end_time = (2.0 * duration) / 3.0
            sample_t = 0.0

            while sample_t <= sample_end_time + 1e-9:
                segment_index, u = self._get_segment_index_and_u(
                    t=sample_t,
                    delta_t=delta_t,
                    num_segments=qnum - 3,
                )
                weights = self._get_position_basis_weights(u)
                local_cp = x[segment_index : segment_index + 4]
                sample_position = weights @ local_cp
                global_sample_time = traj_start_time + sample_t
                reference_position = (
                    tracking_reference.target_position
                    + tracking_reference.target_velocity
                    * (global_sample_time - tracking_reference.target_timestamp)
                    + tracking_reference.formation_offset
                )
                position_error = sample_position - reference_position
                cost_tracking += float(position_error @ position_error) * sample_dt
                grad_position = 2.0 * position_error * sample_dt
                for local_idx, weight in enumerate(weights):
                    grad_tracking[segment_index + local_idx] += weight * grad_position
                sample_t += sample_dt

        cost = (
            self.config.lambda_smooth * cost_smoothness
            + self.config.lambda_feasibility * cost_feasibility
            + lambda_dist_apply * cost_dist
            + lambda_swarm_apply * cost_swarm
            + self.config.lambda_tracking * cost_tracking
        )
        grad = (
            self.config.lambda_smooth * grad_smoothness
            + self.config.lambda_feasibility * grad_feasibility
            + lambda_dist_apply * grad_dist
            + lambda_swarm_apply * grad_swarm
            + self.config.lambda_tracking * grad_tracking
        )
        grad = grad[3:-3, :]
        return float(cost), grad.flatten()

    def _objective_gradient_refine(
        self,
        x: Vector,
        param: _RefineParam,
    ) -> tuple[float, Vector]:
        x = x.reshape(-1, 3)
        control_points_opt = param["ctps_opt"]
        x = np.vstack((control_points_opt[:3, :], x, control_points_opt[-3:, :]))
        qnum, _ = x.shape
        delta_t = param["t_ubs"]
        ref_point = param["ref_point"]

        delta_t = 1.0
        inv_dt3 = (1.0 / delta_t) ** 3
        inv_dt2 = (1.0 / delta_t) ** 2
        jerk_array = np.array([-1.0, 3.0, -3.0, 1.0], dtype=np.float64)
        grad_smoothness = np.zeros_like(x)
        cost_smoothness = 0.0
        for i in range(qnum - 3):
            jerk_i = inv_dt3 * jerk_array @ x[i : i + 4, :]
            cost_smoothness += jerk_i @ jerk_i
            grad_smoothness[i : i + 4, :] += (
                2.0
                * inv_dt3
                * np.outer(
                    jerk_array,
                    jerk_i,
                )
            )
        delta_t = param["t_ubs"]
        inv_dt3 = (1.0 / delta_t) ** 3
        inv_dt2 = (1.0 / delta_t) ** 2

        grad_feasibility = np.zeros_like(x)
        cost_feasibility = 0.0

        for i in range(qnum - 1):
            vi = (x[i + 1] - x[i]) / delta_t
            for j in range(3):
                if vi[j] > self.config.max_vel:
                    cost_feasibility += (vi[j] - self.config.max_vel) ** 2 * inv_dt2
                    grad_contrib = (
                        2.0 * (vi[j] - self.config.max_vel) / delta_t * inv_dt2
                    )
                elif vi[j] < -self.config.max_vel:
                    cost_feasibility += (vi[j] + self.config.max_vel) ** 2 * inv_dt2
                    grad_contrib = (
                        2.0 * (vi[j] + self.config.max_vel) / delta_t * inv_dt2
                    )
                else:
                    grad_contrib = 0.0
                grad_feasibility[i + 1, j] += grad_contrib
                grad_feasibility[i, j] -= grad_contrib

        for i in range(qnum - 2):
            ai = (x[i] - 2.0 * x[i + 1] + x[i + 2]) * inv_dt2
            for j in range(3):
                if ai[j] > self.config.max_acc:
                    cost_feasibility += (ai[j] - self.config.max_acc) ** 2
                    grad_contrib = 2.0 * (ai[j] - self.config.max_acc) * inv_dt2
                elif ai[j] < -self.config.max_acc:
                    cost_feasibility += (ai[j] + self.config.max_acc) ** 2
                    grad_contrib = 2.0 * (ai[j] + self.config.max_acc) * inv_dt2
                else:
                    grad_contrib = 0.0
                grad_feasibility[i, j] += grad_contrib
                grad_feasibility[i + 1, j] -= 2.0 * grad_contrib
                grad_feasibility[i + 2, j] += grad_contrib

        grad_fit = np.zeros_like(x)
        cost_fit = 0.0
        para_a = 5.0
        para_b = 1.0
        for i in range(2, qnum - 2):
            xpt = (x[i - 1] + 4.0 * x[i] + x[i + 1]) / 6.0 - ref_point[i - 1]
            v = ref_point[i] - ref_point[i - 2]
            v_dir = v / (np.linalg.norm(v) + 1e-5)
            x_dot_v = xpt @ v_dir
            x_cross_v = np.cross(xpt, v_dir)
            cost_fit += (x_dot_v / para_a) ** 2 + (
                np.linalg.norm(x_cross_v) / para_b
            ) ** 2

            df_dxi = (2.0 * x_dot_v / para_a**2) * v_dir + (2.0 / para_b**2) * np.cross(
                v_dir, x_cross_v
            )

            grad_fit[i - 1] += df_dxi * (1.0 / 6.0)
            grad_fit[i] += df_dxi * (4.0 / 6.0)
            grad_fit[i + 1] += df_dxi * (1.0 / 6.0)

        cost = (
            self.config.lambda_smooth * cost_smoothness
            + self.config.lambda_feasibility * cost_feasibility
            + self.config.lambda_fitness * cost_fit
        )
        grad = (
            self.config.lambda_smooth * grad_smoothness
            + self.config.lambda_feasibility * grad_feasibility
            + self.config.lambda_fitness * grad_fit
        )
        grad = grad[3:-3, :]
        return float(cost), grad.flatten()

    @staticmethod
    def _get_segment_index_and_u(
        t: float,
        delta_t: float,
        num_segments: int,
    ) -> tuple[int, float]:
        if num_segments <= 0:
            raise ValueError(f"Expected num_segments > 0, but got {num_segments}.")
        if delta_t <= 0.0:
            raise ValueError(f"Expected delta_t > 0, but got {delta_t}.")

        duration = num_segments * delta_t
        t_clamped = min(max(float(t), 0.0), duration)
        if np.isclose(t_clamped, duration):
            return num_segments - 1, 1.0

        quotient, remainder = divmod(t_clamped, delta_t)
        return int(quotient), float(remainder / delta_t)

    @staticmethod
    def _get_position_basis_weights(u: float) -> Vector:
        basis_p = np.array([[1.0, u, u**2, u**3]], dtype=np.float64)
        return (basis_p @ UniformBSpline.P_MATRIX).reshape(4)
