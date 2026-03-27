from __future__ import annotations

from dataclasses import dataclass
from scipy.optimize import minimize

import numpy as np

from trajplan.shared_types import Vector, PvPair
from trajplan.trajectory.bspline import UniformBSpline


@dataclass(slots=True)
class BsplineOptimizerConfig:
    """
    Configuration for collision-aware B-spline optimization.
    """

    lambda_smooth: float
    lambda_dist: float
    lambda_feasibility: float
    lambda_fitness: float
    tol: float
    safe_dist: float
    max_vel: float
    max_acc: float


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
        config: BsplineOptimizerConfig,
    ) -> None:
        self.config = config

    def optimize(
        self,
        initial_bspline: UniformBSpline,
        pv_pair_info: list[list[PvPair]],
        lambda_dist_apply: float,
    ) -> None:
        start_index = 3
        end_index = initial_bspline.num_control_points - 3
        ctp_var = initial_bspline.control_points.copy()
        x0 = ctp_var[start_index:end_index].flatten()
        param = {
            "ctps_opt": ctp_var,
            "pv_pair_info": pv_pair_info,
            "lambda_dist_apply": lambda_dist_apply,
            "t_ubs": initial_bspline.delta_t,
        }
        opt_res = minimize(
            self._objective_gradient_rebound,
            x0,
            method="L-BFGS-B",
            jac=True,
            args=(param,),
            tol=self.config.tol,
        )

        initial_bspline.control_points[start_index:end_index] = opt_res.x.reshape(-1, 3)

        return initial_bspline

    def refine_feasibility(
        self,
        optimized_bspline: UniformBSpline,
        ratio: float,
    ) -> None:
        bspline_opt = optimized_bspline.copy()
        point_set_ref = []
        for t in np.arange(0, bspline_opt.duration + 1e-3, bspline_opt.delta_t):
            pt = bspline_opt.evaluate_pva(t)[:3]
            point_set_ref.append(pt)
        point_set_ref = np.vstack(point_set_ref)

        optimized_bspline.delta_t *= ratio
        param = {
            "ctps_opt": optimized_bspline.control_points.copy(),
            "t_ubs": optimized_bspline.delta_t,
            "ref_point": point_set_ref,
        }

        start_index = 3
        end_index = param["ctps_opt"].shape[0] - 3
        x0 = param["ctps_opt"][start_index:end_index].flatten()
        opt_res = minimize(
            self._objective_gradient_refine,
            x0,
            method="L-BFGS-B",
            jac=True,
            args=(param,),
            tol=self.config.tol,
        )
        optimized_bspline.control_points[start_index:end_index] = opt_res.x.reshape(
            -1, 3
        )

    def _objective_gradient_rebound(
        self,
        x: Vector,
        param: dict,
    ) -> tuple[float, Vector]:
        x = x.reshape(-1, 3)
        ctps_opt = param["ctps_opt"]
        pv_pair_info = param["pv_pair_info"]
        lambda_dist_apply = param["lambda_dist_apply"]
        x = np.vstack((ctps_opt[:3, :], x, ctps_opt[-3:, :]))
        qnum, qdim = x.shape
        # delta_t = param["t_ubs"]

        # 1. Smoothness
        delta_t = 1.0
        inv_dt3 = (1 / delta_t) ** 3
        inv_dt2 = (1 / delta_t) ** 2
        jerk_array = np.array([-1, 3, -3, 1])
        grad_smoothness = np.zeros_like(x)
        cost_smoothness = 0.0
        for i in range(qnum - 3):
            jerk_i = inv_dt3 * jerk_array @ x[i : i + 4, :]
            cost_smoothness += jerk_i @ jerk_i
            grad_smoothness[i : i + 4, :] += 2 * inv_dt3 * np.outer(jerk_array, jerk_i)
        delta_t = param["t_ubs"]

        # 2. Feasibility
        grad_feasibility = np.zeros_like(x)
        cost_feasibility = 0.0
        # 2.1: velocity feasibility
        for i in range(qnum - 1):
            vi = (x[i + 1] - x[i]) / delta_t
            for j in range(3):
                if vi[j] > self.config.max_vel:
                    # multiply inv_dt2 to make vel & acc same scale
                    cost_feasibility += (vi[j] - self.config.max_vel) ** 2 * inv_dt2
                    grad_contrib = 2 * (vi[j] - self.config.max_vel) / delta_t * inv_dt2
                elif vi[j] < -self.config.max_vel:
                    cost_feasibility += (vi[j] + self.config.max_vel) ** 2 * inv_dt2
                    grad_contrib = 2 * (vi[j] + self.config.max_vel) / delta_t * inv_dt2
                else:
                    grad_contrib = 0.0
                grad_feasibility[i + 1, j] += grad_contrib
                grad_feasibility[i, j] -= grad_contrib
        # 2.2: accel feasibility
        for i in range(qnum - 2):
            ai = (x[i] - 2 * x[i + 1] + x[i + 2]) * inv_dt2
            for j in range(3):
                if ai[j] > self.config.max_acc:
                    cost_feasibility += (ai[j] - self.config.max_acc) ** 2
                    grad_contrib = 2 * (ai[j] - self.config.max_acc) * inv_dt2
                elif ai[j] < -self.config.max_acc:
                    cost_feasibility += (ai[j] + self.config.max_acc) ** 2
                    grad_contrib = 2 * (ai[j] + self.config.max_acc) * inv_dt2
                else:
                    grad_contrib = 0.0
                grad_feasibility[i, j] += grad_contrib
                grad_feasibility[i + 1, j] -= 2 * grad_contrib
                grad_feasibility[i + 2, j] += grad_contrib

        # 3. Distance
        grad_dist = np.zeros_like(x)
        cost_dist = 0.0
        d_safe = self.config.safe_dist
        for i in range(qnum):
            pv_pair_list_i = pv_pair_info[i]
            for pv_pair in pv_pair_list_i:
                anchor = pv_pair.surface_anchor
                unit_dir = pv_pair.unit_dir
                dist = (x[i] - anchor) @ unit_dir
                c_dist = d_safe - dist
                if c_dist <= 0:
                    # nothing happened
                    continue
                elif 0 < c_dist <= d_safe:
                    cost_dist += c_dist**3
                    grad_dist[i] += -3 * c_dist**2 * unit_dir
                else:
                    cost_dist += (
                        3 * d_safe * c_dist**2 - 3 * d_safe**2 * c_dist + d_safe**3
                    )
                    grad_dist[i] += -(6 * d_safe * c_dist - 3 * d_safe**2) * unit_dir

        cost = (
            self.config.lambda_smooth * cost_smoothness
            + self.config.lambda_feasibility * cost_feasibility
            + lambda_dist_apply * cost_dist
        )
        grad = (
            self.config.lambda_smooth * grad_smoothness
            + self.config.lambda_feasibility * grad_feasibility
            + lambda_dist_apply * grad_dist
        )
        grad = grad[3:-3, :]
        return cost, grad.flatten()

    def _objective_gradient_refine(
        self,
        x: Vector,
        param: dict,
    ) -> tuple[float, Vector]:
        x = x.reshape(-1, 3)
        ctps_opt = param["ctps_opt"]
        x = np.vstack((ctps_opt[:3, :], x, ctps_opt[-3:, :]))
        qnum, qdim = x.shape
        delta_t = param["t_ubs"]
        ref_point = param["ref_point"]

        # 1. Smoothness
        delta_t = 1.0
        inv_dt3 = (1 / delta_t) ** 3
        inv_dt2 = (1 / delta_t) ** 2
        jerk_array = np.array([-1, 3, -3, 1])
        grad_smoothness = np.zeros_like(x)
        cost_smoothness = 0.0
        for i in range(qnum - 3):
            jerk_i = inv_dt3 * jerk_array @ x[i : i + 4, :]
            cost_smoothness += jerk_i @ jerk_i
            grad_smoothness[i : i + 4, :] += 2 * inv_dt3 * np.outer(jerk_array, jerk_i)
        delta_t = param["t_ubs"]

        # 2. Feasibility
        grad_feasibility = np.zeros_like(x)
        cost_feasibility = 0.0
        # 2.1: velocity feasibility
        for i in range(qnum - 1):
            vi = (x[i + 1] - x[i]) / delta_t
            for j in range(3):
                if vi[j] > self.config.max_vel:
                    # multiply inv_dt2 to make vel & acc same scale
                    cost_feasibility += (vi[j] - self.config.max_vel) ** 2 * inv_dt2
                    grad_contrib = 2 * (vi[j] - self.config.max_vel) / delta_t * inv_dt2
                elif vi[j] < -self.config.max_vel:
                    cost_feasibility += (vi[j] + self.config.max_vel) ** 2 * inv_dt2
                    grad_contrib = 2 * (vi[j] + self.config.max_vel) / delta_t * inv_dt2
                else:
                    grad_contrib = 0.0
                grad_feasibility[i + 1, j] += grad_contrib
                grad_feasibility[i, j] -= grad_contrib
        # 2.2: accel feasibility
        for i in range(qnum - 2):
            ai = (x[i] - 2 * x[i + 1] + x[i + 2]) * inv_dt2
            for j in range(3):
                if ai[j] > self.config.max_acc:
                    cost_feasibility += (ai[j] - self.config.max_acc) ** 2
                    grad_contrib = 2 * (ai[j] - self.config.max_acc) * inv_dt2
                elif ai[j] < -self.config.max_acc:
                    cost_feasibility += (ai[j] + self.config.max_acc) ** 2
                    grad_contrib = 2 * (ai[j] + self.config.max_acc) * inv_dt2
                else:
                    grad_contrib = 0.0
                grad_feasibility[i, j] += grad_contrib
                grad_feasibility[i + 1, j] -= 2 * grad_contrib
                grad_feasibility[i + 2, j] += grad_contrib

        # 3. fitness
        grad_fit = np.zeros_like(x)
        cost_fit = 0.0
        para_a = 5.0
        para_b = 1.0
        for i in range(2, qnum - 2):
            xpt = 1 / 6 * (x[i - 1] + 4 * x[i] + x[i + 1]) - ref_point[i - 1]
            v = ref_point[i] - ref_point[i - 2]
            v_dir = v / (np.linalg.norm(v) + 1e-5)
            x_dot_v = xpt @ v_dir
            x_cross_v = np.cross(xpt, v_dir)
            cost_fit += (x_dot_v / para_a) ** 2 + (
                np.linalg.norm(x_cross_v) / para_b
            ) ** 2

            # gradient w.r.t. xi
            #   df/dxi = 2*(x·v)/a2 * v  +  2/b2 * (v × (xi × v))
            df_dxi = (2 * x_dot_v / para_a**2) * v_dir + (2.0 / para_b**2) * np.cross(
                v_dir, x_cross_v
            )

            # chain‐rule back into the three control‐points
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
        return cost, grad.flatten()
