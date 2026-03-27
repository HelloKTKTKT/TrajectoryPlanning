from __future__ import annotations

from dataclasses import dataclass

import cvxopt as cvx
import numpy as np

from trajplan.shared_types import FloatArray, Matrix, Vector
from trajplan.utils import clamp


@dataclass(slots=True)
class LinearMpcConfig:
    """
    Fixed configuration for linear MPC global trajectory generation.

    Array shape convention
    ----------------------
    - Ad: shape (9, 9)
    - Bd: shape (9, 3)
    - Q: shape (9, 9)
    - QN: shape (9, 9)
    - R: shape (3, 3)
    - max_velocity_3d: shape (3,)
    - max_acceleration_3d: shape (3,)
    - max_jerk_3d: shape (3,)
    """

    ts: float
    planning_horizon_steps: int

    Ad: Matrix
    Bd: Matrix

    Q: Matrix
    QN: Matrix
    R: Matrix

    max_velocity_3d: Vector
    max_acceleration_3d: Vector
    max_jerk_3d: Vector


class LinearMpcTrajectory:
    """
    Linear MPC based global trajectory.

    Design notes
    ------------
    - This class keeps both the MPC setup and the solved trajectory result.
    - It is intentionally close to the simulator_v0.1 LinearMpcTraj design.
    - A separate wrapper is not introduced in v1.
    """

    def __init__(self, config: LinearMpcConfig) -> None:
        self.config = config

        self.ts: float = float(config.ts)
        self.Np: int = int(config.planning_horizon_steps)

        if self.ts <= 0.0:
            raise ValueError(f"Expected ts > 0, but got {self.ts}.")
        if self.Np <= 0:
            raise ValueError(f"Expected planning_horizon_steps > 0, but got {self.Np}.")

        self.Ad: Matrix = np.asarray(config.Ad, dtype=np.float64)
        self.Bd: Matrix = np.asarray(config.Bd, dtype=np.float64)
        self.Q: Matrix = np.asarray(config.Q, dtype=np.float64)
        self.QN: Matrix = np.asarray(config.QN, dtype=np.float64)
        self.R: Matrix = np.asarray(config.R, dtype=np.float64)

        self.max_velocity_3d: Vector = np.asarray(
            config.max_velocity_3d,
            dtype=np.float64,
        ).reshape(-1)
        self.max_acceleration_3d: Vector = np.asarray(
            config.max_acceleration_3d,
            dtype=np.float64,
        ).reshape(-1)
        self.max_jerk_3d: Vector = np.asarray(
            config.max_jerk_3d,
            dtype=np.float64,
        ).reshape(-1)

        if self.Ad.shape != (9, 9):
            raise ValueError(f"Expected Ad shape (9, 9), but got {self.Ad.shape}.")
        if self.Bd.shape != (9, 3):
            raise ValueError(f"Expected Bd shape (9, 3), but got {self.Bd.shape}.")
        if self.Q.shape != (9, 9):
            raise ValueError(f"Expected Q shape (9, 9), but got {self.Q.shape}.")
        if self.QN.shape != (9, 9):
            raise ValueError(f"Expected QN shape (9, 9), but got {self.QN.shape}.")
        if self.R.shape != (3, 3):
            raise ValueError(f"Expected R shape (3, 3), but got {self.R.shape}.")
        if self.max_velocity_3d.shape != (3,):
            raise ValueError(
                "Expected max_velocity_3d shape (3,), "
                f"but got {self.max_velocity_3d.shape}."
            )
        if self.max_acceleration_3d.shape != (3,):
            raise ValueError(
                "Expected max_acceleration_3d shape (3,), "
                f"but got {self.max_acceleration_3d.shape}."
            )
        if self.max_jerk_3d.shape != (3,):
            raise ValueError(
                f"Expected max_jerk_3d shape (3,), but got {self.max_jerk_3d.shape}."
            )

        # Constraints.
        self.x_ub: FloatArray = np.hstack(
            (1e5 * np.ones(3), self.max_velocity_3d, self.max_acceleration_3d)
        ).reshape(-1, 1)
        self.x_lb: FloatArray = -self.x_ub

        self.u_ub: FloatArray = self.max_jerk_3d.reshape(-1, 1)
        self.u_lb: FloatArray = -self.u_ub

        # These fields are intentionally kept, because your old planner-manager
        # logic expects them.
        self.start_time: float = 0.0
        self.duration: float = self.ts * self.Np
        self.current_progress: float = 0.0

        self.x0: FloatArray = np.zeros((9, 1), dtype=np.float64)
        self.xtar: FloatArray = np.zeros((9, 1), dtype=np.float64)

        self.time_index: FloatArray = np.arange(self.Np + 1, dtype=np.float64) * self.ts
        self.Xref: FloatArray = np.zeros((9 * self.Np, 1), dtype=np.float64)

        self.Qcon: Matrix | None = None
        self.Rcon: Matrix | None = None
        self.Ca: Matrix | None = None
        self.Bcon: Matrix | None = None

        self.P_cvx: Matrix | None = None
        self.q_cvx: FloatArray | None = None
        self.G: Matrix | None = None
        self.h: FloatArray | None = None

        self.status: str | None = None
        self.ucon: FloatArray | None = None
        self.Xpred: Matrix | None = None

        self._build_weight_con()
        self._build_sys_con()

    def _build_weight_con(self) -> None:
        q_diag = np.diag(self.Q)
        qn_diag = np.diag(self.QN)

        q_stack = np.tile(q_diag, self.Np - 1)
        q_stack = np.hstack((q_stack, qn_diag))
        self.Qcon = np.diag(q_stack)

        r_diag = np.diag(self.R)
        r_stack = np.tile(r_diag, self.Np)
        self.Rcon = np.diag(r_stack)

    def _build_sys_con(self) -> None:
        a_list = []
        for i in range(self.Np):
            a_list.append(np.linalg.matrix_power(self.Ad, i + 1))
        self.Ca = np.vstack(a_list)

        bcon_list = []
        for i in range(self.Np):
            row_i = [np.zeros_like(self.Bd) for _ in range(self.Np)]
            for j in range(i + 1):
                row_i[j] = np.linalg.matrix_power(self.Ad, i - j) @ self.Bd
            bcon_list.append(np.hstack(row_i))
        self.Bcon = np.vstack(bcon_list)

    def _update_qp_cost(self) -> None:
        if (
            self.Ca is None
            or self.Bcon is None
            or self.Qcon is None
            or self.Rcon is None
        ):
            raise ValueError("System or weight matrices are not initialized.")

        self.P_cvx = (self.Bcon.T @ self.Qcon @ self.Bcon + self.Rcon) * 2.0

        cax = self.Ca @ self.x0
        q_cvx_row = 2.0 * (
            cax.T @ self.Qcon @ self.Bcon - self.Xref.T @ self.Qcon @ self.Bcon
        )
        self.q_cvx = q_cvx_row.T

    def _update_constraint(self) -> None:
        if self.Ca is None or self.Bcon is None:
            raise ValueError("System matrices are not initialized.")

        dim_xplus = self.Np * self.Ad.shape[0]
        gx = np.vstack((np.eye(dim_xplus), -np.eye(dim_xplus)))

        dim_ucon = self.Np * self.Bd.shape[1]
        gu = np.vstack((np.eye(dim_ucon), -np.eye(dim_ucon)))

        self.G = np.vstack((gx @ self.Bcon, gu))

        x_ub_con = np.tile(self.x_ub, (self.Np, 1))
        x_lb_con = np.tile(self.x_lb, (self.Np, 1))
        u_ub_con = np.tile(self.u_ub, (self.Np, 1))
        u_lb_con = np.tile(self.u_lb, (self.Np, 1))

        h_u = np.vstack((x_ub_con, -x_lb_con)) - gx @ self.Ca @ self.x0
        h_l = np.vstack((u_ub_con, -u_lb_con))
        self.h = np.vstack((h_u, h_l))

    def _solve_qp(self) -> None:
        if self.P_cvx is None or self.q_cvx is None or self.G is None or self.h is None:
            raise ValueError("QP matrices are not ready.")

        cvx.solvers.options["show_progress"] = False

        sol = cvx.solvers.qp(
            P=cvx.matrix(self.P_cvx),
            q=cvx.matrix(self.q_cvx),
            G=cvx.matrix(self.G),
            h=cvx.matrix(self.h),
        )

        self.status = sol["status"]
        if self.status != "optimal":
            self.ucon = None
            self.Xpred = None
            return

        self.ucon = np.array(sol["x"], dtype=np.float64)
        self._update_prediction()

    def _update_prediction(self) -> None:
        if self.Ca is None or self.Bcon is None or self.ucon is None:
            raise ValueError("Prediction matrices are not ready.")

        xpred = self.Ca @ self.x0 + self.Bcon @ self.ucon
        self.Xpred = xpred.reshape(-1, self.Ad.shape[0])

    def update_traj_info(
        self,
        start_state: Vector | FloatArray,
        target_state: Vector | FloatArray,
    ) -> None:
        """
        Solve the MPC problem and refresh the stored trajectory result.

        Parameters
        ----------
        start_state
            Initial pva state, shape (9,) or (9, 1).
        target_state
            Target pva state, shape (9,) or (9, 1).
        """
        start_state_array = np.asarray(start_state, dtype=np.float64).reshape(-1, 1)
        target_state_array = np.asarray(target_state, dtype=np.float64).reshape(-1, 1)

        if start_state_array.shape != (9, 1):
            raise ValueError(
                "Expected start_state shape (9,) or (9, 1), "
                f"but got {start_state_array.shape}."
            )
        if target_state_array.shape != (9, 1):
            raise ValueError(
                "Expected target_state shape (9,) or (9, 1), "
                f"but got {target_state_array.shape}."
            )

        self.x0 = start_state_array
        self.xtar = target_state_array
        self.Xref = np.tile(self.xtar, (self.Np, 1))

        self._update_qp_cost()
        self._update_constraint()
        self._solve_qp()

        if self.status != "optimal" or self.Xpred is None:
            self.time_index = np.array([0.0], dtype=np.float64)
            self.duration = 0.0
            self.current_progress = 0.0
            return

        xpred_error = np.linalg.norm(self.Xpred - self.xtar.T, axis=1)
        reach_index = xpred_error.size - 1
        for i in range(xpred_error.size):
            if xpred_error[i] < 1e-3:
                reach_index = i
                break

        self.Xpred = np.vstack((self.x0.T, self.Xpred[: reach_index + 1]))
        self.time_index = np.arange(self.Xpred.shape[0], dtype=np.float64) * self.ts
        self.duration = float(self.time_index[-1])
        self.current_progress = 0.0

    def evaluate_pva(self, t_cur: float) -> Vector:
        """
        Evaluate pva at local trajectory time t_cur.

        Notes
        -----
        This follows the old implementation.
        Between two MPC samples, position is interpolated using constant
        acceleration within the current step, while velocity and acceleration
        are taken from the current predicted state.
        """
        if self.Xpred is None:
            raise ValueError("Trajectory has not been solved yet.")

        t_cur = clamp(float(t_cur), 0.0, self.duration)

        quotient, remainder = divmod(t_cur, self.ts)
        index = int(quotient)

        if index >= self.Xpred.shape[0] - 1:
            return self.Xpred[-1].copy()

        x_select = self.Xpred[index]

        position = (
            x_select[:3]
            + x_select[3:6] * remainder
            + 0.5 * x_select[6:9] * remainder**2
        )
        velocity = x_select[3:6] + x_select[6:9] * remainder
        acceleration = x_select[6:9]

        return np.hstack((position, velocity, acceleration))
