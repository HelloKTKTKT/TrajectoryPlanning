from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cvxopt as cvx
from scipy.spatial.transform import Rotation

from trajplan.quadrotor.model import QuadrotorPhysicalConfig
from trajplan.quadrotor.state import QuadrotorState
from trajplan.shared_types import Matrix, Vector
from trajplan.quadrotor.math_utils import (
    cross,
    euler_zyx2quat,
    q_dot,
    q_rotate,
    vec_scale,
)


@dataclass(slots=True)
class DifferentialFlatnessControllerConfig:
    kp: Matrix
    kv: Matrix
    ka: Matrix
    kvi: Matrix

    kq: Matrix
    kw: Matrix

    ep_max: float
    ev_max: float
    bw_max: Vector
    sat_int_ev: float

    minimum_thrust: float
    maximum_thrust: float
    epsilon: float

    rotor_force_min: float
    rotor_force_max_scale: float

    qp_weight: Matrix

    def __post_init__(self) -> None:
        self.kp = np.asarray(self.kp, dtype=np.float64)
        self.kv = np.asarray(self.kv, dtype=np.float64)
        self.ka = np.asarray(self.ka, dtype=np.float64)
        self.kvi = np.asarray(self.kvi, dtype=np.float64)

        self.kq = np.asarray(self.kq, dtype=np.float64)
        self.kw = np.asarray(self.kw, dtype=np.float64)

        self.bw_max = np.asarray(self.bw_max, dtype=np.float64).reshape(-1)
        self.qp_weight = np.asarray(self.qp_weight, dtype=np.float64)

        self.ep_max = float(self.ep_max)
        self.ev_max = float(self.ev_max)
        self.sat_int_ev = float(self.sat_int_ev)

        self.minimum_thrust = float(self.minimum_thrust)
        self.maximum_thrust = float(self.maximum_thrust)
        self.epsilon = float(self.epsilon)

        self.rotor_force_min = float(self.rotor_force_min)
        self.rotor_force_max_scale = float(self.rotor_force_max_scale)

        if self.kp.shape != (3, 3):
            raise ValueError(f"Expected kp shape (3, 3), but got {self.kp.shape}.")
        if self.kv.shape != (3, 3):
            raise ValueError(f"Expected kv shape (3, 3), but got {self.kv.shape}.")
        if self.ka.shape != (3, 3):
            raise ValueError(f"Expected ka shape (3, 3), but got {self.ka.shape}.")
        if self.kvi.shape != (3, 3):
            raise ValueError(f"Expected kvi shape (3, 3), but got {self.kvi.shape}.")

        if self.kq.shape != (3, 3):
            raise ValueError(f"Expected kq shape (3, 3), but got {self.kq.shape}.")
        if self.kw.shape != (3, 3):
            raise ValueError(f"Expected kw shape (3, 3), but got {self.kw.shape}.")

        if self.bw_max.shape != (3,):
            raise ValueError(
                f"Expected bw_max shape (3,), but got {self.bw_max.shape}."
            )
        if self.qp_weight.shape != (4, 4):
            raise ValueError(
                f"Expected qp_weight shape (4, 4), but got {self.qp_weight.shape}."
            )

        if self.ep_max <= 0.0:
            raise ValueError(f"Expected ep_max > 0, but got {self.ep_max}.")
        if self.ev_max <= 0.0:
            raise ValueError(f"Expected ev_max > 0, but got {self.ev_max}.")
        if np.any(self.bw_max <= 0.0):
            raise ValueError(f"Expected positive bw_max, but got {self.bw_max}.")
        if self.sat_int_ev < 0.0:
            raise ValueError(f"Expected sat_int_ev >= 0, but got {self.sat_int_ev}.")

        if self.minimum_thrust < 0.0:
            raise ValueError(
                f"Expected minimum_thrust >= 0, but got {self.minimum_thrust}."
            )
        if self.maximum_thrust <= 0.0:
            raise ValueError(
                f"Expected maximum_thrust > 0, but got {self.maximum_thrust}."
            )
        if self.minimum_thrust > self.maximum_thrust:
            raise ValueError(
                "Expected minimum_thrust <= maximum_thrust, "
                f"but got {self.minimum_thrust} > {self.maximum_thrust}."
            )
        if self.epsilon <= 0.0:
            raise ValueError(f"Expected epsilon > 0, but got {self.epsilon}.")
        if self.rotor_force_min < 0.0:
            raise ValueError(
                f"Expected rotor_force_min >= 0, but got {self.rotor_force_min}."
            )
        if not (0.0 < self.rotor_force_max_scale <= 1.0):
            raise ValueError(
                "Expected rotor_force_max_scale in (0, 1], "
                f"but got {self.rotor_force_max_scale}."
            )


class DifferentialFlatnessController:
    """
    Differential-flatness-based quadrotor controller.

    Input
    -----
    - current quadrotor state:
      position, velocity, euler angles, body angular rate
    - flat reference:
      p, v, a, j, s, psi, d_psi

    Output
    ------
    - rotor thrust command, shape (4,)

    Notes
    -----
    This controller is intended for the higher-fidelity simulation path.
    It uses the quadrotor physical model and a QP-based thrust allocation.
    """

    def __init__(
        self,
        physical_config: QuadrotorPhysicalConfig,
        config: DifferentialFlatnessControllerConfig,
    ) -> None:
        self.physical_config = physical_config
        self.config = config

        self.int_ev = np.zeros((3, 1), dtype=np.float64)

        rotor_force_max = (
            self.config.rotor_force_max_scale * self.physical_config.max_force
        )
        rotor_force_min = self.config.rotor_force_min

        self.umax_cvx = rotor_force_max * np.ones(
            (self.physical_config.rotor_count, 1),
            dtype=np.float64,
        )
        self.umin_cvx = rotor_force_min * np.ones(
            (self.physical_config.rotor_count, 1),
            dtype=np.float64,
        )

        self.P_cvx = (
            2.0
            * self.physical_config.rforce2pseudo.T
            @ self.config.qp_weight
            @ self.physical_config.rforce2pseudo
        )
        self.G_cvx = np.vstack(
            (
                np.eye(self.physical_config.rotor_count, dtype=np.float64),
                -np.eye(self.physical_config.rotor_count, dtype=np.float64),
            )
        )
        self.h_cvx = np.vstack((self.umax_cvx, -self.umin_cvx))

    def compute_control(
        self,
        current_state: QuadrotorState,
        reference_pva: Vector,
    ) -> np.ndarray:
        """
        Compute rotor thrust command from current state and reference pva.

        Parameters
        ----------
        current_state
            QuadrotorState containing:
            - pva
            - euler
            - angular_rate
        reference_pva
            Desired [position, velocity, acceleration], shape (9,)

        Returns
        -------
        np.ndarray
            Rotor thrust command, shape (4, 1)

        Notes
        -----
        Internally, the original differential-flatness controller expects a
        flat-output-style reference table. For compatibility with the current
        project, we construct that table from pva and set jerk / snap / yaw /
        yaw-rate terms to zero by default.
        """
        reference_pva = np.asarray(reference_pva, dtype=np.float64).reshape(-1)
        if reference_pva.shape != (9,):
            raise ValueError(
                f"Expected reference_pva shape (9,), but got {reference_pva.shape}."
            )

        xflat = np.zeros((4, 5), dtype=np.float64)
        xflat[0:3, 0] = reference_pva[0:3]  # position
        xflat[0:3, 1] = reference_pva[3:6]  # velocity
        xflat[0:3, 2] = reference_pva[6:9]  # acceleration
        # jerk / snap / psi / d_psi are currently set to zero

        x_ref = {
            "p": xflat[0:3, 0].reshape(3, 1),
            "v": xflat[0:3, 1].reshape(3, 1),
            "a": xflat[0:3, 2].reshape(3, 1),
            "j": xflat[0:3, 3].reshape(3, 1),
            "s": xflat[0:3, 4].reshape(3, 1),
            "psi": float(xflat[3, 0]),
            "d_psi": float(xflat[3, 1]),
        }

        x_c = {
            "p": current_state.position.reshape(3, 1),
            "v": current_state.velocity.reshape(3, 1),
            "euler": current_state.euler.reshape(3, 1),
            "q": euler_zyx2quat(current_state.euler).reshape(4, 1),
            "bw": current_state.angular_rate.reshape(3, 1),
        }

        # --------------------------------------------------------------
        # Step 1: translational control
        # --------------------------------------------------------------
        ep = x_ref["p"] - x_c["p"]
        if np.max(np.abs(ep)) > self.config.ep_max:
            ep = ep * self.config.ep_max / np.max(np.abs(ep))

        ev = x_ref["v"] - x_c["v"]
        if np.max(np.abs(ev)) > self.config.ev_max:
            ev = ev * self.config.ev_max / np.max(np.abs(ev))

        ea = x_ref["a"]

        u_pd = self.config.kp @ ep + self.config.kv @ ev

        if np.sum(np.abs(x_ref["v"])) != 0.0:
            self.int_ev = np.zeros((3, 1), dtype=np.float64)
        else:
            for i in range(3):
                if np.abs(u_pd[i, 0]) < 0.2:
                    self.int_ev[i, 0] += u_pd[i, 0] * 0.02
            self.int_ev = vec_scale(
                self.int_ev.reshape(-1),
                self.config.sat_int_ev,
            ).reshape(3, 1)

        ad = (
            self.config.kp @ ep
            + self.config.kv @ ev
            + self.config.ka @ ea
            + self.config.kvi @ self.int_ev
        )

        if ad[2, 0] < -6.0:
            ad = ad * (-6.0) / ad[2, 0]

        fd = (
            ad - np.array([[0.0], [0.0], [-9.81]], dtype=np.float64)
        ) * self.physical_config.mass

        zbt = q_rotate(
            x_c["q"].reshape(-1),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        ).reshape(3, 1)
        ybt = q_rotate(
            x_c["q"].reshape(-1),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        ).reshape(3, 1)
        xbt = q_rotate(
            x_c["q"].reshape(-1),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        ).reshape(3, 1)

        u_thrust = float(
            np.clip(
                (zbt.T @ fd).item(),
                self.config.minimum_thrust,
                self.config.maximum_thrust,
            )
        )

        # --------------------------------------------------------------
        # Step 2: desired attitude
        # --------------------------------------------------------------
        fd_norm = np.linalg.norm(fd)
        zbd = fd / fd_norm if fd_norm > self.config.epsilon else zbt

        xc = np.array(
            [[np.cos(x_ref["psi"])], [np.sin(x_ref["psi"])], [0.0]],
            dtype=np.float64,
        )
        yc = np.array(
            [[-np.sin(x_ref["psi"])], [np.cos(x_ref["psi"])], [0.0]],
            dtype=np.float64,
        )

        xbd_prototype = cross(yc.reshape(-1), zbd.reshape(-1)).reshape(3, 1)
        xbd_prototype_norm = np.linalg.norm(xbd_prototype)

        if xbd_prototype_norm > self.config.epsilon:
            xbd = xbd_prototype / xbd_prototype_norm
        else:
            xb_proj = xbt - (yc.T @ xbt) * yc
            xb_proj_norm = np.linalg.norm(xb_proj)
            if xb_proj_norm > self.config.epsilon:
                xbd = xb_proj / xb_proj_norm
            else:
                xbd = xc

        ybd = cross(zbd.reshape(-1), xbd.reshape(-1)).reshape(3, 1)

        Rd = np.hstack((xbd, ybd, zbd))
        qd_xyzw = Rotation.from_matrix(Rd).as_quat(canonical=False)
        qd = np.roll(qd_xyzw, shift=1).reshape(4, 1)

        # --------------------------------------------------------------
        # Step 3: desired body rate and attitude control
        # --------------------------------------------------------------
        d_norm_fd = self.physical_config.mass * (x_ref["j"].T @ zbt).item()

        thrust_for_hw = max(u_thrust, self.config.epsilon)
        hw = (self.physical_config.mass * x_ref["j"] - d_norm_fd * zbt) / thrust_for_hw

        bwx_ref = float((-hw.T @ ybt).item())
        bwy_ref = float((hw.T @ xbt).item())
        bwz_ref = float(x_ref["d_psi"] * (np.array([[0.0, 0.0, 1.0]]) @ zbt).item())
        bw_ref = np.array([[bwx_ref], [bwy_ref], [bwz_ref]], dtype=np.float64)
        bw_ref = np.clip(
            bw_ref,
            -self.config.bw_max.reshape(3, 1),
            self.config.bw_max.reshape(3, 1),
        )

        q_conj = x_c["q"] * np.array([[1.0], [-1.0], [-1.0], [-1.0]])
        qe = q_dot(qd.reshape(-1), q_conj.reshape(-1)).reshape(4, 1)

        qew, qex, qey, qez = qe[:, 0]
        yaw_norm = np.sqrt(qew**2 + qez**2)
        yaw_norm = max(yaw_norm, self.config.epsilon)

        qe_red = (
            np.array(
                [
                    [qew * qex - qey * qez],
                    [qew * qey + qex * qez],
                    [0.0],
                ],
                dtype=np.float64,
            )
            / yaw_norm
        )
        qe_yaw = np.array([[0.0], [0.0], [qez]], dtype=np.float64) / yaw_norm

        dbw_desired = (
            self.config.kq @ qe_red
            + np.sign(qew) * (self.config.kq @ qe_yaw)
            + self.config.kw @ (bw_ref - x_c["bw"])
        )

        tau_desired = self.physical_config.inertia_matrix @ dbw_desired + cross(
            x_c["bw"].reshape(-1),
            (self.physical_config.inertia_matrix @ x_c["bw"]).reshape(-1),
        ).reshape(3, 1)

        u_pseudo_desired = np.vstack(
            (
                np.array([[u_thrust]], dtype=np.float64),
                tau_desired,
            )
        )

        return self.thrust_allocation_cvx(u_pseudo_desired)

    def thrust_allocation_cvx(
        self,
        u_pseudo_desired: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the rotor thrust allocation QP.

        Parameters
        ----------
        u_pseudo_desired
            Desired pseudo input [total thrust, tau_x, tau_y, tau_z],
            shape (4, 1) or (4,).

        Returns
        -------
        np.ndarray
            Rotor thrust command, shape (4, 1).
        """
        u_pseudo_desired = np.asarray(
            u_pseudo_desired,
            dtype=np.float64,
        ).reshape(-1, 1)

        if u_pseudo_desired.shape != (4, 1):
            raise ValueError(
                "Expected u_pseudo_desired shape (4, 1), "
                f"but got {u_pseudo_desired.shape}."
            )

        q_cvx = (
            -2.0
            * self.physical_config.rforce2pseudo.T
            @ self.config.qp_weight
            @ u_pseudo_desired
        )

        cvx.solvers.options["show_progress"] = False
        solution = cvx.solvers.qp(
            P=cvx.matrix(self.P_cvx),
            q=cvx.matrix(q_cvx),
            G=cvx.matrix(self.G_cvx),
            h=cvx.matrix(self.h_cvx),
        )

        status = solution["status"]
        if status != "optimal":
            raise RuntimeError(f"Thrust allocation QP failed with status '{status}'.")

        rotor_thrust = np.array(solution["x"], dtype=np.float64)
        return rotor_thrust
