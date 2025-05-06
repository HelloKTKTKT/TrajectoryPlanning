import numpy as np
import cvxopt as cvx
import src.quadrotor.parameters as quad_para
import src.utils.func as func
from scipy.spatial.transform import Rotation


class DifferentialFlatness:
    def __init__(self):
        self.int_ev = np.zeros((3, 1))
        # self.kp = np.diag([5, 5, 5])
        # self.kv = np.diag([4, 4, 4])
        # self.ka = np.diag([1, 1, 1])
        self.kp = np.diag([8, 8, 8])
        self.kv = np.diag([4.5, 4.5, 4.5])
        self.ka = np.diag([1, 1, 1])
        self.kvi = 0 * np.eye(3)
        self.kq = np.diag([250.0, 250.0, 5.0])
        self.kw = np.diag([20, 20.0, 10.0])

        self.ep_max = 2
        self.ev_max = 2

        self.bw_max = np.array([[18.0, 18.0, 3.0]]).T
        self.sat_int_ev = 0.5
        self.minimum_thrust = 0.01
        self.maximum_thrust = 16.0
        self.epsilon = 1e-3

        self.umax_cvx = 0.99 * quad_para.max_force * np.ones((4, 1))
        self.umin_cvx = 0.01 * np.ones((4, 1))
        self.w_cvx = np.diag([0.001, 10, 10, 0.1])
        self.p_cvx = 2 * quad_para.rforce2pseudo.T @ self.w_cvx @ quad_para.rforce2pseudo
        self.g_cvx = np.vstack((np.eye(4), -np.eye(4)))
        self.h_cvx = np.vstack((self.umax_cvx, -self.umin_cvx))

    def execute(self, xin, uin, xflat):
        uin = np.maximum(uin, 0.01)
        xin.shape = (-1, 1)
        uin.shape = (-1, 1)
        
        # step 1: extract information
        x_ref = {
            "p": xflat[0:3, 0, np.newaxis],
            "v": xflat[0:3, 1, np.newaxis],
            "a": xflat[0:3, 2, np.newaxis],
            "j": xflat[0:3, 3, np.newaxis],
            "s": xflat[0:3, 4, np.newaxis],
            "psi": xflat[3, 0],
            "d_psi": xflat[3, 1]
        }
        x_c = {
            "p": xin[0:3],
            "v": xin[3:6],
            "euler": xin[6:9],
            "q": func.euler_zyx2quat(xin[6:9]),
            "bw": xin[9:12],
            "thrust": np.sum(uin)
        }

        # step 2: derive ad --> fd --> u_thrust
        ep = x_ref["p"] - x_c["p"]
        ep = ep * self.ep_max / np.max(abs(ep)) if np.max(abs(ep)) > self.ep_max else ep
        ev = x_ref["v"] - x_c["v"]
        ev = ev * self.ev_max / np.max(abs(ev)) if np.max(abs(ev)) > self.ev_max else ev
        ea = x_ref["a"]

        u_pd = self.kp @ ep + self.kv @ ev

        # derive the integration part
        if np.sum(abs(x_ref["v"])) != 0:
            self.int_ev = np.zeros((3, 1))
        else:
            for i in range(3):
                if abs(u_pd[i, 0]) < 0.2:
                    self.int_ev[i, 0] += u_pd[i, 0] * 0.02
            self.int_ev = func.vec_scale(self.int_ev, self.sat_int_ev)

        ad = self.kp @ ep + self.kv @ ev + self.ka @ ea + self.kvi @ self.int_ev  # ad = fd/m + [0,0,-g]
        if ad[2, 0] < -6:
            ad = ad * (-6) / ad[2, 0]  # make sure the gravity can still provide the downward acceleration

        fd = (ad - np.array([[0, 0, -9.81]]).T) * quad_para.mass
        zbt = func.q_rotate(x_c["q"], np.array([[0, 0, 1.0]]).T)
        ybt = func.q_rotate(x_c["q"], np.array([[0, 1.0, 0]]).T)
        xbt = func.q_rotate(x_c["q"], np.array([[1.0, 0, 0]]).T)
        u_thrust = np.minimum(np.maximum(self.minimum_thrust, (zbt.T @ fd).item()), self.maximum_thrust)

        # step 3: derive qd
        fd_norm = np.linalg.norm(fd)
        zbd = fd / fd_norm if fd_norm > self.epsilon else zbt

        xc = np.array([[np.cos(x_ref["psi"]), np.sin(x_ref["psi"]), 0]]).T
        yc = np.array([[-np.sin(x_ref["psi"]), np.cos(x_ref["psi"]), 0]]).T
        xbd_prototype = func.cross(yc, zbd)
        xbd_prototype_norm = np.linalg.norm(xbd_prototype)
        if xbd_prototype_norm > self.epsilon:
            xbd = xbd_prototype / xbd_prototype_norm
        else:  # yc & zbd almost in the same line. yc_|_(xbd, ybd) plane; yc_|_(xc, zc) plane --> xbd inside (xc, zc)
            # first try to project xb onto (xc, zc) plane to get xbd
            xb_proj = xbt - (yc.T @ xbt) * yc
            xb_proj_norm = np.linalg.norm(xb_proj)
            xbd = xb_proj / xb_proj_norm if xbd_prototype_norm > self.epsilon else xc

        ybd = func.cross(zbd, xbd)
        rmat_d = np.hstack((xbd, ybd, zbd))
        qd_xyzw = Rotation.from_matrix(rmat_d).as_quat(canonical=False)
        qd = np.roll(qd_xyzw, shift=1).reshape(4, 1)

        # step 4: derive hw -->Bwd
        d_norm_fd = quad_para.mass * (x_ref["j"].T @ zbt).item()
        hw = (quad_para.mass * x_ref["j"] - d_norm_fd * zbt) / x_c["thrust"]
        bwx_ref = -hw.T @ ybt
        bwy_ref = hw.T @ xbt
        bwz_ref = x_ref["d_psi"] * np.array([[0, 0, 1.0]]) @ zbt
        bw_ref = np.vstack((bwx_ref, bwy_ref, bwz_ref))
        # Bw_ref = Bw_ref * self.Bw_max / np.max(abs(Bw_ref)) if np.max(abs(Bw_ref)) > self.Bw_max else Bw_ref
        bw_ref = np.clip(bw_ref, -self.bw_max, self.bw_max)

        # step 5: orientation control
        q_conj = x_c["q"] * np.array([[1, -1, -1, -1.0]]).T
        qe = func.q_dot(qd, q_conj)
        qew, qex, qey, qez = qe[0, 0], qe[1, 0], qe[2, 0], qe[3, 0]
        qe_red = 1 / np.sqrt(qew ** 2 + qez ** 2) * np.array([[qew * qex - qey * qez, qew * qey + qex * qez, 0]]).T
        qe_yaw = 1 / np.sqrt(qew ** 2 + qez ** 2) * np.array([[0, 0, qez]]).T
        d_bwd = self.kq @ qe_red + np.sign(qew) * self.kq @ qe_yaw + self.kw @ (bw_ref - x_c["bw"])

        # step 6: derive u_pseudo
        taud = quad_para.inertia_matrix @ d_bwd + func.cross(x_c["bw"], quad_para.inertia_matrix @ x_c["bw"])
        u_pseudo = np.vstack((np.array([[u_thrust]]), taud))
        u_thrust = self.thrust_allocation_cvx(u_pseudo)
        return u_thrust

    def thrust_allocation_cvx(self, upse_d):
        q_cvx = -2 * quad_para.rforce2pseudo.T @ self.w_cvx @ upse_d
        sol = cvx.solvers.qp(P=cvx.matrix(self.p_cvx), q=cvx.matrix(q_cvx), G=cvx.matrix(self.g_cvx),
                             h=cvx.matrix(self.h_cvx), options={"show_progress": False})
        u_thrust = np.array(sol["x"])
        return u_thrust


