import math
import numpy as np
import cvxopt as cvx
import src.trajectory.parameters as traj_para


class LinearMpcTraj:
    def __init__(self):
        """
        start_state: 9x1 2D numpy array (p, v, a)
        target_state: 9x1 2D numpy array (p, v, a)
        start_time: float
        :Np: int
        :Ad: 2D numpy array
        :Bd: 2D numpy array
        :Q: 2D numpy array, diagonal PSD
        :QN: 2D numpy array, diagonal PSD
        :R: 2D numpy array, diagonal PD
        :x_ub: 2D numpy array, 1 column describes the state upper bound
        :x_lb: 2D numpy array, 1 column describes the state lower bound
        :u_ub: 2D numpy array, 1 column describes the input upper bound
        :u_lb: 2D numpy array, 1 column describes the input lower bound
        :x0: 2D numpy array, 1 column describes the initial state
        :Xref: 2D numpy array, 1 column that vertically concatenated the xref at each prediction time step
        ps: if no upper / lower bound, then just put a huge value e.g., 1e5 or -1e5.
        """
        self.ts = traj_para.MPC_TS
        self.Np = int(traj_para.GLOBAL_MAX_DURATION / self.ts)
        self.time_index = np.arange(self.Np + 1) * self.ts
        self.Ad, self.Bd = traj_para.Ad, traj_para.Bd
        self.Q, self.QN, self.R = traj_para.Q, traj_para.QN, traj_para.R
        self.x_ub = np.hstack((1e5 * np.ones(3), traj_para.MAX_V_3D, traj_para.MAX_A_3D)).reshape(-1, 1)
        self.x_lb = -self.x_ub
        self.u_ub = traj_para.MAX_J_3D.reshape(-1, 1)
        self.u_lb = -self.u_ub
        self.Xref = np.zeros((self.Ad.shape[1] * self.Np, 1))

        # These attributes should be available for each trajectory type.
        self.start_time = 0.0
        self.duration = self.ts * self.Np
        self.current_progress = 0.0
        self.x0 = np.zeros((self.Ad.shape[1], 1))
        self.xtar = np.zeros((self.Ad.shape[1], 1))

        self.Qcon, self.Rcon = None, None
        self.Ca, self.Bcon = None, None
        self.P_cvx, self.q_cvx = None, None
        self.G, self.h = None, None

        self.status = None
        self.ucon = None
        self.Xpred = None

        self.get_weight_con()
        self.get_syscon()
        self.get_qp_cost()
        self.get_constraint()

    def get_weight_con(self):
        """
        call if Np or weight matrices change
        """
        q_diag = np.diag(self.Q)
        qn_diag = np.diag(self.QN)
        temp = np.tile(q_diag, (1, self.Np - 1)).flatten()
        temp = np.hstack((temp, qn_diag))
        q_con = np.diag(temp)
        r_diag = np.diag(self.R)
        temp = np.tile(r_diag, (1, self.Np)).flatten()
        r_con = np.diag(temp)
        self.Qcon, self.Rcon = q_con, r_con

    def get_syscon(self):
        """
        call if system changes.
        """
        a_list = []
        for i in range(self.Np):
            a_list.append(np.linalg.matrix_power(self.Ad, i + 1))
        ca = np.vstack(a_list)

        bcon_list = []
        for i in range(self.Np):
            row_i = [np.zeros_like(self.Bd) for _ in range(self.Np)]
            for j in range(i + 1):
                row_i[j] = np.linalg.matrix_power(self.Ad, i - j) @ self.Bd
            bcon_list.append(np.hstack(row_i))
        bcon = np.vstack(bcon_list)
        self.Ca, self.Bcon = ca, bcon

    def get_qp_cost(self):
        """
        call every time step.
        """
        self.P_cvx = (self.Bcon.T @ self.Qcon @ self.Bcon + self.Rcon) * 2
        cax = self.Ca @ self.x0
        q_cvx_t = 2 * (cax.T @ self.Qcon @ self.Bcon - self.Xref.T @ self.Qcon @ self.Bcon)
        self.q_cvx = q_cvx_t.T

    def get_constraint(self):
        """
        call every time step
        """
        dim_xplus = self.Np * self.Ad.shape[0]
        gx = np.vstack((np.eye(dim_xplus), -np.eye(dim_xplus)))
        dim_ucon = self.Np * self.Bd.shape[1]
        gu = np.vstack((np.eye(dim_ucon), -np.eye(dim_ucon)))
        g = np.vstack((gx @ self.Bcon, gu))
        x_ub_con = np.tile(self.x_ub, (self.Np, 1))
        x_lb_con = np.tile(self.x_lb, (self.Np, 1))
        u_ub_con = np.tile(self.u_ub, (self.Np, 1))
        u_lb_con = np.tile(self.u_lb, (self.Np, 1))
        h_u = np.vstack((x_ub_con, -x_lb_con)) - gx @ self.Ca @ self.x0
        h_l = np.vstack((u_ub_con, -u_lb_con))
        h = np.vstack((h_u, h_l))
        self.G, self.h = g, h

    def solve_qp(self):
        cvx.solvers.options['show_progress'] = False
        sol = cvx.solvers.qp(P=cvx.matrix(self.P_cvx), q=cvx.matrix(self.q_cvx), G=cvx.matrix(self.G),
                             h=cvx.matrix(self.h))

        self.status = sol['status']
        if self.status != "optimal":
            print(f'Solver status = {self.status}')
        self.ucon = np.array(sol['x'])
        self.get_prediction()

    def get_prediction(self):
        xpred = self.Ca @ self.x0 + self.Bcon @ self.ucon
        xpred = xpred.reshape(-1, self.Ad.shape[0])
        self.Xpred = xpred

    def upadte_traj_info(self, start_state, target_state, start_time):
        self.start_time = start_time
        self.x0 = start_state
        self.xtar = target_state
        self.Xref = np.tile(self.xtar, (self.Np, 1))
        self.get_qp_cost()
        self.get_constraint()
        self.solve_qp()
        xpred_error = np.linalg.norm(self.Xpred - self.xtar.T, axis=1)
        reach_index = xpred_error.size - 1
        for i in range(xpred_error.size):
            if xpred_error[i] < 1e-3:
                reach_index = i
                break
        self.Xpred = np.vstack((start_state.T, self.Xpred[:reach_index+1]))
        self.time_index = np.arange(self.Xpred.shape[0]) * self.ts
        self.duration = self.time_index[-1]
        self.current_progress = 0.0

    def evaluate_pva(self, t_cur):
        quotient, remainder = divmod(t_cur, self.ts)
        index = int(quotient)
        if index >= self.Xpred.shape[0] - 1:
            return self.Xpred[-1, :, np.newaxis]
        else:
            x_select = self.Xpred[index, :, np.newaxis]
            p = x_select[:3] + x_select[3:6] * remainder + 1/2 * x_select[6:] * remainder**2
            v = x_select[3:6] + x_select[6:] * remainder
            a = x_select[6:]
            return np.vstack((p, v, a))


class UniformBspline:
    def __init__(self):
        # These attributes should be available for each trajectory type.
        self.start_time = 0.0
        self.duration = 0.0
        self.current_progress = 0.0
        self.x0 = np.zeros((9, 1))
        self.xtar = np.zeros((9, 1))

        self.point_set = np.zeros((2, 3))  # each row is a way point
        self.ts = 0.0
        self.control_points = np.zeros((self.point_set.shape[0] + 2, 3))  # control_points_num = point_set_num + 2
        self.tip_derivatives = np.zeros((4, 3))  # start_v, end_v, start_a, end_a

        self.p_matrix = np.array([[1, 4, 1, 0],
                                  [-3, 0, 3, 0],
                                  [3, -6, 3, 0],
                                  [-1, 3, -3, 1]]) / 6
        self.v_matrix = np.array([[-1/2, 0, 1/2, 0],
                                  [1, -2, 1, 0],
                                  [-1/2, 3/2, -3/2, 1/2]])
        self.a_matrix = np.array([[1.0, -2, 1, 0],
                                  [-1, 3, -3, 1]])
        self.j_matrix = np.array([[-1, 3, -3, 1]])

    def generate_unifrom_bspline_trajectory(self, start_time, ts, point_set, tip_derivatives):
        point_set_num = point_set.shape[0]
        segments_num = point_set_num + 1
        control_points_num = segments_num + 3
        self.start_time = start_time
        self.ts = ts
        self.point_set = point_set
        self.tip_derivatives = tip_derivatives
        self.duration = segments_num * self.ts
        self.current_progress = 0.0
        self.x0 = np.hstack((point_set[0, :], tip_derivatives[0, :], tip_derivatives[2, :])).reshape(-1, 1)
        self.xtar = np.hstack((point_set[-1, :], tip_derivatives[1, :], tip_derivatives[3, :])).reshape(-1, 1)
        self.v_matrix = np.array([[-1/2, 0, 1/2, 0],
                                  [1, -2, 1, 0],
                                  [-1/2, 3/2, -3/2, 1/2]]) / self.ts
        self.a_matrix = np.array([[1.0, -2, 1, 0],
                                  [-1, 3, -3, 1]]) / self.ts**2

        b = np.vstack((self.point_set, self.tip_derivatives))
        a_mat = np.zeros((control_points_num, control_points_num))
        prow = np.array([1.0, 4, 1]) / 6
        vrow = np.array([-1.0, 0, 1]) / (2 * self.ts)
        arow = np.array([1.0, -2, 1]) / self.ts ** 2

        # specify point_set_num constraints (p0,...,p_end)
        a_mat[0, :3] = prow  # segment[0] starts at p0
        for i in range(2, segments_num - 1):
            a_mat[i-1, i:i+3] = prow  # segment[2->before last one] starts at p1, p2...
        a_mat[segments_num - 2, -3:] = prow  # segment[-1] ends at p_end

        # specify tip_derivatives
        a_mat[point_set_num, :3] = vrow
        a_mat[point_set_num + 1, -3:] = vrow
        a_mat[point_set_num + 2, :3] = arow
        a_mat[point_set_num + 3, -3:] = arow

        q, r = np.linalg.qr(a_mat)
        self.control_points = np.linalg.solve(r, q.T @ b)

    def check_feasibility(self):
        ctps_num = self.control_points.shape[0]
        fea_flag = True

        # 1. Velocity check
        enlarged_vel_lim = traj_para.MAX_V * (1.0 + traj_para.FEA_TOLERANCE) + 1e-4
        max_vel = 0.0
        for i in range(ctps_num - 1):
            vel_ctp_i = 1 / self.ts * (self.control_points[i+1] - self.control_points[i])
            if np.any(np.abs(vel_ctp_i) > enlarged_vel_lim):
                fea_flag = False
                max_vel = max(max_vel, np.max(np.abs(vel_ctp_i)))

        # 2. Accel check
        enlarged_acc_lim = traj_para.MAX_A * (1.0 + traj_para.FEA_TOLERANCE) + 1e-4
        max_acc = 0.0
        for i in range(ctps_num - 2):
            acc_ctp_i = 1 / self.ts**2 * (self.control_points[i+2] - 2 * self.control_points[i+1] +
                                          self.control_points[i])
            if np.any(np.abs(acc_ctp_i) > enlarged_acc_lim):
                fea_flag = False
                max_acc = max(max_acc, np.max(np.abs(acc_ctp_i)))

        # 3. Compute the stretch-time ratio
        ratio = max(max_vel / traj_para.MAX_V, np.sqrt(np.abs(max_acc) / traj_para.MAX_A))

        return fea_flag, ratio

    def evaluate_pva(self, t_cur):
        t_cur = min(t_cur, self.duration)
        quotient, remainder = divmod(t_cur, self.ts)
        index = int(quotient)
        s_t = min(1.0, remainder / self.ts)
        if index == self.point_set.shape[0] + 1:  # when t_cur = self.duration
            index -= 1
            s_t = 1.0
        selected_control_points = self.control_points[index:index+4]
        p_cur = np.array([[1, s_t, s_t**2, s_t**3]]) @ self.p_matrix @ selected_control_points  # res = 2D row_vector
        v_cur = np.array([[1, s_t, s_t**2]]) @ self.v_matrix @ selected_control_points  # res = 2D row_vector
        a_cur = np.array([[1, s_t]]) @ self.a_matrix @ selected_control_points  # res = 2D row_vector
        return np.vstack((p_cur.T, v_cur.T, a_cur.T))

    def evaluate_pvaj(self, t_cur):
        t_cur = min(t_cur, self.duration)
        quotient, remainder = divmod(t_cur, self.ts)
        index = int(quotient)
        s_t = min(1.0, remainder / self.ts)
        if index == self.point_set.shape[0] + 1:  # when t_cur = self.duration
            index -= 1
            s_t = 1.0
        selected_control_points = self.control_points[index:index+4]
        p_cur = np.array([[1, s_t, s_t**2, s_t**3]]) @ self.p_matrix @ selected_control_points  # res = 2D row_vector
        v_cur = np.array([[1, s_t, s_t**2]]) @ self.v_matrix @ selected_control_points  # res = 2D row_vector
        a_cur = np.array([[1, s_t]]) @ self.a_matrix @ selected_control_points  # res = 2D row_vector
        j_matrix = self.j_matrix / self.ts ** 3
        j_cur = j_matrix @ selected_control_points
        return np.vstack((p_cur.T, v_cur.T, a_cur.T, j_cur.T))


class MinisnapTraj:
    """
    Unconstrained BIVP, solve minimum jerk / snap via Mc=b (analytical solution), where M is the selection matrix,
    c is the coefficients for each segment, b is the condition. Each segment is c_0+c_1*t+c_2*t^2...
    Given the waypoints, start and end points higher order terms, and the time index at each point. Make sure the
    intermediate points has certain order of continuity to guarantee the unique solution of Mc=b
    Polynomial degree = 2s-1
    Start / end points information: [s-1]
    Intermediate points continuity: [2s-di-1], where di-1 is the specified intermediate point order information
    s=3: minimum jerk c_0+c_1*t+c_2*t^2...+c_5*t^5
    s=4: minimum snap c_0+c_1*t+c_2*t^2...+c_7*t^7
    """

    def __init__(self, waypoints, ho_start, ho_end, s, time_idx):
        """
        :param waypoints: 2D numpy array row instance
        :param ho_start: 2D numpy array row instance
        :param ho_end: 2D numpy array row instance
        :param s: int >= 2
        :param time_idx: 1D numpy array (ascending)
        """
        self.start_time = time_idx[0]
        self.waypoints = waypoints
        self.s = s
        self.ho_start = ho_start
        self.ho_end = ho_end
        self.time_idx = time_idx
        self.feasibility_check()
        self.coefficient_list = self.get_coefficient()

    def feasibility_check(self):
        if self.waypoints.shape[0] < 2:
            raise Exception("Waypoints not enough")
        if self.s < 2:
            raise Exception("Minimum order is not fulfilled!")
        self.ho_start = self.check_tip_cond(self.ho_start)
        self.ho_end = self.check_tip_cond(self.ho_end)
        if self.time_idx.size != self.waypoints.shape[0]:
            raise Exception("Time index does not match with waypoints number!")
        else:
            self.time_idx -= self.time_idx[0]

    def check_tip_cond(self, tip_cond):
        if tip_cond.shape[0] >= self.s - 1:
            return tip_cond[:self.s - 1, :]
        else:
            compen = np.zeros((self.s - 1 - tip_cond.shape[0], tip_cond.shape[1]))
            return np.vstack((tip_cond, compen))

    def get_coefficient(self):
        f0 = np.zeros((self.s, self.s * 2))
        diag = np.diag([math.factorial(i) for i in range(self.s)])
        f0[:, :self.s] = diag
        t_end = self.time_idx[-1] - self.time_idx[-2]
        em = self.poly_matrix(t_end)[:self.s, :]
        row_mid = []
        for i in range(1, self.waypoints.shape[0] - 1):
            ti = self.time_idx[i] - self.time_idx[i - 1]
            ei = self.poly_matrix(ti)[:self.s * 2 - 1]
            ei = np.vstack((ei[0, :], ei))
            fi = -self.poly_matrix(0)[:self.s * 2 - 1]
            fi = np.vstack((np.zeros(self.s * 2), fi))
            row_i = np.zeros((ei.shape[0], (self.waypoints.shape[0] - 1) * ei.shape[1]))
            row_i[:, (i - 1) * ei.shape[1]:(i + 1) * ei.shape[1]] = np.hstack((ei, fi))
            row_mid.append(row_i)
        row_0 = np.zeros((f0.shape[0], (self.waypoints.shape[0] - 1) * f0.shape[1]))
        row_0[:, :f0.shape[1]] = f0
        row_end = np.zeros((em.shape[0], (self.waypoints.shape[0] - 1) * em.shape[1]))
        row_end[:, -em.shape[1]:] = em
        if not row_mid:  # row_mid is empty
            m_mat = np.vstack((row_0, row_end))
        else:
            row_mid = np.vstack(row_mid)
            m_mat = np.vstack((row_0, row_mid, row_end))

        b = []
        for i in range(self.waypoints.shape[0]):
            if i == 0:
                bi = np.vstack((self.waypoints[i, :], self.ho_start))
            elif i == self.waypoints.shape[0] - 1:
                bi = np.vstack((self.waypoints[i, :], self.ho_end))
            else:
                bi = np.vstack((self.waypoints[i, :], np.zeros((self.s * 2 - 1, self.waypoints.shape[1]))))
            b.append(bi)
        b = np.vstack(b)

        c = np.linalg.inv(m_mat) @ b
        coefficient_list = []
        for i in range(self.waypoints.shape[0] - 1):
            c_i = c[i * self.s * 2:(i + 1) * self.s * 2, :]
            coefficient_list.append(c_i)
        return coefficient_list

    def poly_matrix(self, t):
        e = np.zeros((self.s * 2, self.s * 2))
        for i in range(self.s * 2):
            for j in range(self.s * 2):
                if j >= i:
                    coefficient = math.factorial(j) / math.factorial(j - i)
                    e[i, j] = coefficient * (t ** (j - i))
                else:
                    e[i, j] = 0
        return e

    def get_pos(self, tc):
        tc = np.maximum(np.minimum(tc, self.time_idx[-1]), 0)
        idx = 0
        for i in range(1, self.time_idx.size):
            if tc <= self.time_idx[i]:
                idx = i - 1
                break
        delta_t = tc - self.time_idx[idx]
        c = self.coefficient_list[idx]
        st = np.array([[delta_t ** i for i in range(self.s * 2)]])
        return st @ c
