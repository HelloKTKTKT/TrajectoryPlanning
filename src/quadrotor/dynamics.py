import numpy as np
import src.quadrotor.parameters as quad_para
import src.utils.func as func


def quadrotor_dynamics(xin, uin):
    """
    :param xin: 2D column numpy array [Np, Nv, euler, Bw].T
    :param uin: 2D column numpy array [F1, F2, F3, F4].T
    :return: 2D column numpy array
    """
    xin.shape = (-1, 1)
    uin.shape = (-1, 1)

    u_pseudo = quad_para.rforce2pseudo @ uin
    ft = u_pseudo[0, 0]
    tau = u_pseudo[1:]
    n_p, nv, euler, bw = xin[:3], xin[3:6], xin[6:9], xin[9:12]
    q = func.euler_zyx2quat(eul=euler)

    nf = func.q_rotate(q, np.array([[0.0, 0.0, ft]]).T)
    na = nf / quad_para.mass + np.array([[0.0, 0.0, -9.81]]).T

    phi, theta, psi = euler.flatten()
    rm_angular_rate = np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                                [0, np.cos(phi), -np.sin(phi)],
                                [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])
    d_euler = rm_angular_rate @ bw

    d_bw = quad_para.inertia_inv @ tau - quad_para.inertia_inv @ func.cross(bw, quad_para.inertia_matrix @ bw)

    xdot = np.vstack((nv, na, d_euler, d_bw))

    return xdot


def rk4(xin, uin, dt, dynamics=quadrotor_dynamics):
    k1 = dt * dynamics(xin, uin)
    k2 = dt * dynamics(xin + 0.5 * k1, uin)
    k3 = dt * dynamics(xin + 0.5 * k2, uin)
    k4 = dt * dynamics(xin + k3, uin)
    xnext = xin + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return xnext
