import numpy as np


def cross(arr1, arr2):
    x1, y1, z1 = arr1.flatten()
    x2, y2, z2 = arr2.flatten()
    cross_product = np.array([
        y1 * z2 - z1 * y2,
        z1 * x2 - x1 * z2,
        x1 * y2 - y1 * x2
    ])
    return cross_product.reshape(3, 1)


def q_dot(q1, q2):
    q1 = q1.reshape(q1.size, 1)
    q2 = q2.reshape(q2.size, 1)
    q1w, q1v = q1[0, 0], q1[1:4]
    q2w, q2v = q2[0, 0], q2[1:4]
    qout_w = q1w * q2w - q1v.T @ q2v  # the result is a 2-D np array
    qout_v = q1w * q2v + q2w * q1v + cross(q1v, q2v)
    qout = np.vstack((qout_w, qout_v))
    return qout


def q_rotate(q, vec):
    q = q.reshape(q.size, 1)
    q_conj = q * np.array([[1.0, -1, -1, -1]]).T
    vec = vec.reshape(vec.size, 1)
    vec_q = np.vstack((np.array([[0.0]]), vec))
    vec_rotate = q_dot(q_dot(q, vec_q), q_conj)[1:]
    return vec_rotate


def euler_zyx2quat(eul):
    eul = eul.reshape(eul.size, 1)
    phi, theta, psi = eul[0, 0], eul[1, 0], eul[2, 0]
    qw = np.cos(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qx = np.sin(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) - np.cos(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qy = np.cos(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2)
    qz = np.cos(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2) - np.sin(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2)
    q = np.array([[qw, qx, qy, qz]]).T
    return q


def vec_scale(vec, vmax):
    vec_abs_max = np.max(abs(vec))
    vec_s = vec if vec_abs_max < vmax else vec * vmax / vec_abs_max
    return vec_s
