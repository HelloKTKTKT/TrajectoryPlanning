from __future__ import annotations

import numpy as np

from trajplan.shared_types import Vector


def cross(
    arr1: Vector,
    arr2: Vector,
) -> Vector:
    arr1 = np.asarray(arr1, dtype=np.float64).reshape(-1)
    arr2 = np.asarray(arr2, dtype=np.float64).reshape(-1)

    if arr1.shape != (3,):
        raise ValueError(f"Expected arr1 shape (3,), but got {arr1.shape}.")
    if arr2.shape != (3,):
        raise ValueError(f"Expected arr2 shape (3,), but got {arr2.shape}.")

    return np.cross(arr1, arr2)


def q_dot(
    q1: Vector,
    q2: Vector,
) -> Vector:
    q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
    q2 = np.asarray(q2, dtype=np.float64).reshape(-1)

    if q1.shape != (4,):
        raise ValueError(f"Expected q1 shape (4,), but got {q1.shape}.")
    if q2.shape != (4,):
        raise ValueError(f"Expected q2 shape (4,), but got {q2.shape}.")

    q1w = q1[0]
    q1v = q1[1:4]
    q2w = q2[0]
    q2v = q2[1:4]

    qout_w = q1w * q2w - np.dot(q1v, q2v)
    qout_v = q1w * q2v + q2w * q1v + cross(q1v, q2v)

    return np.hstack((qout_w, qout_v))


def q_rotate(
    q: Vector,
    vec: Vector,
) -> Vector:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)

    if q.shape != (4,):
        raise ValueError(f"Expected q shape (4,), but got {q.shape}.")
    if vec.shape != (3,):
        raise ValueError(f"Expected vec shape (3,), but got {vec.shape}.")

    q_conj = q * np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    vec_q = np.hstack((0.0, vec))
    vec_rotated = q_dot(q_dot(q, vec_q), q_conj)[1:4]
    return vec_rotated


def euler_zyx2quat(
    eul: Vector,
) -> Vector:
    eul = np.asarray(eul, dtype=np.float64).reshape(-1)

    if eul.shape != (3,):
        raise ValueError(f"Expected eul shape (3,), but got {eul.shape}.")

    phi, theta, psi = eul

    qw = np.cos(phi / 2.0) * np.cos(theta / 2.0) * np.cos(psi / 2.0) + np.sin(
        phi / 2.0
    ) * np.sin(theta / 2.0) * np.sin(psi / 2.0)
    qx = np.sin(phi / 2.0) * np.cos(theta / 2.0) * np.cos(psi / 2.0) - np.cos(
        phi / 2.0
    ) * np.sin(theta / 2.0) * np.sin(psi / 2.0)
    qy = np.cos(phi / 2.0) * np.sin(theta / 2.0) * np.cos(psi / 2.0) + np.sin(
        phi / 2.0
    ) * np.cos(theta / 2.0) * np.sin(psi / 2.0)
    qz = np.cos(phi / 2.0) * np.cos(theta / 2.0) * np.sin(psi / 2.0) - np.sin(
        phi / 2.0
    ) * np.sin(theta / 2.0) * np.cos(psi / 2.0)

    return np.array([qw, qx, qy, qz], dtype=np.float64)


def vec_scale(
    vec: Vector,
    vmax: float,
) -> Vector:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    vmax = float(vmax)

    if vmax < 0.0:
        raise ValueError(f"Expected vmax >= 0, but got {vmax}.")

    vec_abs_max = np.max(np.abs(vec))
    if vec_abs_max <= vmax or vec_abs_max < 1e-12:
        return vec

    return vec * vmax / vec_abs_max
