from __future__ import annotations

import numpy as np

from trajplan.quadrotor.math_utils import cross, euler_zyx2quat, q_rotate
from trajplan.quadrotor.model import QuadrotorPhysicalConfig
from trajplan.quadrotor.state import QuadrotorState
from trajplan.shared_types import Vector


def quadrotor_dynamics(
    xin: Vector,
    rotor_thrust: Vector,
    physical_config: QuadrotorPhysicalConfig,
) -> Vector:
    """
    Continuous-time quadrotor dynamics.

    State convention
    ----------------
    xin = [position(3), velocity(3), euler(3), body_rate(3)]
        shape (12,)

    Input convention
    ----------------
    rotor_thrust = [F1, F2, F3, F4]
        shape (4,)

    Returns
    -------
    xdot : ndarray, shape (12,)
        [p_dot, v_dot, euler_dot, body_rate_dot]
    """
    xin = np.asarray(xin, dtype=np.float64).reshape(-1)
    rotor_thrust = np.asarray(rotor_thrust, dtype=np.float64).reshape(-1)

    if xin.shape != (12,):
        raise ValueError(f"Expected xin shape (12,), but got {xin.shape}.")
    if rotor_thrust.shape != (physical_config.rotor_count,):
        raise ValueError(
            "Expected rotor_thrust shape "
            f"({physical_config.rotor_count},), but got {rotor_thrust.shape}."
        )

    # position = xin[0:3]
    velocity = xin[3:6]
    euler = xin[6:9]
    body_rate = xin[9:12]

    u_pseudo = physical_config.rforce2pseudo @ rotor_thrust.reshape(-1, 1)
    total_thrust = float(u_pseudo[0, 0])
    torque = u_pseudo[1:4, 0]

    q = euler_zyx2quat(euler)

    thrust_world = q_rotate(
        q,
        np.array([0.0, 0.0, total_thrust], dtype=np.float64),
    )
    acceleration_world = thrust_world / physical_config.mass + np.array(
        [0.0, 0.0, -9.81], dtype=np.float64
    )

    phi, theta, _ = euler
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_theta = np.tan(theta)
    cos_theta = np.cos(theta)

    if np.abs(cos_theta) < 1e-8:
        raise ValueError(
            "Euler angle theta is too close to +/- pi/2, "
            "causing a singularity in ZYX Euler kinematics."
        )

    transform = np.array(
        [
            [1.0, sin_phi * tan_theta, cos_phi * tan_theta],
            [0.0, cos_phi, -sin_phi],
            [0.0, sin_phi / cos_theta, cos_phi / cos_theta],
        ],
        dtype=np.float64,
    )
    euler_dot = transform @ body_rate

    body_rate_dot = (
        physical_config.inertia_inv @ torque
        - physical_config.inertia_inv
        @ cross(body_rate, physical_config.inertia_matrix @ body_rate)
    )

    p_dot = velocity
    v_dot = acceleration_world

    xdot = np.hstack((p_dot, v_dot, euler_dot, body_rate_dot))
    return xdot


def rk4_step(
    xk: Vector,
    uk: Vector,
    ts: float,
    physical_config: QuadrotorPhysicalConfig,
) -> Vector:
    """
    4th-order Runge-Kutta integration step.
    """
    xk = np.asarray(xk, dtype=np.float64).reshape(-1)
    uk = np.asarray(uk, dtype=np.float64).reshape(-1)
    ts = float(ts)

    if xk.shape != (12,):
        raise ValueError(f"Expected xk shape (12,), but got {xk.shape}.")
    if uk.shape != (physical_config.rotor_count,):
        raise ValueError(
            f"Expected uk shape ({physical_config.rotor_count},), but got {uk.shape}."
        )
    if ts <= 0.0:
        raise ValueError(f"Expected ts > 0, but got {ts}.")

    k1 = quadrotor_dynamics(xk, uk, physical_config)
    k2 = quadrotor_dynamics(xk + 0.5 * ts * k1, uk, physical_config)
    k3 = quadrotor_dynamics(xk + 0.5 * ts * k2, uk, physical_config)
    k4 = quadrotor_dynamics(xk + ts * k3, uk, physical_config)

    x_next = xk + (ts / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x_next


def first_euler_step(
    xk: Vector,
    uk: Vector,
    ts: float,
    physical_config: QuadrotorPhysicalConfig,
) -> Vector:
    """
    Forward-Euler integration step.
    """
    xk = np.asarray(xk, dtype=np.float64).reshape(-1)
    uk = np.asarray(uk, dtype=np.float64).reshape(-1)
    ts = float(ts)

    if xk.shape != (12,):
        raise ValueError(f"Expected xk shape (12,), but got {xk.shape}.")
    if uk.shape != (physical_config.rotor_count,):
        raise ValueError(
            f"Expected uk shape ({physical_config.rotor_count},), but got {uk.shape}."
        )
    if ts <= 0.0:
        raise ValueError(f"Expected ts > 0, but got {ts}.")

    xdot = quadrotor_dynamics(xk, uk, physical_config)
    x_next = xk + ts * xdot
    return x_next


def state_to_dynamics_vector(
    state: QuadrotorState,
) -> Vector:
    """
    Convert QuadrotorState to the 12D dynamics state vector.

    Output convention
    -----------------
    x = [position(3), velocity(3), euler(3), body_rate(3)]
    """
    return np.hstack(
        (
            np.asarray(state.position, dtype=np.float64).reshape(-1),
            np.asarray(state.velocity, dtype=np.float64).reshape(-1),
            np.asarray(state.euler, dtype=np.float64).reshape(-1),
            np.asarray(state.angular_rate, dtype=np.float64).reshape(-1),
        )
    )


def dynamics_vector_to_state(
    xin: Vector,
    rotor_thrust: Vector,
    physical_config: QuadrotorPhysicalConfig,
) -> QuadrotorState:
    """
    Convert a 12D dynamics state vector back to QuadrotorState.

    The acceleration field in pva is derived from the current state/input pair:
        acceleration = xdot[3:6]
    """
    xin = np.asarray(xin, dtype=np.float64).reshape(-1)
    if xin.shape != (12,):
        raise ValueError(f"Expected xin shape (12,), but got {xin.shape}.")

    xdot = quadrotor_dynamics(
        xin=xin,
        rotor_thrust=rotor_thrust,
        physical_config=physical_config,
    )
    acceleration = xdot[3:6]

    position = xin[0:3]
    velocity = xin[3:6]
    euler = xin[6:9]
    body_rate = xin[9:12]

    return QuadrotorState(
        pva=np.hstack((position, velocity, acceleration)),
        euler=euler,
        angular_rate=body_rate,
    )
