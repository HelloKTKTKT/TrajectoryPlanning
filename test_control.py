import numpy as np
from src.quadrotor.controller import DifferentialFlatness
from src.quadrotor.dynamics import rk4 as quad_dynamics


def generate_circular_trajectory(total_time=10.0,
                                 dt=0.05,
                                 radius=5.0,
                                 height=3.0,
                                 linear_speed=5.0):
    """
    Returns an (N,9) array of [p,v,a] for a horizontal circle:
      - Radius = `radius`, center at (0,0)
      - Starts at (radius, 0, height), CCW at speed `linear_speed`
      - Constant-altitude z = `height`
      - total_time seconds, dt between samples
    """
    # Angular speed (rad/s)
    omega = linear_speed / radius

    # Time stamps
    t = np.arange(0, total_time, dt)  # shape (N,)

    # Position
    px =  radius * np.cos(omega * t)
    py =  radius * np.sin(omega * t)
    pz =  np.full_like(t, height)

    # Velocity (derivative of position)
    vx = -radius * omega * np.sin(omega * t)
    vy =  radius * omega * np.cos(omega * t)
    vz =  np.zeros_like(t)

    # Acceleration (derivative of velocity)
    ax = -radius * omega**2 * np.cos(omega * t)
    ay = -radius * omega**2 * np.sin(omega * t)
    az =  np.zeros_like(t)

    # Stack into (N,9)
    ref_traj = np.vstack((px, py, pz,
                          vx, vy, vz,
                          ax, ay, az)).T
    return ref_traj


def control_performance_test(xref_pva_log):
    controller = DifferentialFlatness()
    x_init = np.zeros((12, 1))
    x_init[:6] = xref_pva_log[0, :6].reshape(3, 1)
    u_init = np.ones((4, 1)) * (9.81 / 4)
    xlog = [x_init]
    ulog = [u_init]
    for i in range(1, xref_pva_log.shape[0]):
        xref_pva = xref_pva_log[i]
        xref_flat = np.zeros((4, 5))
        xref_flat[:3, :3] = xref_pva.reshape((3, 3), order='F')

        u_apply = controller.execute(xlog[-1], ulog[-1], xref_flat)
        xnext = quad_dynamics(xlog[-1], u_apply, 0.05)
        xlog.append(xnext)
        ulog.append(u_apply)
    return xlog, ulog
