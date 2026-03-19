from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from trajplan.trajectory.bspline import UniformBSpline


def main() -> None:
    # Visible waypoints in 3D.
    # For easy plotting, keep z = 0 so the trajectory lies in a 2D plane.
    waypoints = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.5, 0.0],
            [3.0, 2.0, 0.0],
            [5.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )

    # Boundary derivatives:
    # [start_velocity, end_velocity, start_acceleration, end_acceleration]
    boundary_derivatives = np.zeros((4, 3), dtype=np.float64)

    delta_t = 0.2

    traj = UniformBSpline.from_point_set(
        point_set=waypoints,
        delta_t=delta_t,
        boundary_derivatives=boundary_derivatives,
    )

    print("=== UniformBSpline Debug Info ===")
    print(f"num_control_points = {traj.num_control_points}")
    print(f"num_segments       = {traj.num_segments}")
    print(f"delta_t            = {traj.delta_t:.4f}")
    print(f"duration           = {traj.duration:.4f}")
    print("control_points:")
    print(traj.control_points)
    print()

    max_velocity = 10.0
    max_acceleration = 20.0
    feasible_flag, required_ratio = traj.check_feasibility(
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        tolerance=0.0,
    )

    print("=== Feasibility Check ===")
    print(f"max_velocity       = {max_velocity}")
    print(f"max_acceleration   = {max_acceleration}")
    print(f"feasible_flag      = {feasible_flag}")
    print(f"required_ratio     = {required_ratio:.6f}")
    print()

    sample_dt = 0.01
    times = np.arange(0.0, traj.duration + sample_dt, sample_dt, dtype=np.float64)
    times[-1] = traj.duration

    positions = np.vstack([traj.evaluate_pva(t)[:3] for t in times])
    velocities = np.vstack([traj.evaluate_pva(t)[3:6] for t in times])
    accelerations = np.vstack([traj.evaluate_pva(t)[6:9] for t in times])

    print("=== Endpoint Check ===")
    print("start position:", positions[0])
    print("end position:  ", positions[-1])
    print("start velocity:", velocities[0])
    print("end velocity:  ", velocities[-1])
    print("start accel:   ", accelerations[0])
    print("end accel:     ", accelerations[-1])
    print()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the spline trajectory.
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        linewidth=2.0,
        label="B-spline trajectory",
    )

    # Plot visible waypoints.
    ax.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        s=80,
        marker="o",
        label="Visible waypoints",
    )

    # Plot control points and control polygon for debugging.
    ax.plot(
        traj.control_points[:, 0],
        traj.control_points[:, 1],
        "--",
        linewidth=1.2,
        label="Control polygon",
    )
    ax.scatter(
        traj.control_points[:, 0],
        traj.control_points[:, 1],
        s=40,
        marker="x",
        label="Control points",
    )

    # Annotate waypoint indices.
    for i, point in enumerate(waypoints):
        ax.annotate(
            f"W{i}",
            (point[0], point[1]),
            textcoords="offset points",
            xytext=(6, 6),
        )

    ax.set_title("Uniform Cubic B-spline Debug Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
