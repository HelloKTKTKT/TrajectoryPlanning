#!/usr/bin/env python

from pathlib import Path

import numpy as np
from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory


def normalize(v):
    norm = np.linalg.norm(v)
    assert norm > 0
    return v / norm


def generate_circle(t, r=1.0, v=1.0):
    omega = v / r
    px = r * np.cos(omega * t)
    py = r * np.sin(omega * t)
    pz = 0.0
    p = np.array([px, py, pz])

    vx = -r * omega * np.sin(omega * t)
    vy = r * omega * np.cos(omega * t)
    vz = 0.0
    vel = np.array([vx, vy, vz])

    ax = -r * omega**2 * np.cos(omega * t)
    ay = -r * omega**2 * np.sin(omega * t)
    az = 0.0
    acc = np.array([ax, ay, az])

    # Calculate jerk (3rd derivative of position)
    jx = r * omega**3 * np.sin(omega * t)
    jy = -r * omega**3 * np.cos(omega * t)
    jz = 0.0
    jerk = np.array([jx, jy, jz])

    # Calculate thrust (acceleration + gravity)
    thrust = acc + np.array([0, 0, 9.81])

    # Calculate body z-axis (normalized thrust direction)
    z_body = normalize(thrust)

    # Calculate yaw (tangent to circle, pointing in direction of motion)
    yaw = 0.0  # np.arctan2(vy, vx)

    # Calculate world x-axis from yaw
    x_world = np.array([np.cos(yaw), np.sin(yaw), 0])

    # Calculate body y-axis (perpendicular to z_body and x_world)
    y_body = normalize(np.cross(z_body, x_world))

    # Calculate body x-axis (perpendicular to y_body and z_body)
    x_body = np.cross(y_body, z_body)

    # Calculate omega using the same method as uav_trajectory.py
    jerk_orth_zbody = jerk - (np.dot(jerk, z_body) * z_body)
    h_w = jerk_orth_zbody / np.linalg.norm(thrust)

    omega_vec = np.array([-np.dot(h_w, y_body), np.dot(h_w, x_body), z_body[2] * omega])

    return p, vel, acc, yaw, omega_vec


def executeCircleTraj(
    timeHelper,
    cf,
    radius=1.0,
    velocity=1.0,
    duration=10.0,
    rate=100,
    offset=np.zeros(3),
):
    """Execute circular trajectory using cmdFullState"""
    # Frequency tracking variables
    cmd_count = 0
    last_log_time = timeHelper.time()
    log_interval = 1.0  # Log every 1 second

    print(f"Starting circle trajectory execution at target rate: {rate} Hz")
    print(
        f"Circle parameters: radius={radius}m, velocity={velocity}m/s, duration={duration}s"
    )

    start_time = timeHelper.time()
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        if t > duration:
            break

        # Generate circle trajectory
        pos, vel, acc, yaw, omega = generate_circle(t, radius, velocity)

        # Send full state command
        cf.cmdFullState(
            pos + np.array(cf.initialPosition) + offset, vel, acc, yaw, omega
        )

        cmd_count += 1
        current_time = timeHelper.time()

        # Log frequency every log_interval seconds
        if current_time - last_log_time >= log_interval:
            elapsed = current_time - last_log_time
            actual_rate = cmd_count / elapsed
            print(
                f"cmdFullState frequency: {actual_rate:.1f} Hz (target: {rate} Hz, commands sent: {cmd_count})"
            )
            cmd_count = 0
            last_log_time = current_time

        timeHelper.sleepForRate(rate)

    total_time = timeHelper.time() - start_time
    print(
        f"Circle trajectory execution completed. Total time: {total_time:.2f} seconds"
    )


def load_trajectory_from_csv(filepath, dt=0.01):
    """Load trajectory data from CSV file.

    Args:
        filepath: Path to CSV file
        dt: Time step between waypoints in seconds (default: 0.01 = 100Hz)
            Can also be a string path to load from file

    CSV Format:
        9 columns: px, py, pz, vx, vy, vz, ax, ay, az
        No header row expected

    Returns:
        List of tuples (delta_t, pos, vel, acc) suitable for executeCustomTrajectory
    """
    import csv

    trajectory_data = []

    try:
        with open(filepath, "r") as f:
            reader = csv.reader(f)

            for row in reader:
                # Skip empty rows or comments
                if not row or row[0].strip().startswith("#"):
                    continue

                # Parse the 9 columns
                if len(row) < 9:
                    print(f"Warning: Skipping row with insufficient columns: {row}")
                    continue

                try:
                    px, py, pz, vx, vy, vz, ax, ay, az = [float(x) for x in row[:9]]

                    pos = np.array([px, py, pz])
                    vel = np.array([vx, vy, vz])
                    acc = np.array([ax, ay, az])

                    trajectory_data.append((dt, pos, vel, acc))

                except ValueError as e:
                    print(f"Warning: Skipping row with invalid data: {row} - {e}")
                    continue

        print(f"Loaded {len(trajectory_data)} waypoints from {filepath}")
        return trajectory_data

    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return []
    except Exception as e:
        print(f"Error loading trajectory file: {e}")
        return []


def calculate_omega_from_trajectory(pos, vel, acc, jerk, yaw=0.0):
    """Calculate omega (angular velocity) from trajectory state.

    Args:
        pos: position [x, y, z]
        vel: velocity [vx, vy, vz]
        acc: acceleration [ax, ay, az]
        jerk: jerk [jx, jy, jz] (derivative of acceleration)
        yaw: yaw angle in radians (default: 0.0)

    Returns:
        omega: angular velocity [wx, wy, wz] in body frame
    """
    # Calculate thrust (acceleration + gravity)
    thrust = acc + np.array([0, 0, 9.81])

    # Calculate body z-axis (normalized thrust direction)
    thrust_norm = np.linalg.norm(thrust)
    if thrust_norm < 1e-6:
        # If thrust is too small, use default up direction
        z_body = np.array([0, 0, 1])
    else:
        z_body = thrust / thrust_norm

    # Calculate world x-axis from yaw
    x_world = np.array([np.cos(yaw), np.sin(yaw), 0])

    # Calculate body y-axis (perpendicular to z_body and x_world)
    y_body_unnorm = np.cross(z_body, x_world)
    y_body_norm = np.linalg.norm(y_body_unnorm)
    if y_body_norm < 1e-6:
        # Handle singularity when z_body is parallel to world z
        y_body = np.array([0, 1, 0])
    else:
        y_body = y_body_unnorm / y_body_norm

    # Calculate body x-axis (perpendicular to y_body and z_body)
    x_body = np.cross(y_body, z_body)

    # Calculate omega using the same method as in generate_circle
    jerk_orth_zbody = jerk - (np.dot(jerk, z_body) * z_body)

    if thrust_norm < 1e-6:
        h_w = np.zeros(3)
    else:
        h_w = jerk_orth_zbody / thrust_norm

    # Calculate yaw rate
    yaw_rate = 0.0  # Assuming constant yaw for now

    omega_vec = np.array([-np.dot(h_w, y_body), np.dot(h_w, x_body), yaw_rate])

    return omega_vec


def executeCustomTrajectory(
    timeHelper,
    cf,
    trajectory_data,
    rate=100,
    offset=np.zeros(3),
):
    """Execute trajectory from a list of waypoints using cmdFullState.

    Args:
        timeHelper: Crazyswarm time helper
        cf: Crazyflie object
        trajectory_data: List of tuples (delta_t, pos, vel, acc)
            - delta_t: time delta from previous waypoint (seconds)
            - pos: position [x, y, z] (meters)
            - vel: velocity [vx, vy, vz] (m/s)
            - acc: acceleration [ax, ay, az] (m/s^2)
        rate: Control rate in Hz (default: 100)
        offset: Position offset [x, y, z] (meters)
    """
    if not trajectory_data:
        print("Error: Empty trajectory data")
        return

    # Frequency tracking variables
    cmd_count = 0
    last_log_time = timeHelper.time()
    log_interval = 1.0  # Log every 1 second

    print(f"Starting custom trajectory execution at target rate: {rate} Hz")
    print(f"Trajectory waypoints: {len(trajectory_data)}")

    # Calculate jerk numerically for each waypoint
    trajectory_with_jerk = []
    for i in range(len(trajectory_data)):
        delta_t, pos, vel, acc = trajectory_data[i]

        # Calculate jerk by numerical differentiation
        if i < len(trajectory_data) - 1:
            # Forward difference
            _, _, _, acc_next = trajectory_data[i + 1]
            dt = trajectory_data[i + 1][0]
            if dt > 0:
                jerk = (acc_next - acc) / dt
            else:
                jerk = np.zeros(3)
        elif i > 0:
            # Backward difference
            _, _, _, acc_prev = trajectory_data[i - 1]
            dt = delta_t
            if dt > 0:
                jerk = (acc - acc_prev) / dt
            else:
                jerk = np.zeros(3)
        else:
            jerk = np.zeros(3)

        trajectory_with_jerk.append((delta_t, pos, vel, acc, jerk))

    # Execute trajectory
    start_time = timeHelper.time()
    waypoint_idx = 0
    cumulative_time = 0.0

    while not timeHelper.isShutdown() and waypoint_idx < len(trajectory_with_jerk):
        t = timeHelper.time() - start_time
        if t > 15.0:
            break

        # Find current waypoint based on cumulative time
        while waypoint_idx < len(trajectory_with_jerk) - 1:
            next_cumulative_time = (
                cumulative_time + trajectory_with_jerk[waypoint_idx + 1][0]
            )
            if t < next_cumulative_time:
                break
            waypoint_idx += 1
            cumulative_time = next_cumulative_time

        # Get current waypoint data
        delta_t, pos, vel, acc, jerk = trajectory_with_jerk[waypoint_idx]

        # Calculate omega assuming yaw = 0
        yaw = 0.0
        omega = calculate_omega_from_trajectory(pos, vel, acc, jerk, yaw)

        # Send full state command
        cf.cmdFullState(
            pos + np.array(cf.initialPosition) + offset, vel, acc, yaw, omega
        )

        cmd_count += 1
        current_time = timeHelper.time()

        # Log frequency every log_interval seconds
        if current_time - last_log_time >= log_interval:
            elapsed = current_time - last_log_time
            actual_rate = cmd_count / elapsed
            print(
                f"cmdFullState frequency: {actual_rate:.1f} Hz (target: {rate} Hz, "
                f"waypoint: {waypoint_idx + 1}/{len(trajectory_with_jerk)})"
            )
            cmd_count = 0
            last_log_time = current_time

        timeHelper.sleepForRate(rate)
        if waypoint_idx + 1 == len(trajectory_with_jerk):
            break
    total_time = timeHelper.time() - start_time
    print(
        f"Custom trajectory execution completed. Total time: {total_time:.2f} seconds"
    )


def executeTrajectory(timeHelper, cf, trajpath, rate=100, offset=np.zeros(3)):
    traj = Trajectory()
    traj.loadcsv(trajpath)

    # Frequency tracking variables
    cmd_count = 0
    last_log_time = timeHelper.time()
    log_interval = 1.0  # Log every 1 second

    print(f"Starting trajectory execution at target rate: {rate} Hz")

    start_time = timeHelper.time()
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        if t > traj.duration:
            break

        e = traj.eval(t)
        cf.cmdFullState(
            e.pos + np.array(cf.initialPosition) + offset, e.vel, e.acc, e.yaw, e.omega
        )
        print(e.pos.size())

        cmd_count += 1
        current_time = timeHelper.time()

        # Log frequency every log_interval seconds
        if current_time - last_log_time >= log_interval:
            elapsed = current_time - last_log_time
            actual_rate = cmd_count / elapsed
            print(
                f"cmdFullState frequency: {actual_rate:.1f} Hz (target: {rate} Hz, commands sent: {cmd_count})"
            )
            cmd_count = 0
            last_log_time = current_time

        timeHelper.sleepForRate(rate)

    total_time = timeHelper.time() - start_time
    print(f"Trajectory execution completed. Total time: {total_time:.2f} seconds")


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    rate = 100.0
    Z = 0.5

    # Circle parameters
    radius = 1.0
    velocity = 1.0
    duration = 10.0
    offset = np.array([0, 0, 0.5])

    cf.takeoff(targetHeight=Z, duration=Z + 1.0)
    timeHelper.sleep(Z + 2.0)

    # Calculate initial circle position (starting at angle 0)
    initial_pos, initial_vel, initial_acc, initial_yaw, initial_omega = generate_circle(
        0, radius, velocity
    )
    target_pos = initial_pos + np.array(cf.initialPosition) + offset

    print(f"Flying to initial circle position: {target_pos}")
    cf.goTo(target_pos, yaw=initial_yaw, duration=3.0)
    timeHelper.sleep(3.5)

    executeCircleTraj(
        timeHelper,
        cf,
        radius=radius,
        velocity=velocity,
        duration=duration,
        rate=rate,
        offset=offset,
    )

    cf.notifySetpointsStop()
    cf.land(targetHeight=0.03, duration=Z + 1.0)
    timeHelper.sleep(Z + 2.0)


def example_custom_trajectory(csv_filepath=None):
    """Example usage of executeCustomTrajectory with custom waypoints.

    Args:
        csv_filepath: Optional path to CSV file with trajectory data
                     If None, generates a simple example trajectory
    """
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    Z = 1.0
    offset = np.array([0.2, 0.0, -0.5])

    # Takeoff
    cf.takeoff(targetHeight=Z, duration=Z + 1.0)
    timeHelper.sleep(Z + 2.0)

    # Load or generate trajectory data
    if csv_filepath is not None:
        # Option 1: Load from CSV file
        # CSV format: 9 columns (px, py, pz, vx, vy, vz, ax, ay, az)
        print(f"Loading trajectory from CSV: {csv_filepath}")
        trajectory_data = load_trajectory_from_csv(csv_filepath, dt=0.01)

        print("Flying to initial circle position ")
        cf.goTo([1.4, -1.2, Z], yaw=0.0, duration=3.0)
        timeHelper.sleep(3.5)
        if not trajectory_data:
            print("Failed to load trajectory, landing...")
            cf.land(targetHeight=0.03, duration=Z + 1.0)
            timeHelper.sleep(Z + 2.0)
            return
    else:
        # Option 2: Generate trajectory programmatically
        # Create example trajectory data: a simple straight line with constant velocity
        # Format: List of (delta_t, position, velocity, acceleration)
        trajectory_data = []

        # Example: Move in a straight line along x-axis for 5 seconds
        num_waypoints = 50
        duration = 5.0
        distance = 1.0  # 1 meter

        for i in range(num_waypoints):
            t = i * duration / num_waypoints
            delta_t = duration / num_waypoints

            # Linear motion along x
            pos = np.array([distance * t / duration, 0.0, 0.0])
            vel = np.array([distance / duration, 0.0, 0.0])
            acc = np.array([0.0, 0.0, 0.0])

            trajectory_data.append((delta_t, pos, vel, acc))

    # Execute the custom trajectory
    executeCustomTrajectory(
        timeHelper,
        cf,
        trajectory_data,
        rate=100,
        offset=offset,
    )

    cf.notifySetpointsStop()
    cf.land(targetHeight=0.03, duration=Z + 1.0)
    timeHelper.sleep(Z + 2.0)


if __name__ == "__main__":
    # Run the default circle trajectory example
    # main()

    # Uncomment to run custom trajectory example (programmatically generated):
    # example_custom_trajectory()

    # Uncomment to run custom trajectory from CSV file:
    example_custom_trajectory(csv_filepath="data/first_track_pva_3.csv")
