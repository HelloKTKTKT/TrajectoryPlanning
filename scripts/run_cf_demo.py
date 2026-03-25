#!/usr/bin/env python3
"""
run_cf_demo.py
==============
ROS 2 entry-point that runs the TrajectoryPlanning pipeline on a Crazyflie.
Works with both the Crazyswarm2 simulation backend and real hardware.

Simulation usage
----------------
  # Terminal 1: ros2 launch crazyflie launch.py backend:=sim
  # Terminal 2: ros2 run crazyflie gui      (NiceGUI at http://localhost:8080)
  # Terminal 3:
  python scripts/run_cf_demo.py --ros-args -p use_sim_time:=true

  Set SIM_MODE = True below before running in simulation.

Real hardware usage
-------------------
  Set SIM_MODE = False (default).
  # Terminal 1: ros2 launch crazyflie launch.py
  # Terminal 2:
  python scripts/run_cf_demo.py

State source
------------
  SIM_MODE=True  → TfStateProvider  (reads TF2 transforms published by sim)
  SIM_MODE=False → CfStateProvider  (subscribes to /{cf_name}/pose from MoCap)
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Make sure the TrajectoryPlanning source is importable when running directly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trajplan.config import (build_bspline_optimizer_config,
                             build_linear_mpc_config,
                             build_local_planner_config,
                             build_planner_manager_config,
                             load_project_configs)
from trajplan.controller import Controller
from trajplan.map.grid_map import GridMap
from trajplan.planning.bspline_optimizer import BsplineOptimizer
from trajplan.planning.local_planner import LocalPlanner
from trajplan.planning.planner_manager import PlannerManager
from trajplan.quadrotor.agent import QuadrotorAgent
from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.runtime.cf_backend import CfBackend
from trajplan.runtime.cf_state_provider import CfStateProvider
from trajplan.runtime.tf_state_provider import TfStateProvider

from trajectory_planning_interfaces.srv import AddGoal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set True to use the Crazyswarm2 sim backend (backend:=sim).
# Set False for real Crazyflie hardware.
SIM_MODE: bool = False

# Flight altitude [m]
TAKEOFF_HEIGHT = 1.0

# Execution loop rate [Hz] — must be high enough to satisfy the Crazyflie
# setpoint watchdog (default watchdog timeout: 100 ms → need >= 10 Hz minimum,
# 30-50 Hz recommended).
EXEC_RATE = 30.0

# Planning is triggered every dt_plan seconds in the background thread.
# MPC + B-spline optimization typically takes < 200 ms, so 1.0 s is safe.
DT_PLAN = 1.0

# Maximum flight time [s] before forcing a landing.
MAX_FLIGHT_TIME = 60.0

# Log a status line every N steps in each loop.
LOG_EVERY_N_STEPS = 30

# Goal positions to visit (world frame, metres).
# Adjust to match your flight space.  All Z values should be >= TAKEOFF_HEIGHT.
GOAL_POSITIONS: list[list[float]] = [
    [0.0, 0.0, 1.5],  # corner (-x, +y)
    # [1.2, -1.2, 1.0],  # corner (+x, +y) — path crosses central obstacle
    # [-1.2, 1.2, 1.0],  # corner (+x, -y)
    # [0.0, 0.0, 1.0],  # corner (-x, +y)
]


# ---------------------------------------------------------------------------
# Obstacle map builder
# ---------------------------------------------------------------------------


def format_vec(v) -> str:
    return f"[{v[0]:6.3f}, {v[1]:6.3f}, {v[2]:6.3f}]"


def build_grid_map(planning_cfg: dict[str, Any]) -> GridMap:
    """
    Build the occupancy grid map for the flight space.

    Edit this function to match your actual environment.  Start with an empty
    map for initial testing, then add obstacles as needed.
    """
    resolution = float(planning_cfg["grid_map"]["resolution"])
    # Map spans (-2, -2, 0) → (2, 2, 2): origin is the minimum corner.
    origin = np.array([-1.5, -1.5, 0.0], dtype=np.float64)
    # origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    size = np.asarray(planning_cfg["grid_map"]["map_size"], dtype=np.int64)
    grid_map = GridMap(resolution=resolution, origin=origin, map_size=size)

    # 1×1×1 m box centred at world (0, 0).
    # World span: (-0.5, -0.5, 0.0) → (0.5, 0.5, 1.0)
    # Grid span:  start [15, 15, 0], size [10, 10, 10] at 0.1 m/cell.
    # grid_map.add_obstacle(
    #     obs_start_index=np.array([15, 15, 0], dtype=np.int64),
    #     obs_size=np.array([10, 10, 10], dtype=np.int64),
    # )

    return grid_map


# ---------------------------------------------------------------------------
# Helper: wait for MoCap state
# ---------------------------------------------------------------------------


def wait_for_state(
    state_provider,
    time_helper,
    timeout_sec: float = 10.0,
    poll_interval: float = 0.1,
) -> bool:
    """Block until the state provider receives its first pose / TF message.

    Uses ``time_helper.sleep()`` so that rclpy callbacks (including the TF
    TransformListener subscription) are processed while waiting.
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if state_provider.is_initialized():
            return True
        time_helper.sleep(poll_interval)
    return False


# ---------------------------------------------------------------------------
# Background planning thread
# ---------------------------------------------------------------------------


def planning_loop(
    planner_manager: PlannerManager,
    agent: QuadrotorAgent,
    get_time,  # callable() → float
    dt_plan: float,
    stop_event: threading.Event,
    command_lock: threading.Lock,
    state_provider,
    log_every_n_steps: int = LOG_EVERY_N_STEPS,
) -> None:
    """
    Periodically call the planner and push the resulting command to the agent.

    Runs in a daemon thread so it is killed automatically when the main thread
    exits.  ``stop_event`` can be set to request a clean shutdown.
    On a fatal planning error, ``agent.backend.emergency_land()`` is called and
    the thread exits; the main loop detects this via
    ``agent.backend.emergency_triggered``.
    """
    next_plan_time = get_time()
    step_count = 0

    while not stop_event.is_set():
        now = get_time()

        if step_count % log_every_n_steps == 0:
            state_now = state_provider.get_state()
            position = state_now.pva[:3]
            velocity = state_now.pva[3:6]
            v_norm = np.linalg.norm(velocity)

            if agent.active_command is None:
                command_mode = "none"
            else:
                command_mode = agent.active_command.mode.value

            active_goal = planner_manager.active_goal_position
            active_goal_str = "None" if active_goal is None else format_vec(active_goal)

            print(
                f"[t={now:6.2f}s] "
                f"pos={format_vec(position)} "
                f"vel_norm={v_norm:.2f} "
                f"cmd={command_mode} "
                f"goal={active_goal_str} "
                f"global_progress={planner_manager.global_progress:.2f}"
            )

        if now >= next_plan_time:
            current_state = state_provider.get_state()

            with command_lock:
                active_cmd = agent.active_command

            try:
                new_command = planner_manager.update_and_get_command(
                    current_state=current_state,
                    current_time=now,
                    active_command=active_cmd,
                )
                print(new_command.message)
                print(f"new command: {new_command.mode}")
            except RuntimeError as exc:
                print(f"[planning_loop] Fatal planning error: {exc}")
                print("[planning_loop] Triggering emergency landing.")
                agent.backend.emergency_land(TAKEOFF_HEIGHT)
                return

            if new_command.is_track:
                new_command.start_time = get_time()

            with command_lock:
                agent.set_active_command(new_command)

            next_plan_time = now + dt_plan

        step_count += 1
        time.sleep(0.01)  # 100 Hz polling — low CPU overhead


# ---------------------------------------------------------------------------
# ROS 2 service callback factory
# ---------------------------------------------------------------------------


def make_add_goal_callback(planner_manager: PlannerManager, cf_name: str):
    """Return a ROS2 service callback that appends a goal to planner_manager."""

    def callback(request, response):
        goal = np.array([request.x, request.y, request.z], dtype=np.float64)
        try:
            planner_manager.add_goal(goal)
            queued = len(planner_manager.goal_position_list)
            response.success = True
            response.message = (
                f"[{cf_name}] Goal [{request.x:.2f}, {request.y:.2f}, "
                f"{request.z:.2f}] added. Queue depth: {queued}"
            )
            response.goals_queued = queued
            print(response.message)
        except ValueError as exc:
            response.success = False
            response.message = f"[{cf_name}] Invalid goal: {exc}"
            response.goals_queued = len(planner_manager.goal_position_list)
        return response

    return callback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from crazyflie_py import Crazyswarm

    # -----------------------------------------------------------------------
    # 1. Initialise Crazyswarm / ROS 2
    # -----------------------------------------------------------------------
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]
    CF_NAME = cf.prefix.lstrip("/")  # e.g. "cf231" — read from crazyflies.yaml

    print(f"[run_cf_demo] Connected to {CF_NAME}")

    # -----------------------------------------------------------------------
    # 2. Load configuration
    # -----------------------------------------------------------------------
    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor_cf.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )

    # -----------------------------------------------------------------------
    # 3. Build planning stack
    # -----------------------------------------------------------------------
    grid_map = build_grid_map(planning_cfg)

    bspline_optimizer = BsplineOptimizer(
        config=build_bspline_optimizer_config(quadrotor_cfg, planning_cfg)
    )
    local_planner = LocalPlanner(
        config=build_local_planner_config(quadrotor_cfg, planning_cfg),
        optimizer=bspline_optimizer,
    )
    planner_manager = PlannerManager(
        local_planner=local_planner,
        global_lmpc_config=build_linear_mpc_config(quadrotor_cfg, planning_cfg),
        grid_map=grid_map,
        config=build_planner_manager_config(planning_cfg),
    )

    # -----------------------------------------------------------------------
    # 4. Build Crazyflie adapters
    # -----------------------------------------------------------------------
    # cf.node is the rclpy Node held by the crazyflie_py.Crazyflie object
    # (confirmed at crazyflie_py/crazyflie_py/crazyflie.py:113).
    ros_node = cf.node

    if SIM_MODE:
        # Sim backend publishes TF2 transforms (world → cf_name), not /pose topics.
        state_provider = TfStateProvider(node=ros_node, cf_name=CF_NAME)
        print("[run_cf_demo] Using TfStateProvider (simulation mode)")
    else:
        # Real hardware: MoCap publishes /{cf_name}/pose (PoseStamped).
        state_provider = CfStateProvider(node=ros_node, cf_name=CF_NAME)
        print("[run_cf_demo] Using CfStateProvider (real hardware mode)")

    cf_backend = CfBackend(cf=cf)

    max_acc = float(
        np.max(
            np.asarray(quadrotor_cfg["limits"]["max_acceleration_3d"], dtype=np.float64)
        )
    )
    controller = Controller(
        kp_position=2.0,
        kv_velocity=1.5,
        max_acceleration_norm=max_acc,
    )

    agent = QuadrotorAgent(
        controller=controller,
        state_provider=state_provider,
        backend=cf_backend,
    )

    # -----------------------------------------------------------------------
    # 4b. Register /CF_NAME/add_goal service
    # -----------------------------------------------------------------------
    _add_goal_srv = ros_node.create_service(
        AddGoal,
        f"/{CF_NAME}/add_goal",
        make_add_goal_callback(planner_manager, CF_NAME),
    )
    print(f"[run_cf_demo] Service ready: /{CF_NAME}/add_goal")

    # -----------------------------------------------------------------------
    # 5. Wait for state to arrive (MoCap pose or TF transform)
    # -----------------------------------------------------------------------
    source_name = "TF transform" if SIM_MODE else "MoCap pose"
    print(f"[run_cf_demo] Waiting for {source_name} ...")
    if not wait_for_state(state_provider, timeHelper, timeout_sec=10.0):
        print(f"[run_cf_demo] ERROR: No {source_name} received within 10 s. Aborting.")
        return

    initial_state = state_provider.get_state()
    print(
        f"[run_cf_demo] Initial position: {np.array2string(initial_state.pva[:3], precision=3)}"
    )

    # -----------------------------------------------------------------------
    # 6. Arm (real hardware only) and take off
    # -----------------------------------------------------------------------
    if not SIM_MODE:
        try:
            cf.arm(True)
            timeHelper.sleep(0.5)
        except Exception as e:
            print(f"[run_cf_demo] arm() failed (ignored): {e}")

    print(f"[run_cf_demo] Taking off to {TAKEOFF_HEIGHT} m ...")
    cf.takeoff(targetHeight=TAKEOFF_HEIGHT, duration=TAKEOFF_HEIGHT + 1.5)
    timeHelper.sleep(TAKEOFF_HEIGHT + 2.0)
    print("[run_cf_demo] Airborne.")

    # -----------------------------------------------------------------------
    # 7. Register goals
    # -----------------------------------------------------------------------
    for goal in GOAL_POSITIONS:
        planner_manager.add_goal(np.array(goal, dtype=np.float64))
    print(f"[run_cf_demo] {len(GOAL_POSITIONS)} goal(s) queued.")

    # -----------------------------------------------------------------------
    # 7b. Pre-warm planner while HLC still holds altitude
    # -----------------------------------------------------------------------
    # The first LMPC + B-spline solve is slow (cold-start).  Running it here
    # — before the execution loop sends any cmdFullState — lets the Crazyflie
    # HLC maintain altitude safely during the wait.  Subsequent calls reuse
    # the warm solver and return in ~0.1 s.
    print("[run_cf_demo] Pre-warming planner ...")
    planner_manager.update_and_get_command(
        current_state=state_provider.get_state(),
        current_time=timeHelper.time(),
        active_command=None,
    )
    planner_manager.clear_goals()
    for goal in GOAL_POSITIONS:
        planner_manager.add_goal(np.array(goal, dtype=np.float64))
    print("[run_cf_demo] Planner warmed up.")

    # -----------------------------------------------------------------------
    # 8. Start background planning thread
    # -----------------------------------------------------------------------
    stop_event = threading.Event()
    command_lock = threading.Lock()

    planner_thread = threading.Thread(
        target=planning_loop,
        args=(
            planner_manager,
            agent,
            timeHelper.time,
            DT_PLAN,
            stop_event,
            command_lock,
            state_provider,
        ),
        daemon=True,
    )
    agent.set_active_command(QuadrotorCommand.hover(
        start_time=timeHelper.time(),
        message="Waiting for first plan",
    ))
    planner_thread.start()
    print("[run_cf_demo] Planning thread started.")

    # -----------------------------------------------------------------------
    # 9. Main execution loop (30 Hz)
    # -----------------------------------------------------------------------
    flight_start = timeHelper.time()
    dt = 1.0 / EXEC_RATE
    step_count = 0

    print("[run_cf_demo] Execution loop started.")

    while not timeHelper.isShutdown():
        now = timeHelper.time()
        elapsed = now - flight_start

        if agent.backend.emergency_triggered:
            print("[run_cf_demo] Emergency landing triggered by planning failure.")
            break

        if agent.is_finished:
            print("[run_cf_demo] All goals reached.")
            break

        if elapsed > MAX_FLIGHT_TIME:
            print(f"[run_cf_demo] Max flight time ({MAX_FLIGHT_TIME} s) exceeded.")
            break

        agent.step(dt=dt, now_time=now)
        if agent.active_command is not None:
            print(f"after agent step command: {agent.active_command.mode}")
            print(f"message: {agent.active_command.message}")
        else:
            print("after agent step command: command is None")

        # if step_count % LOG_EVERY_N_STEPS == 0:
        #     state_now = state_provider.get_state()
        #     position = state_now.pva[:3]
        #     velocity = state_now.pva[3:6]
        #     v_norm = np.linalg.norm(velocity)
        #
        #     if agent.active_command is None:
        #         command_mode = "none"
        #     else:
        #         command_mode = agent.active_command.mode.value
        #
        #     active_goal = planner_manager.active_goal_position
        #     active_goal_str = "None" if active_goal is None else format_vec(active_goal)
        #
        #     print(
        #         f"[t={now:6.2f}s] "
        #         f"pos={format_vec(position)} "
        #         f"vel_norm={v_norm:.2f} "
        #         f"cmd={command_mode} "
        #         f"goal={active_goal_str} "
        #         f"global_progress={planner_manager.global_progress:.2f}"
        #     )

        step_count += 1
        timeHelper.sleepForRate(EXEC_RATE)

    # -----------------------------------------------------------------------
    # 10. Land
    # -----------------------------------------------------------------------
    stop_event.set()
    print("[run_cf_demo] Landing ...")
    cf.notifySetpointsStop()
    cf.land(targetHeight=0.04, duration=TAKEOFF_HEIGHT + 1.5)
    timeHelper.sleep(TAKEOFF_HEIGHT + 2.0)
    if not SIM_MODE:
        try:
            cf.arm(False)
        except Exception as e:
            print(f"[run_cf_demo] arm(False) failed (ignored): {e}")
    print("[run_cf_demo] Done.")


if __name__ == "__main__":
    main()
