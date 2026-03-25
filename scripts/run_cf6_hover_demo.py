#!/usr/bin/env python3
"""
run_cf6_hover_demo.py
=====================
Single-drone trajectory planning demo for cf6.

Behaviour
---------
* Takes off and flies through any pre-configured goals.
* After all goals are reached the drone **hovers indefinitely** — it does NOT land.
* New goals can be added at any time via the /cf6/add_goal ROS2 service.
* Landing is triggered by calling /cf6/request_land (std_srvs/srv/Trigger).
  This stops the planning loop and commands the physical landing in one call.
  (Crazyswarm2 already owns /cf6/land with a different type, so we use a
  separate service name to avoid the conflict.)

Services
--------
  /cf6/add_goal  (trajectory_planning_interfaces/srv/AddGoal)
      Add a goal waypoint (world frame, metres).
      Example:
        ros2 service call /cf6/add_goal \
          trajectory_planning_interfaces/srv/AddGoal "{x: 1.0, y: 0.5, z: 0.5}"

  /cf6/request_land  (std_srvs/srv/Trigger)
      Stop the planning loop and land the drone.
      Example:
        ros2 service call /cf6/request_land std_srvs/srv/Trigger {}

Usage
-----
  # Terminal 1: Crazyswarm2 sim backend
  ros2 launch crazyflie launch.py backend:=sim

  # Terminal 2: this script
  python scripts/run_cf6_hover_demo.py --ros-args -p use_sim_time:=true

Set SIM_MODE = False and remove --ros-args for real hardware.
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

from std_srvs.srv import Trigger
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

SIM_MODE: bool = True

CF_NAME: str = "cf6"

TAKEOFF_HEIGHT = 0.5  # m

EXEC_RATE = 30.0  # Hz

DT_PLAN = 1.0  # s — planning period

# Initial goals to fly on startup.  Set to [] to start hovering immediately.
GOAL_POSITIONS: list[list[float]] = [
    # [1.5, 1.5, 0.5],
    # [1.5, -1.5, 0.5],
    # [-1.5, 1.5, 0.5],
]


# ---------------------------------------------------------------------------
# Obstacle map builder
# ---------------------------------------------------------------------------


def build_grid_map(planning_cfg: dict[str, Any]) -> GridMap:
    resolution = float(planning_cfg["grid_map"]["resolution"])
    origin = np.array([-2.0, -2.0, 0.0], dtype=np.float64)
    size = np.asarray(planning_cfg["grid_map"]["map_size"], dtype=np.int64)
    grid_map = GridMap(resolution=resolution, origin=origin, map_size=size)

    # 1×1×1 m box centred at world (0, 0).
    grid_map.add_obstacle(
        obs_start_index=np.array([15, 15, 0], dtype=np.int64),
        obs_size=np.array([10, 10, 10], dtype=np.int64),
    )
    return grid_map


# ---------------------------------------------------------------------------
# Helper: wait for state
# ---------------------------------------------------------------------------


def wait_for_state(
    state_provider,
    time_helper,
    timeout_sec: float = 10.0,
    poll_interval: float = 0.1,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if state_provider.is_initialized():
            return True
        time_helper.sleep(poll_interval)
    return False


# ---------------------------------------------------------------------------
# Background planning loop — FINISH is replaced with HOVER
# ---------------------------------------------------------------------------


def planning_loop_hover(
    planner_manager: PlannerManager,
    agent: QuadrotorAgent,
    get_time,
    dt_plan: float,
    stop_event: threading.Event,
    command_lock: threading.Lock,
) -> None:
    """
    Like the standard planning_loop but converts any FINISH command to HOVER.

    This keeps the drone airborne and accepting new goals after the queue
    empties, rather than triggering agent.is_finished and halting setpoints.
    """
    next_plan_time = get_time()

    while not stop_event.is_set():
        now = get_time()

        if now >= next_plan_time:
            current_state = agent.get_current_state()

            with command_lock:
                active_cmd = agent.active_command

            try:
                new_command = planner_manager.update_and_get_command(
                    current_state=current_state,
                    current_time=now,
                    active_command=active_cmd,
                )
            except RuntimeError as exc:
                print(f"[planning_loop/{CF_NAME}] Fatal planning error: {exc}")
                print(f"[planning_loop/{CF_NAME}] Triggering emergency landing.")
                agent.backend.emergency_land(TAKEOFF_HEIGHT)
                return

            # Replace FINISH with HOVER so the drone stays airborne and
            # ready to accept new goals via the /add_goal service.
            if new_command.is_finish:
                new_command = QuadrotorCommand.hover(start_time=now)
                print(f"[{CF_NAME}] All goals reached — hovering.")

            with command_lock:
                agent.set_active_command(new_command)

            next_plan_time = now + dt_plan

        time.sleep(0.01)


# ---------------------------------------------------------------------------
# ROS2 service callback factories
# ---------------------------------------------------------------------------


def make_land_callback(
    land_event: threading.Event, cf, cf_name: str, takeoff_height: float
):
    def callback(request, response):
        if not land_event.is_set():
            land_event.set()
            try:
                cf.notifySetpointsStop()
                cf.land(targetHeight=0.04, duration=takeoff_height + 1.5)
            except Exception as exc:
                print(f"[{cf_name}] land command failed (ignored): {exc}")
        response.success = True
        response.message = f"[{cf_name}] Landing requested."
        print(response.message)
        return response

    return callback


def make_add_goal_callback(planner_manager: PlannerManager, cf_name: str):
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

    # Find cf6 among active crazyflies.
    cf = next(
        (c for c in swarm.allcfs.crazyflies if c.prefix.lstrip("/") == CF_NAME),
        None,
    )
    if cf is None:
        print(f"[cf6_hover] ERROR: drone '{CF_NAME}' not found in crazyflies.yaml.")
        return

    print(f"[cf6_hover] Connected to {CF_NAME}")

    ros_node = cf.node

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
    if SIM_MODE:
        state_provider = TfStateProvider(node=ros_node, cf_name=CF_NAME)
        print("[cf6_hover] Using TfStateProvider (simulation mode)")
    else:
        state_provider = CfStateProvider(node=ros_node, cf_name=CF_NAME)
        print("[cf6_hover] Using CfStateProvider (real hardware mode)")

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
    # 4b. Register ROS2 services
    #
    # /add_goal lives on cf.node (Crazyswarm2's node) — no conflict there.
    # /land already exists on cf.node (Crazyswarm2 owns it), so we create
    # our own server on a SEPARATE node to avoid the "existing service"
    # error.  Both servers receive every /cf6/land call: Crazyswarm2 handles
    # the hardware command; ours sets land_event to stop the planning loop.
    # -----------------------------------------------------------------------
    land_event = threading.Event()

    _add_goal_srv = ros_node.create_service(
        AddGoal,
        f"/{CF_NAME}/add_goal",
        make_add_goal_callback(planner_manager, CF_NAME),
    )
    _request_land_srv = ros_node.create_service(
        Trigger,
        f"/{CF_NAME}/request_land",
        make_land_callback(land_event, cf, CF_NAME, TAKEOFF_HEIGHT),
    )

    print(f"[cf6_hover] Services ready: /{CF_NAME}/add_goal  /{CF_NAME}/request_land")

    # -----------------------------------------------------------------------
    # 5. Wait for state to arrive
    # -----------------------------------------------------------------------
    source_name = "TF transform" if SIM_MODE else "MoCap pose"
    print(f"[cf6_hover] Waiting for {source_name} ...")
    if not wait_for_state(state_provider, timeHelper, timeout_sec=10.0):
        print(f"[cf6_hover] ERROR: No {source_name} received within 10 s. Aborting.")
        return

    initial_state = state_provider.get_state()
    print(
        f"[cf6_hover] Initial position: {np.array2string(initial_state.pva[:3], precision=3)}"
    )

    # -----------------------------------------------------------------------
    # 6. Arm (real hardware only) and take off
    # -----------------------------------------------------------------------
    if not SIM_MODE:
        try:
            cf.arm(True)
            timeHelper.sleep(0.5)
        except Exception as e:
            print(f"[cf6_hover] arm() failed (ignored): {e}")

    print(f"[cf6_hover] Taking off to {TAKEOFF_HEIGHT} m ...")
    cf.takeoff(targetHeight=TAKEOFF_HEIGHT, duration=TAKEOFF_HEIGHT + 1.5)
    timeHelper.sleep(TAKEOFF_HEIGHT + 2.0)
    print("[cf6_hover] Airborne.")

    # -----------------------------------------------------------------------
    # 7. Register initial goals (if any)
    # -----------------------------------------------------------------------
    for goal in GOAL_POSITIONS:
        planner_manager.add_goal(np.array(goal, dtype=np.float64))
    if GOAL_POSITIONS:
        print(f"[cf6_hover] {len(GOAL_POSITIONS)} initial goal(s) queued.")
    else:
        print("[cf6_hover] No initial goals — hovering immediately.")

    # -----------------------------------------------------------------------
    # 8. Start background planning thread
    # -----------------------------------------------------------------------
    stop_event = threading.Event()
    command_lock = threading.Lock()

    planner_thread = threading.Thread(
        target=planning_loop_hover,
        args=(
            planner_manager,
            agent,
            timeHelper.time,
            DT_PLAN,
            stop_event,
            command_lock,
        ),
        daemon=True,
    )
    planner_thread.start()
    print("[cf6_hover] Planning thread started.")

    # -----------------------------------------------------------------------
    # 9. Main execution loop (30 Hz)
    #    Runs until /cf6/land is called (land_event) or MAX_FLIGHT_TIME.
    #    There is NO is_finished check — the drone always keeps hovering.
    # -----------------------------------------------------------------------
    dt = 1.0 / EXEC_RATE

    print(
        f"[cf6_hover] Execution loop started. "
        f"Call 'ros2 service call /{CF_NAME}/land std_srvs/srv/Trigger {{}}' to land."
    )

    while not timeHelper.isShutdown() and not land_event.is_set():
        if agent.backend.emergency_triggered:
            print("[cf6_hover] Emergency landing triggered by planning failure.")
            break

        agent.step(dt=dt, now_time=timeHelper.time())
        timeHelper.sleepForRate(EXEC_RATE)

    # -----------------------------------------------------------------------
    # 10. Stop planning and wait for the drone to land
    # -----------------------------------------------------------------------
    stop_event.set()
    timeHelper.sleep(TAKEOFF_HEIGHT + 2.0)
    if not SIM_MODE:
        try:
            cf.arm(False)
        except Exception as e:
            print(f"[cf6_hover] arm(False) failed (ignored): {e}")
    print("[cf6_hover] Done.")


if __name__ == "__main__":
    main()
