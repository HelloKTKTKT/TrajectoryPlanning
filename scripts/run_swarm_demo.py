#!/usr/bin/env python3
"""
run_swarm_demo.py
=================
Multi-drone trajectory planning demo using Crazyswarm2.

Supports any number of Crazyflies enabled in crazyflies.yaml.
Each drone gets its own independent planning pipeline; inter-drone collision
avoidance is achieved by injecting each other drone's current position as a
temporary sphere obstacle into a cloned map before every planning call.

Usage
-----
  # Simulation (backend:=sim already running):
  python scripts/run_swarm_demo.py --ros-args -p use_sim_time:=true

  # Real hardware:
  # Edit swarm_cf.yaml: sim_mode: false
  python scripts/run_swarm_demo.py

Configuration
-------------
  configs/swarm_cf.yaml   — sim_mode, inter_drone_safe_radius, per-drone goals
  crazyflies.yaml          — which drones are active (managed by crazyswarm2)

Architecture
------------
  Shared static GridMap (never mutated after construction)
       │
       ├─ per drone: planning thread
       │    clone(static_map) → mark other drones as obstacles → PlannerManager
       │
       └─ main 30 Hz loop: for each QuadrotorAgent: agent.step()
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trajplan.config import (
    build_bspline_optimizer_config,
    build_linear_mpc_config,
    build_local_planner_config,
    build_planner_manager_config,
    load_project_configs,
)
from trajplan.controller import Controller
from trajplan.map.grid_map import GridMap
from trajplan.planning.bspline_optimizer import BsplineOptimizer
from trajplan.planning.local_planner import LocalPlanner
from trajplan.planning.planner_manager import PlannerManager
from trajplan.quadrotor.agent import QuadrotorAgent
from trajplan.runtime.cf_backend import CfBackend
from trajplan.runtime.cf_state_provider import CfStateProvider
from trajplan.runtime.tf_state_provider import TfStateProvider

from trajectory_planning_interfaces.srv import AddGoal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SWARM_CONFIG_PATH = PROJECT_ROOT / "configs" / "swarm_cf.yaml"
QUADROTOR_CONFIG_PATH = PROJECT_ROOT / "configs" / "quadrotor_cf.yaml"
PLANNING_CONFIG_PATH = PROJECT_ROOT / "configs" / "planning.yaml"

TAKEOFF_HEIGHT = 0.5  # m
EXEC_RATE = 30.0  # Hz — setpoint streaming rate
DT_PLAN = 1.0  # s — planning period per drone
MAX_FLIGHT_TIME = 60.0  # s — safety cutoff


# ---------------------------------------------------------------------------
# Per-drone context
# ---------------------------------------------------------------------------


@dataclass
class DroneContext:
    cf_name: str
    cf: Any  # crazyflie_py.Crazyflie
    state_provider: Any  # TfStateProvider or CfStateProvider
    backend: CfBackend
    controller: Controller
    agent: QuadrotorAgent
    planner_manager: PlannerManager
    command_lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)


def build_drone_context(
    cf,
    cf_name: str,
    sim_mode: bool,
    quadrotor_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
    static_grid_map: GridMap,
) -> DroneContext:
    ros_node = cf.node

    if sim_mode:
        state_provider = TfStateProvider(node=ros_node, cf_name=cf_name)
    else:
        state_provider = CfStateProvider(node=ros_node, cf_name=cf_name)

    backend = CfBackend(cf=cf)

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
        backend=backend,
    )

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
        grid_map=static_grid_map,  # replaced per planning call with a clone
        config=build_planner_manager_config(planning_cfg),
    )

    return DroneContext(
        cf_name=cf_name,
        cf=cf,
        state_provider=state_provider,
        backend=backend,
        controller=controller,
        agent=agent,
        planner_manager=planner_manager,
    )


# ---------------------------------------------------------------------------
# Obstacle map builder
# ---------------------------------------------------------------------------


def build_static_grid_map(planning_cfg: dict[str, Any]) -> GridMap:
    """
    Build the shared static occupancy grid.
    This map is NEVER mutated after construction — each planning call clones it.
    """
    resolution = float(planning_cfg["grid_map"]["resolution"])
    origin = np.array([-2.0, -2.0, 0.0], dtype=np.float64)
    size = np.asarray(planning_cfg["grid_map"]["map_size"], dtype=np.int64)
    grid_map = GridMap(resolution=resolution, origin=origin, map_size=size)

    # 1×1×1 m box centred at world (0, 0): (-0.5,-0.5,0) → (0.5,0.5,1.0)
    grid_map.add_obstacle(
        obs_start_index=np.array([15, 15, 0], dtype=np.int64),
        obs_size=np.array([10, 10, 10], dtype=np.int64),
    )

    return grid_map


# ---------------------------------------------------------------------------
# Background planning thread
# ---------------------------------------------------------------------------


def planning_loop_for_drone(
    ctx: DroneContext,
    static_grid_map: GridMap,
    all_contexts: dict[str, DroneContext],
    inter_drone_radius: float,
    get_time,
    dt_plan: float,
) -> None:
    """
    Background planning thread for one drone.

    Every dt_plan seconds:
    1. Clone the static map.
    2. Mark all other drones' current positions as sphere obstacles.
    3. Run the planner on the per-call clone (thread-safe — no shared map mutation).
    4. Push the resulting command to the agent.
    """
    next_plan_time = get_time()

    while not ctx.stop_event.is_set():
        now = get_time()

        if now >= next_plan_time:
            # --- Step 1-2: build drone-specific map ---
            local_map = static_grid_map.clone()
            for other_name, other_ctx in all_contexts.items():
                if other_name == ctx.cf_name:
                    continue
                other_state = other_ctx.agent.get_current_state()
                if other_state is not None:
                    local_map.mark_dynamic_obstacle(
                        center=other_state.pva[:3],
                        radius=inter_drone_radius,
                    )

            # --- Step 3: update this drone's planner map and replan ---
            ctx.planner_manager.grid_map = local_map

            current_state = ctx.agent.get_current_state()
            with ctx.command_lock:
                active_cmd = ctx.agent.active_command

            try:
                new_command = ctx.planner_manager.update_and_get_command(
                    current_state=current_state,
                    current_time=now,
                    active_command=active_cmd,
                )
            except RuntimeError as exc:
                print(f"[planning_loop/{ctx.cf_name}] Fatal planning error: {exc}")
                print(f"[planning_loop/{ctx.cf_name}] Triggering emergency landing.")
                ctx.backend.emergency_land(TAKEOFF_HEIGHT)
                return

            # --- Step 4: commit command ---
            with ctx.command_lock:
                ctx.agent.set_active_command(new_command)

            next_plan_time = now + dt_plan

        time.sleep(0.01)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wait_for_all_states(
    contexts: dict[str, DroneContext],
    time_helper,
    timeout_sec: float = 10.0,
    poll_interval: float = 0.1,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if all(ctx.state_provider.is_initialized() for ctx in contexts.values()):
            return True
        time_helper.sleep(poll_interval)
    not_ready = [
        n for n, c in contexts.items() if not c.state_provider.is_initialized()
    ]
    print(f"[swarm] Timeout waiting for: {not_ready}")
    return False


# ---------------------------------------------------------------------------
# ROS 2 service callback factory
# ---------------------------------------------------------------------------


def make_add_goal_callback(ctx: DroneContext):
    """Return a ROS2 service callback that appends a goal to ctx.planner_manager."""

    def callback(request, response):
        goal = np.array([request.x, request.y, request.z], dtype=np.float64)
        try:
            ctx.planner_manager.add_goal(goal)
            queued = len(ctx.planner_manager.goal_position_list)
            response.success = True
            response.message = (
                f"[{ctx.cf_name}] Goal [{request.x:.2f}, {request.y:.2f}, "
                f"{request.z:.2f}] added. Queue depth: {queued}"
            )
            response.goals_queued = queued
            print(response.message)
        except ValueError as exc:
            response.success = False
            response.message = f"[{ctx.cf_name}] Invalid goal: {exc}"
            response.goals_queued = len(ctx.planner_manager.goal_position_list)
        return response

    return callback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from crazyflie_py import Crazyswarm

    # -----------------------------------------------------------------------
    # Load swarm config
    # -----------------------------------------------------------------------
    with open(SWARM_CONFIG_PATH, "r") as f:
        swarm_cfg = yaml.safe_load(f)

    sim_mode: bool = bool(swarm_cfg.get("sim_mode", False))
    inter_drone_radius: float = float(swarm_cfg.get("inter_drone_safe_radius", 0.4))
    goals_cfg: dict[str, list] = swarm_cfg.get("goals", {})

    print(f"[swarm] sim_mode={sim_mode}  inter_drone_radius={inter_drone_radius} m")

    # -----------------------------------------------------------------------
    # Initialise Crazyswarm — drone list comes from crazyflies.yaml
    # -----------------------------------------------------------------------
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    all_cf_objects = {cf.prefix.lstrip("/"): cf for cf in swarm.allcfs.crazyflies}
    print(f"[swarm] Active drones: {list(all_cf_objects.keys())}")

    # -----------------------------------------------------------------------
    # Load planning / hardware configs
    # -----------------------------------------------------------------------
    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=QUADROTOR_CONFIG_PATH,
        planning_yaml_path=PLANNING_CONFIG_PATH,
    )

    # -----------------------------------------------------------------------
    # Build shared static map (never mutated)
    # -----------------------------------------------------------------------
    static_grid_map = build_static_grid_map(planning_cfg)

    # -----------------------------------------------------------------------
    # Build per-drone contexts
    # -----------------------------------------------------------------------
    contexts: dict[str, DroneContext] = {}
    for cf_name, cf in all_cf_objects.items():
        ctx = build_drone_context(
            cf=cf,
            cf_name=cf_name,
            sim_mode=sim_mode,
            quadrotor_cfg=quadrotor_cfg,
            planning_cfg=planning_cfg,
            static_grid_map=static_grid_map,
        )
        contexts[cf_name] = ctx
        print(f"[swarm] Built pipeline for {cf_name}")

    # -----------------------------------------------------------------------
    # Register per-drone /cf_name/add_goal services
    # -----------------------------------------------------------------------
    _add_goal_services = {}
    for cf_name, ctx in contexts.items():
        _add_goal_services[cf_name] = ctx.cf.node.create_service(
            AddGoal,
            f"/{cf_name}/add_goal",
            make_add_goal_callback(ctx),
        )
        print(f"[swarm] Service ready: /{cf_name}/add_goal")

    # -----------------------------------------------------------------------
    # Wait for state providers to receive first measurements
    # -----------------------------------------------------------------------
    source = "TF transforms" if sim_mode else "MoCap poses"
    print(f"[swarm] Waiting for {source} on all drones ...")
    if not wait_for_all_states(contexts, timeHelper, timeout_sec=10.0):
        print("[swarm] ERROR: Not all drones have state. Aborting.")
        return

    for cf_name, ctx in contexts.items():
        pos = ctx.state_provider.get_state().pva[:3]
        print(f"[swarm]   {cf_name} initial pos: {np.array2string(pos, precision=3)}")

    # -----------------------------------------------------------------------
    # Arm (real hardware only) and take off — stagger by 0.5 s per drone
    # -----------------------------------------------------------------------
    for i, (cf_name, ctx) in enumerate(contexts.items()):
        if not sim_mode:
            try:
                ctx.cf.arm(True)
                timeHelper.sleep(0.3)
            except Exception as e:
                print(f"[swarm] arm({cf_name}) failed (ignored): {e}")

        print(f"[swarm] Takeoff {cf_name} to {TAKEOFF_HEIGHT} m ...")
        ctx.cf.takeoff(targetHeight=TAKEOFF_HEIGHT, duration=TAKEOFF_HEIGHT + 1.5)
        timeHelper.sleep(0.5)  # stagger takeoffs

    # Wait for all to reach altitude
    timeHelper.sleep(TAKEOFF_HEIGHT + 1.5)
    print("[swarm] All drones airborne.")

    # -----------------------------------------------------------------------
    # Register goals from swarm_cf.yaml
    # -----------------------------------------------------------------------
    for cf_name, ctx in contexts.items():
        drone_goals = goals_cfg.get(cf_name, [])
        for goal in drone_goals:
            ctx.planner_manager.add_goal(np.array(goal, dtype=np.float64))
        print(f"[swarm] {cf_name}: {len(drone_goals)} goal(s) queued")

    # -----------------------------------------------------------------------
    # Start one planning thread per drone
    # -----------------------------------------------------------------------
    planning_threads = []
    for cf_name, ctx in contexts.items():
        t = threading.Thread(
            target=planning_loop_for_drone,
            args=(
                ctx,
                static_grid_map,
                contexts,
                inter_drone_radius,
                timeHelper.time,
                DT_PLAN,
            ),
            daemon=True,
            name=f"plan_{cf_name}",
        )
        t.start()
        planning_threads.append(t)
        print(f"[swarm] Planning thread started for {cf_name}")

    # -----------------------------------------------------------------------
    # Main 30 Hz execution loop
    # -----------------------------------------------------------------------
    flight_start = timeHelper.time()
    dt = 1.0 / EXEC_RATE
    print("[swarm] Execution loop started.")

    while not timeHelper.isShutdown():
        now = timeHelper.time()
        elapsed = now - flight_start

        failed = [n for n, c in contexts.items() if c.backend.emergency_triggered]
        if failed:
            print(f"[swarm] Emergency landing triggered by planning failure on: {failed}")
            break

        if elapsed > MAX_FLIGHT_TIME:
            print(f"[swarm] Max flight time ({MAX_FLIGHT_TIME} s) exceeded.")
            break

        if all(ctx.agent.is_finished for ctx in contexts.values()):
            print("[swarm] All drones reached their goals.")
            break

        for ctx in contexts.values():
            ctx.agent.step(dt=dt, now_time=now)

        timeHelper.sleepForRate(EXEC_RATE)

    # -----------------------------------------------------------------------
    # Land all drones
    # -----------------------------------------------------------------------
    for ctx in contexts.values():
        ctx.stop_event.set()

    print("[swarm] Landing all drones ...")
    for cf_name, ctx in contexts.items():
        ctx.cf.notifySetpointsStop()
        ctx.cf.land(targetHeight=0.04, duration=TAKEOFF_HEIGHT + 1.5)
        timeHelper.sleep(TAKEOFF_HEIGHT + 2.0)  # wait for full landing before next

    if not sim_mode:
        for cf_name, ctx in contexts.items():
            try:
                ctx.cf.arm(False)
            except Exception as e:
                print(f"[swarm] arm(False) {cf_name} failed (ignored): {e}")

    print("[swarm] Done.")


if __name__ == "__main__":
    main()
