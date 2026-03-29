from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from trajplan.config import (  # noqa: E402
    build_bspline_optimizer_config,
    build_linear_mpc_config,
    build_local_planner_config,
    build_planner_manager_config,
    load_project_configs,
)
from trajplan.map.grid_map import GridMap  # noqa: E402
from trajplan.planning.bspline_optimizer import BsplineOptimizer  # noqa: E402
from trajplan.planning.local_planner import LocalPlanner  # noqa: E402
from trajplan.planning.planner_manager import PlannerManager  # noqa: E402
from trajplan.quadrotor.agent import QuadrotorAgent  # noqa: E402
from trajplan.quadrotor.command import QuadrotorCommand  # noqa: E402
from trajplan.runtime.crazyflie_backend import CrazyflieBackend  # noqa: E402
from trajplan.runtime.mocap_state_provider import MocapStateProvider  # noqa: E402


TAKEOFF_HEIGHT = 0.6
PLANNING_INTERVAL = 1.0
EXEC_RATE = 30.0
MAX_FLIGHT_TIME = 60.0
WAIT_FOR_STATE_TIMEOUT = 10.0
GOAL_POSITIONS = [
    np.array([4.5, 4.5, 1.0], dtype=np.float64),
    np.array([1.0, 2.0, 1.5], dtype=np.float64),
]


def build_bspline_optimizer(
    quadrotor_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
) -> BsplineOptimizer:
    return BsplineOptimizer(
        config=build_bspline_optimizer_config(
            quadrotor_cfg=quadrotor_cfg,
            planning_cfg=planning_cfg,
        )
    )


def build_grid_map(planning_cfg: dict[str, Any]) -> GridMap:
    resolution = float(planning_cfg["grid_map"]["resolution"])
    origin = np.zeros(3, dtype=np.float64)
    size = np.asarray(planning_cfg["grid_map"]["map_size"], dtype=np.int64)
    grid_map = GridMap(resolution=resolution, origin=origin, map_size=size)

    grid_map.add_obstacle(
        obs_start_index=np.array([10, 30, 0], dtype=np.int64),
        obs_size=np.array([30, 10, 30], dtype=np.int64),
    )
    grid_map.add_obstacle(
        obs_start_index=np.array([20, 10, 0], dtype=np.int64),
        obs_size=np.array([20, 10, 30], dtype=np.int64),
    )
    return grid_map


def wait_for_state(
    state_provider: MocapStateProvider,
    timeout_sec: float,
) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout_sec:
        if state_provider.is_initialized():
            return True
        time.sleep(0.05)
    return False


def main() -> None:
    try:
        from crazyflie_py import Crazyswarm
    except ImportError as exc:
        raise ImportError(
            "crazyflie_py is not available. Make sure the Crazyswarm2 environment is sourced."
        ) from exc

    swarm = Crazyswarm()
    cf = swarm.allcfs.crazyflies[0]
    cf_name = cf.prefix.lstrip("/")
    ros_node = cf.node

    print(f"[run_cf_demo] Connected to {cf_name}")

    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )

    grid_map = build_grid_map(planning_cfg)
    bspline_optimizer = build_bspline_optimizer(quadrotor_cfg, planning_cfg)
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
    planner_manager.add_goals(GOAL_POSITIONS)

    state_provider = MocapStateProvider(
        node=ros_node,
        cf_name=cf_name,
        ema_alpha=0.3,
    )
    backend = CrazyflieBackend(
        cf=cf,
        landing_height=0.04,
        landing_duration=TAKEOFF_HEIGHT + 1.5,
    )
    agent = QuadrotorAgent(
        state_provider=state_provider,
        backend=backend,
        controller=None,
    )

    print("[run_cf_demo] Waiting for mocap pose ...")
    if not wait_for_state(
        state_provider=state_provider,
        timeout_sec=WAIT_FOR_STATE_TIMEOUT,
    ):
        print("[run_cf_demo] ERROR: No mocap pose received within timeout.")
        return

    initial_state = state_provider.get_state()
    if initial_state is None:
        print("[run_cf_demo] ERROR: State provider initialized but returned no state.")
        return

    print(
        "[run_cf_demo] Initial position: "
        f"{np.array2string(initial_state.position, precision=3)}"
    )

    try:
        backend.arm(True)
        time.sleep(0.5)
    except Exception as exc:  # noqa: BLE001
        print(f"[run_cf_demo] arm(True) failed (ignored): {exc}")

    print(f"[run_cf_demo] Taking off to {TAKEOFF_HEIGHT:.2f} m ...")
    backend.takeoff(
        target_height=TAKEOFF_HEIGHT,
        duration=TAKEOFF_HEIGHT + 1.5,
    )
    time.sleep(TAKEOFF_HEIGHT + 2.0)
    print("[run_cf_demo] Airborne.")

    now_time = time.monotonic()
    agent.set_active_command(
        QuadrotorCommand.hover(
            start_time=now_time,
            message="Waiting for first plan.",
        )
    )

    execution_dt = 1.0 / EXEC_RATE
    last_planning_time = -np.inf
    flight_start = time.monotonic()

    print("[run_cf_demo] Execution loop started.")

    try:
        while True:
            now_time = time.monotonic()
            elapsed = now_time - flight_start

            if backend.emergency_triggered:
                print("[run_cf_demo] Emergency landing already triggered.")
                break

            if elapsed > MAX_FLIGHT_TIME:
                print(f"[run_cf_demo] Max flight time exceeded: {MAX_FLIGHT_TIME:.1f} s")
                break

            if now_time - last_planning_time >= PLANNING_INTERVAL:
                planning_state = agent.get_current_state()
                if planning_state is None:
                    planning_state = state_provider.get_state()

                new_command = planner_manager.update_and_get_command(
                    current_state=planning_state,
                    current_time=now_time,
                    active_command=agent.active_command,
                )
                agent.set_active_command(new_command)
                last_planning_time = now_time
                print(
                    "[run_cf_demo] planner command: "
                    f"{new_command.mode} | {new_command.message}"
                )

            agent.step(
                dt=execution_dt,
                now_time=now_time,
            )

            if agent.is_finished:
                print("[run_cf_demo] All goals completed.")
                break

            time.sleep(execution_dt)

    except KeyboardInterrupt:
        print("[run_cf_demo] KeyboardInterrupt received.")

    finally:
        print("[run_cf_demo] Landing ...")
        try:
            backend.land()
            time.sleep(TAKEOFF_HEIGHT + 2.0)
        finally:
            try:
                backend.arm(False)
            except Exception as exc:  # noqa: BLE001
                print(f"[run_cf_demo] arm(False) failed (ignored): {exc}")
        print("[run_cf_demo] Done.")


if __name__ == "__main__":
    main()
