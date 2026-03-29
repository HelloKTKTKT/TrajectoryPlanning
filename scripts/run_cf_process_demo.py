from __future__ import annotations

import sys
import time
from multiprocessing import Event, Queue
from pathlib import Path
from queue import Empty
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
from trajplan.runtime.agent_process_crazyflie import CrazyflieAgentProcess  # noqa: E402
from trajplan.runtime.channels import PlannerManagerAgentQueues  # noqa: E402
from trajplan.runtime.planner_manager_process import PlannerManagerProcess  # noqa: E402


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


def drain_all(queue_obj: Queue) -> list[Any]:
    items: list[Any] = []
    while True:
        try:
            items.append(queue_obj.get(block=False))
        except Empty:
            break
    return items


def main() -> None:
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
    planner_manager.add_goal(np.array([4.5, 4.5, 1.0], dtype=np.float64))
    planner_manager.add_goal(np.array([1.0, 2.0, 1.5], dtype=np.float64))

    planner_manager_agent_queues = PlannerManagerAgentQueues(
        pm_to_agent=Queue(maxsize=1),
        agent_to_pm=Queue(maxsize=1),
    )
    state_log_queue = Queue()
    reference_log_queue = Queue()
    planner_stop_event = Event()
    agent_stop_event = Event()

    planning_interval = 1.0
    execution_interval = 1.0 / 30.0
    idle_sleep_time = 0.005
    max_runtime = 60.0

    planner_process = PlannerManagerProcess(
        planner_manager_agent_queues=planner_manager_agent_queues,
        planner_manager=planner_manager,
        stop_event=planner_stop_event,
        planning_interval=planning_interval,
        idle_sleep_time=idle_sleep_time,
    )
    agent_process = CrazyflieAgentProcess(
        planner_manager_agent_queues=planner_manager_agent_queues,
        agent_stop_event=agent_stop_event,
        execution_interval=execution_interval,
        idle_sleep_time=idle_sleep_time,
        cf_index=0,
        ema_alpha=0.3,
        takeoff_height=0.6,
        takeoff_duration=2.1,
        landing_height=0.04,
        landing_duration=2.1,
        wait_for_state_timeout=10.0,
        arm_before_takeoff=True,
        disarm_after_landing=True,
        state_log_queue=state_log_queue,
        reference_log_queue=reference_log_queue,
    )

    start_time = time.monotonic()

    try:
        planner_process.start()
        agent_process.start()
        print("planner and crazyflie agent processes started")

        while True:
            elapsed = time.monotonic() - start_time

            for _ in drain_all(state_log_queue):
                pass
            for _ in drain_all(reference_log_queue):
                pass

            if elapsed >= max_runtime:
                print("max runtime reached, stopping both processes")
                break

            if not planner_process.is_alive():
                print(f"planner process exited, exitcode={planner_process.exitcode}")
                planner_stop_event.set()

            if not agent_process.is_alive():
                print(f"agent process exited, exitcode={agent_process.exitcode}")
                agent_stop_event.set()
                break

            time.sleep(0.1)

    finally:
        planner_stop_event.set()
        agent_stop_event.set()

        planner_process.join(timeout=5.0)
        agent_process.join(timeout=5.0)

        if planner_process.is_alive():
            planner_process.terminate()
            planner_process.join(timeout=5.0)

        if agent_process.is_alive():
            agent_process.terminate()
            agent_process.join(timeout=5.0)

        print(f"final planner exitcode={planner_process.exitcode}")
        print(f"final agent exitcode={agent_process.exitcode}")
        print("cf process demo finished")


if __name__ == "__main__":
    main()
