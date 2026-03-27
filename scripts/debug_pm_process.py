from __future__ import annotations

import sys
from multiprocessing import Event, Queue
from pathlib import Path
from typing import Any

import numpy as np

from trajplan.map.grid_map import GridMap
from trajplan.planning.bspline_optimizer import BsplineOptimizer
from trajplan.config import (
    build_bspline_optimizer_config,
    build_linear_mpc_config,
    build_local_planner_config,
    build_planner_manager_config,
    load_project_configs,
)
from trajplan.planning.local_planner import LocalPlanner
from trajplan.planning.planner_manager import PlannerManager
from trajplan.runtime.channels import PlannerManagerAgentQueues
from trajplan.runtime.planner_manager_process import PlannerManagerProcess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def format_vec(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=4, suppress_small=True)


def build_bspline_optimizer(
    quadrotor_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
) -> BsplineOptimizer:
    bspline_optimizer_cfg = build_bspline_optimizer_config(
        quadrotor_cfg=quadrotor_cfg,
        planning_cfg=planning_cfg,
    )
    return BsplineOptimizer(config=bspline_optimizer_cfg)


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


def main() -> None:
    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )

    grid_map = build_grid_map(planning_cfg)
    bspline_optimizer = build_bspline_optimizer(quadrotor_cfg, planning_cfg)

    local_planner_cfg = build_local_planner_config(quadrotor_cfg, planning_cfg)
    local_planner = LocalPlanner(config=local_planner_cfg, optimizer=bspline_optimizer)

    planner_manager_cfg = build_planner_manager_config(planning_cfg)
    linear_mpc_cfg = build_linear_mpc_config(quadrotor_cfg, planning_cfg)
    planner_manager = PlannerManager(
        local_planner=local_planner,
        global_lmpc_config=linear_mpc_cfg,
        grid_map=grid_map,
        config=planner_manager_cfg,
    )

    planner_manager.add_goal(np.array([4.5, 4.5, 1.0], dtype=np.float64))
    planner_manager.add_goal(np.array([1.0, 2.0, 0.5], dtype=np.float64))

    planner_manager_agent_queues = PlannerManagerAgentQueues(
        pm_to_agent=Queue(maxsize=1),
        agent_to_pm=Queue(maxsize=1),
    )
    stop_event = Event()

    dt_plan = 1.0
    idle_sleep_time = 0.01
    max_sim_time = 30.0
    planner_process = PlannerManagerProcess(
        planner_manager_agent_queues=planner_manager_agent_queues,
        planner_manager=planner_manager,
        stop_event=stop_event,
        planning_interval=dt_plan,
        idle_sleep_time=idle_sleep_time,
    )

    sim_time = 0.0
    try:
        planner_process.start()
        print("process start successfully")
        while sim_time <= max_sim_time and not stop_event.is_set():
            print(f"current sim time: {sim_time:.2f}")
            sim_time += 0.5
    finally:
        stop_event.set()
        planner_process.join(timeout=5.0)
        if planner_process.is_alive():
            planner_process.terminate()
            planner_process.join(timeout=5.0)


if __name__ == "__main__":
    main()
