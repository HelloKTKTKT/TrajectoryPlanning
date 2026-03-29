from __future__ import annotations

import sys
import time
from multiprocessing import Event, Queue
from pathlib import Path
from queue import Empty
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from trajplan.config import (  # noqa: E402
    build_bspline_optimizer_config,
    build_differential_flatness_controller_config,
    build_linear_mpc_config,
    build_local_planner_config,
    build_planner_manager_config,
    build_quadrotor_physical_config,
    load_project_configs,
)
from trajplan.controller import DifferentialFlatnessController  # noqa: E402
from trajplan.map.grid_map import GridMap  # noqa: E402
from trajplan.planning.bspline_optimizer import BsplineOptimizer  # noqa: E402
from trajplan.planning.local_planner import LocalPlanner  # noqa: E402
from trajplan.planning.planner_manager import PlannerManager  # noqa: E402
from trajplan.quadrotor.agent import QuadrotorAgent  # noqa: E402
from trajplan.quadrotor.state import QuadrotorState  # noqa: E402
from trajplan.runtime.agent_process import AgentProcess  # noqa: E402
from trajplan.runtime.channels import PlannerManagerAgentQueues  # noqa: E402
from trajplan.runtime.planner_manager_process import PlannerManagerProcess  # noqa: E402
from trajplan.runtime.sim_backend import SimBackend  # noqa: E402
from trajplan.runtime.sim_state_provider import SimStateProvider  # noqa: E402


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

    physical_config = build_quadrotor_physical_config(quadrotor_cfg)
    controller_config = build_differential_flatness_controller_config(quadrotor_cfg)

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

    initial_state = QuadrotorState(
        pva=np.array(
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        euler=np.zeros(3, dtype=np.float64),
        angular_rate=np.zeros(3, dtype=np.float64),
    )
    state_provider = SimStateProvider(initial_state=initial_state)
    controller = DifferentialFlatnessController(
        physical_config=physical_config,
        config=controller_config,
    )
    backend = SimBackend(
        state_provider=state_provider,
        physical_config=physical_config,
        integration_method="rk4",
        min_rotor_thrust=0.0,
    )
    agent = QuadrotorAgent(
        state_provider=state_provider,
        backend=backend,
        controller=controller,
    )

    planner_manager_agent_queues = PlannerManagerAgentQueues(
        pm_to_agent=Queue(maxsize=1),
        agent_to_pm=Queue(maxsize=1),
    )
    state_log_queue = Queue()
    reference_log_queue = Queue()
    planner_stop_event = Event()
    agent_stop_event = Event()

    planning_interval = 1.0
    execution_interval = 0.01
    idle_sleep_time = 0.005
    max_runtime = 30.0

    planner_process = PlannerManagerProcess(
        planner_manager_agent_queues=planner_manager_agent_queues,
        planner_manager=planner_manager,
        stop_event=planner_stop_event,
        planning_interval=planning_interval,
        idle_sleep_time=idle_sleep_time,
    )
    agent_process = AgentProcess(
        planner_manager_agent_queues=planner_manager_agent_queues,
        agent=agent,
        agent_stop_event=agent_stop_event,
        execution_interval=execution_interval,
        idle_sleep_time=idle_sleep_time,
        state_log_queue=state_log_queue,
        reference_log_queue=reference_log_queue,
    )

    start_time = time.monotonic()
    position_history: list[np.ndarray] = [initial_state.position.copy()]
    reference_position_history: list[np.ndarray] = [initial_state.position.copy()]

    try:
        planner_process.start()
        agent_process.start()
        print("planner and agent processes started")

        while True:
            elapsed = time.monotonic() - start_time
            for logged_state in drain_all(state_log_queue):
                position_history.append(logged_state.position.copy())
            for logged_reference_pva in drain_all(reference_log_queue):
                reference_position_history.append(logged_reference_pva[:3].copy())

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
        print("process demo finished")

    if len(position_history) > 1:
        ax = grid_map.plot_3d()
        path = np.vstack(position_history)
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color="tomato",
            linewidth=2,
            label="agent trajectory",
        )
        if len(reference_position_history) > 1:
            ref_path = np.vstack(reference_position_history)
            ax.plot(
                ref_path[:, 0],
                ref_path[:, 1],
                ref_path[:, 2],
                color="black",
                linewidth=1.5,
                linestyle="--",
                label="reference p",
            )
        ax.scatter(
            path[0, 0],
            path[0, 1],
            path[0, 2],
            s=80,
            color="limegreen",
            marker="o",
            label="start",
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            path[-1, 2],
            s=80,
            color="orange",
            marker="D",
            label="end",
        )
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
