from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Event, Queue
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MPLCONFIGDIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from trajplan.config import (  # noqa: E402
    build_grid_map_config,
    load_project_configs,
    load_swarm_scenario_config,
)
from trajplan.quadrotor.state import QuadrotorState  # noqa: E402
from trajplan.runtime.channels import (  # noqa: E402
    PlannerManagerAgentQueues,
    PlannerSwarmQueues,
    PlannerTargetQueues,
)
from trajplan.runtime.ipc import drain_latest  # noqa: E402
from trajplan.runtime.planner_manager_process import PlannerManagerProcess  # noqa: E402
from trajplan.runtime.rk4_simulator.agent_process import AgentProcess  # noqa: E402
from trajplan.runtime.swarm_relay_process import SwarmRelayProcess  # noqa: E402
from trajplan.runtime.target_process import TargetProcess  # noqa: E402
from trajplan.visualization.live_visualizer import LiveVisualizer  # noqa: E402


def _has_abnormal_exit(processes: list[object]) -> bool:
    for process in processes:
        exitcode = getattr(process, "exitcode", None)
        if exitcode is not None and exitcode != 0:
            return True
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=Path,
        default=PROJECT_ROOT / "configs" / "swarm_following_scenario.yaml",
        help="Path to the swarm scenario yaml.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Maximum runtime in seconds before forcing shutdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )
    swarm_scenario = load_swarm_scenario_config(args.scenario)
    if len(swarm_scenario.agents) == 0:
        raise ValueError("Swarm scenario must contain at least one agent.")

    stop_event = Event()

    planner_processes: dict[int, PlannerManagerProcess] = {}
    agent_processes: dict[int, AgentProcess] = {}
    agent_visualization_queues: dict[int, Queue] = {}
    relay_to_planner_queues: dict[int, Queue] = {}
    target_to_planner_queues: dict[int, Queue] = {}
    planner_to_relay_queue: Queue = Queue(maxsize=128)
    target_visualization_queue: Queue = Queue(maxsize=1)

    agent_initial_positions = {
        agent.agent_id: agent.initial_position.copy() for agent in swarm_scenario.agents
    }
    agent_goal_positions = {
        agent.agent_id: [goal.copy() for goal in agent.goal_positions]
        for agent in swarm_scenario.agents
    }

    for agent in swarm_scenario.agents:
        planner_agent_queues = PlannerManagerAgentQueues(
            pm_to_agent=Queue(maxsize=1),
            agent_to_pm=Queue(maxsize=1),
        )
        agent_visualization_queue = Queue(maxsize=1)
        relay_to_planner_queue = Queue(maxsize=64)
        target_to_planner_queue = Queue(maxsize=1)
        agent_visualization_queues[agent.agent_id] = agent_visualization_queue
        relay_to_planner_queues[agent.agent_id] = relay_to_planner_queue
        target_to_planner_queues[agent.agent_id] = target_to_planner_queue

        initial_state = QuadrotorState(
            pva=np.hstack(
                (
                    agent.initial_position,
                    np.zeros(6, dtype=np.float64),
                )
            )
        )
        trajectory_log_path = (
            PROJECT_ROOT / "outputs" / f"debug_agent_{agent.agent_id}_trajectory.csv"
        )

        planner_processes[agent.agent_id] = PlannerManagerProcess(
            agent_id=agent.agent_id,
            planner_manager_agent_queues=planner_agent_queues,
            quadrotor_cfg=quadrotor_cfg,
            planning_cfg=planning_cfg,
            goal_positions=agent.goal_positions,
            stop_event=stop_event,
            planning_interval=float(
                planning_cfg["planner"]["pm_process_planning_interval"]
            ),
            idle_sleep_time=float(
                planning_cfg["planner"]["pm_process_idle_sleep_time"]
            ),
            swarm_queues=PlannerSwarmQueues(
                planner_to_relay=planner_to_relay_queue,
                relay_to_planner=relay_to_planner_queue,
            ),
            target_queues=PlannerTargetQueues(
                target_to_planner=target_to_planner_queue,
            ),
            visualization_queue=None,
        )
        agent_processes[agent.agent_id] = AgentProcess(
            agent_id=agent.agent_id,
            planner_manager_agent_queues=planner_agent_queues,
            quadrotor_cfg=quadrotor_cfg,
            stop_event=stop_event,
            control_interval=float(
                planning_cfg["planner"]["agent_process_control_interval"]
            ),
            idle_sleep_time=float(
                planning_cfg["planner"]["agent_process_idle_sleep_time"]
            ),
            initial_state=initial_state,
            visualization_queue=agent_visualization_queue,
            trajectory_log_path=trajectory_log_path,
        )

    swarm_relay_process = SwarmRelayProcess(
        inbound_queue=planner_to_relay_queue,
        outbound_queues=relay_to_planner_queues,
        stop_event=stop_event,
        idle_sleep_time=float(planning_cfg["planner"]["pm_process_idle_sleep_time"]),
    )
    target_process = (
        None
        if swarm_scenario.target is None
        else TargetProcess(
            target_scenario=swarm_scenario.target,
            outbound_queues=target_to_planner_queues,
            stop_event=stop_event,
            visualization_queue=target_visualization_queue,
        )
    )

    visualizer = LiveVisualizer(
        grid_map_config=build_grid_map_config(planning_cfg),
        title="RK4 Swarm Visualizer",
        agent_initial_positions=agent_initial_positions,
        agent_goal_positions=agent_goal_positions,
    )

    start_time = time.monotonic()
    mission_processes = [
        *planner_processes.values(),
        *agent_processes.values(),
    ]
    all_processes = [
        swarm_relay_process,
        *([] if target_process is None else [target_process]),
        *mission_processes,
    ]
    for process in all_processes:
        process.start()
    print(
        f"[Main] started {len(planner_processes)} planner processes and "
        f"{len(agent_processes)} agent processes, and 1 swarm relay process"
        f"{'' if target_process is None else ', and 1 target process'}. "
        f"scenario={args.scenario}",
        flush=True,
    )

    timeout_s = float(args.timeout)
    try:
        while not stop_event.is_set():
            elapsed = time.monotonic() - start_time

            agent_snapshots = {
                agent_id: snapshot
                for agent_id, queue_obj in agent_visualization_queues.items()
                if (snapshot := drain_latest(queue_obj)) is not None
            }
            target_snapshot = drain_latest(target_visualization_queue)
            visualizer.update(
                agent_snapshots=agent_snapshots,
                target_snapshot=target_snapshot,
            )

            if elapsed >= timeout_s:
                print(f"[Main] timeout after {elapsed:.3f}s.", flush=True)
                stop_event.set()
                break

            if _has_abnormal_exit(all_processes):
                print("[Main] detected abnormal child-process exit.", flush=True)
                stop_event.set()
                break

            if all(not process.is_alive() for process in mission_processes):
                print("[Main] all planner and agent processes finished.", flush=True)
                stop_event.set()
                break

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[Main] keyboard interrupt, stopping.", flush=True)
        stop_event.set()
    finally:
        stop_event.set()

        swarm_relay_process.join(timeout=5.0)
        if swarm_relay_process.is_alive():
            swarm_relay_process.terminate()
            swarm_relay_process.join(timeout=5.0)

        if target_process is not None:
            target_process.join(timeout=5.0)
            if target_process.is_alive():
                target_process.terminate()
                target_process.join(timeout=5.0)

        for process in planner_processes.values():
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

        for process in agent_processes.values():
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

    elapsed = time.monotonic() - start_time
    planner_exitcodes = {
        agent_id: process.exitcode for agent_id, process in planner_processes.items()
    }
    agent_exitcodes = {
        agent_id: process.exitcode for agent_id, process in agent_processes.items()
    }
    relay_exitcode = swarm_relay_process.exitcode
    target_exitcode = None if target_process is None else target_process.exitcode
    print(
        "[Main] done. "
        f"elapsed={elapsed:.3f}s, "
        f"relay_exitcode={relay_exitcode}, "
        f"target_exitcode={target_exitcode}, "
        f"planner_exitcodes={planner_exitcodes}, "
        f"agent_exitcodes={agent_exitcodes}",
        flush=True,
    )
    visualizer.wait_until_closed()


if __name__ == "__main__":
    main()
