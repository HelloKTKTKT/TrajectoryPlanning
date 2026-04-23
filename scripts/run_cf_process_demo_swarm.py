from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Event, Queue
from pathlib import Path


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
from trajplan.planning.messages import (  # noqa: E402
    NeighborTrajectoryMessage,
    PlannerTickInput,
    PlannerTickOutput,
)
from trajplan.runtime.channels import (  # noqa: E402
    PlannerManagerAgentQueues,
    PlannerSwarmQueues,
)
from trajplan.runtime.crazyflie.agent_process import CrazyflieAgentProcess  # noqa: E402
from trajplan.runtime.ipc import drain_latest  # noqa: E402
from trajplan.runtime.planner_manager_process import PlannerManagerProcess  # noqa: E402
from trajplan.runtime.swarm_relay_process import SwarmRelayProcess  # noqa: E402
from trajplan.visualization.live_visualizer import LiveVisualizer  # noqa: E402
from trajplan.visualization.messages import AgentVisualizationSnapshot  # noqa: E402


def _has_abnormal_exit(processes: list[object]) -> bool:
    for process in processes:
        exitcode = getattr(process, "exitcode", None)
        if exitcode is not None and exitcode != 0:
            return True
    return False


def _resolve_cf_indices(
    cf_indices: list[int] | None,
    agent_count: int,
) -> list[int]:
    if cf_indices is None:
        return list(range(agent_count))

    resolved_cf_indices = [int(cf_index) for cf_index in cf_indices]
    if len(resolved_cf_indices) != agent_count:
        raise ValueError(
            "Expected the number of --cf-indices entries to match the number "
            f"of agents in the scenario ({agent_count}), but got "
            f"{len(resolved_cf_indices)}."
        )
    if len(set(resolved_cf_indices)) != len(resolved_cf_indices):
        raise ValueError(
            f"Expected unique --cf-indices entries, but got {resolved_cf_indices}."
        )
    if any(cf_index < 0 for cf_index in resolved_cf_indices):
        raise ValueError(
            f"Expected non-negative --cf-indices, but got {resolved_cf_indices}."
        )
    return resolved_cf_indices


def _compute_agent_shutdown_timeout(landing_duration: float) -> float:
    return max(float(landing_duration) + 3.0, 7.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PlannerManagerProcess with multiple CrazyflieAgentProcess workers."
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=PROJECT_ROOT / "configs" / "swarm_scenario_crossing_cf.yaml",
        help="Path to the swarm scenario yaml. Agents are read in yaml order.",
    )
    parser.add_argument(
        "--cf-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Crazyflie indices in Crazyswarm order, one per agent in scenario order. "
            "Defaults to 0..N-1."
        ),
    )
    parser.add_argument("--max-runtime", type=float, default=60.0)
    parser.add_argument("--execution-interval", type=float, default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--takeoff-height", type=float, default=0.6)
    parser.add_argument("--takeoff-duration", type=float, default=None)
    parser.add_argument("--landing-height", type=float, default=0.04)
    parser.add_argument("--landing-duration", type=float, default=2.0)
    parser.add_argument("--wait-for-state-timeout", type=float, default=10.0)
    parser.add_argument("--visualize", action="store_true")
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

    cf_indices = _resolve_cf_indices(
        cf_indices=args.cf_indices,
        agent_count=len(swarm_scenario.agents),
    )
    agent_execution_interval = (
        float(args.execution_interval)
        if args.execution_interval is not None
        else float(planning_cfg["planner"]["agent_process_control_interval"])
    )
    agent_shutdown_timeout = _compute_agent_shutdown_timeout(
        landing_duration=float(args.landing_duration),
    )
    agent_idle_sleep_time = float(
        planning_cfg["planner"]["agent_process_idle_sleep_time"]
    )

    stop_event = Event()

    planner_processes: dict[int, PlannerManagerProcess] = {}
    agent_processes: dict[int, CrazyflieAgentProcess] = {}
    agent_visualization_queues: dict[int, Queue[AgentVisualizationSnapshot]] = {}
    relay_to_planner_queues: dict[int, Queue[NeighborTrajectoryMessage]] = {}
    planner_to_relay_queue: Queue[NeighborTrajectoryMessage] = Queue(maxsize=128)

    agent_initial_positions = {
        agent.agent_id: agent.initial_position.copy() for agent in swarm_scenario.agents
    }
    agent_goal_positions = {
        agent.agent_id: [goal.copy() for goal in agent.goal_positions]
        for agent in swarm_scenario.agents
    }
    agent_to_cf_mapping = {
        agent.agent_id: cf_index
        for agent, cf_index in zip(swarm_scenario.agents, cf_indices, strict=True)
    }

    for agent, cf_index in zip(swarm_scenario.agents, cf_indices, strict=True):
        pm_to_agent: Queue[PlannerTickOutput] = Queue(maxsize=1)
        agent_to_pm: Queue[PlannerTickInput] = Queue(maxsize=1)
        planner_agent_queues = PlannerManagerAgentQueues(
            pm_to_agent=pm_to_agent,
            agent_to_pm=agent_to_pm,
        )
        agent_visualization_queue: Queue[AgentVisualizationSnapshot] = Queue(maxsize=1)
        relay_to_planner_queue: Queue[NeighborTrajectoryMessage] = Queue(maxsize=64)
        agent_visualization_queues[agent.agent_id] = agent_visualization_queue
        relay_to_planner_queues[agent.agent_id] = relay_to_planner_queue

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
            idle_sleep_time=float(planning_cfg["planner"]["pm_process_idle_sleep_time"]),
            swarm_queues=PlannerSwarmQueues(
                planner_to_relay=planner_to_relay_queue,
                relay_to_planner=relay_to_planner_queue,
            ),
            visualization_queue=None,
        )
        agent_processes[agent.agent_id] = CrazyflieAgentProcess(
            agent_id=agent.agent_id,
            planner_manager_agent_queues=planner_agent_queues,
            stop_event=stop_event,
            execution_interval=agent_execution_interval,
            idle_sleep_time=agent_idle_sleep_time,
            cf_index=cf_index,
            ema_alpha=float(args.ema_alpha),
            takeoff_height=float(args.takeoff_height),
            takeoff_duration=args.takeoff_duration,
            landing_height=float(args.landing_height),
            landing_duration=float(args.landing_duration),
            wait_for_state_timeout=float(args.wait_for_state_timeout),
            arm_before_takeoff=True,
            disarm_after_landing=True,
            visualization_queue=agent_visualization_queue,
        )

    swarm_relay_process = SwarmRelayProcess(
        inbound_queue=planner_to_relay_queue,
        outbound_queues=relay_to_planner_queues,
        stop_event=stop_event,
        idle_sleep_time=float(planning_cfg["planner"]["pm_process_idle_sleep_time"]),
    )
    visualizer = (
        LiveVisualizer(
            grid_map_config=build_grid_map_config(planning_cfg),
            title="Crazyflie Swarm Visualizer",
            agent_initial_positions=agent_initial_positions,
            agent_goal_positions=agent_goal_positions,
        )
        if args.visualize
        else None
    )

    start_time = time.monotonic()
    mission_processes = [
        *planner_processes.values(),
        *agent_processes.values(),
    ]
    all_processes = [
        swarm_relay_process,
        *mission_processes,
    ]
    for process in all_processes:
        process.start()
    print(
        f"[Main] started {len(planner_processes)} planner processes, "
        f"{len(agent_processes)} crazyflie agent processes, and 1 swarm relay process. "
        f"scenario={args.scenario}, agent_to_cf_mapping={agent_to_cf_mapping}",
        flush=True,
    )

    try:
        while not stop_event.is_set():
            elapsed = time.monotonic() - start_time
            if visualizer is not None:
                agent_snapshots = {
                    agent_id: snapshot
                    for agent_id, queue_obj in agent_visualization_queues.items()
                    if (snapshot := drain_latest(queue_obj)) is not None
                }
                visualizer.update(agent_snapshots=agent_snapshots)

            if elapsed >= float(args.max_runtime):
                print(f"[Main] timeout after {elapsed:.3f}s.", flush=True)
                stop_event.set()
                break

            if _has_abnormal_exit(all_processes):
                print("[Main] detected abnormal child-process exit.", flush=True)
                stop_event.set()
                break

            if all(not process.is_alive() for process in mission_processes):
                print("[Main] all planner and crazyflie agent processes finished.", flush=True)
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

        for process in planner_processes.values():
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

        for process in agent_processes.values():
            process.join(timeout=agent_shutdown_timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=agent_shutdown_timeout)

    elapsed = time.monotonic() - start_time
    planner_exitcodes = {
        agent_id: process.exitcode for agent_id, process in planner_processes.items()
    }
    agent_exitcodes = {
        agent_id: process.exitcode for agent_id, process in agent_processes.items()
    }
    relay_exitcode = swarm_relay_process.exitcode
    print(
        "[Main] done. "
        f"elapsed={elapsed:.3f}s, "
        f"relay_exitcode={relay_exitcode}, "
        f"planner_exitcodes={planner_exitcodes}, "
        f"agent_exitcodes={agent_exitcodes}",
        flush=True,
    )
    if visualizer is not None:
        visualizer.wait_until_closed()


if __name__ == "__main__":
    main()
