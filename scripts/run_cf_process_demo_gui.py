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


from trajplan.config import build_grid_map_config, load_project_configs  # noqa: E402
from trajplan.quadrotor.state import QuadrotorState  # noqa: E402
from trajplan.runtime.channels import ControlQueues, PlannerManagerAgentQueues  # noqa: E402
from trajplan.runtime.crazyflie.agent_process import CrazyflieAgentProcess  # noqa: E402
from trajplan.runtime.planner_manager_process import (  # noqa: E402
    PlannerManagerProcess,
)
from trajplan.runtime.rk4_simulator.agent_process import (  # noqa: E402
    AgentProcess as Rk4SimulatorAgentProcess,
)
from trajplan.visualization.interactive_app import run_interactive_app  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an interactive GUI for a single Crazyflie + planner."
    )
    parser.add_argument("--cf-index", type=int, default=0)
    parser.add_argument("--max-runtime", type=float, default=600.0)
    parser.add_argument("--execution-interval", type=float, default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--takeoff-height", type=float, default=0.6)
    parser.add_argument("--takeoff-duration", type=float, default=None)
    parser.add_argument("--landing-height", type=float, default=0.04)
    parser.add_argument("--landing-duration", type=float, default=2.0)
    parser.add_argument("--wait-for-state-timeout", type=float, default=10.0)
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Use the RK4 simulator backend instead of a real Crazyflie.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )
    agent_execution_interval = (
        float(args.execution_interval)
        if args.execution_interval is not None
        else float(planning_cfg["planner"]["agent_process_control_interval"])
    )
    agent_idle_sleep_time = float(
        planning_cfg["planner"]["agent_process_idle_sleep_time"]
    )

    queues = PlannerManagerAgentQueues(
        pm_to_agent=Queue(maxsize=1),
        agent_to_pm=Queue(maxsize=1),
    )
    agent_visualization_queue = Queue(maxsize=1)
    planner_visualization_queue = Queue(maxsize=1)
    control_queue = ControlQueues(gui_to_planner=Queue(maxsize=16))
    stop_event = Event()
    initial_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    goal_positions = [np.array([1.0, 1.0, 0.8], dtype=np.float64)]

    planner_process = PlannerManagerProcess(
        agent_id=0,
        planner_manager_agent_queues=queues,
        quadrotor_cfg=quadrotor_cfg,
        planning_cfg=planning_cfg,
        goal_positions=goal_positions,
        stop_event=stop_event,
        planning_interval=float(
            planning_cfg["planner"]["pm_process_planning_interval"]
        ),
        idle_sleep_time=float(planning_cfg["planner"]["pm_process_idle_sleep_time"]),
        visualization_queue=planner_visualization_queue,
        control_queue=control_queue.gui_to_planner,
    )

    if args.sim:
        initial_state = QuadrotorState(
            pva=np.hstack(
                (
                    initial_position.copy(),
                    np.zeros(6, dtype=np.float64),
                )
            )
        )
        agent_process = Rk4SimulatorAgentProcess(
            agent_id=0,
            planner_manager_agent_queues=queues,
            quadrotor_cfg=quadrotor_cfg,
            stop_event=stop_event,
            control_interval=agent_execution_interval,
            idle_sleep_time=agent_idle_sleep_time,
            initial_state=initial_state,
            visualization_queue=agent_visualization_queue,
        )
    else:
        agent_process = CrazyflieAgentProcess(
            agent_id=0,
            planner_manager_agent_queues=queues,
            stop_event=stop_event,
            execution_interval=agent_execution_interval,
            idle_sleep_time=agent_idle_sleep_time,
            cf_index=int(args.cf_index),
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

    start_time = time.monotonic()
    planner_process.start()
    agent_process.start()
    print(
        "[Main] PlannerManagerProcess and agent process started. "
        f"sim={bool(args.sim)}, cf_index={int(args.cf_index)}.",
        flush=True,
    )

    try:
        run_interactive_app(
            grid_map_config=build_grid_map_config(planning_cfg),
            agent_visualization_queues={0: agent_visualization_queue},
            planner_visualization_queues={0: planner_visualization_queue},
            control_queues={0: control_queue},
            agent_initial_positions={0: initial_position},
            agent_goal_positions={0: goal_positions},
            swarm=False,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        print("[Main] keyboard interrupt, stopping.", flush=True)
        stop_event.set()
    finally:
        stop_event.set()

        if time.monotonic() - start_time > args.max_runtime:
            stop_event.set()

        planner_process.join(timeout=5.0)
        if planner_process.is_alive():
            planner_process.terminate()
            planner_process.join(timeout=5.0)

        agent_shutdown_timeout = max(float(args.landing_duration) + 3.0, 7.0)
        agent_process.join(timeout=agent_shutdown_timeout)
        if agent_process.is_alive():
            agent_process.terminate()
            agent_process.join(timeout=agent_shutdown_timeout)

    elapsed = time.monotonic() - start_time
    print(
        "[Main] done. "
        f"elapsed={elapsed:.3f}s, "
        f"planner_exitcode={planner_process.exitcode}, "
        f"agent_exitcode={agent_process.exitcode}",
        flush=True,
    )


if __name__ == "__main__":
    main()
