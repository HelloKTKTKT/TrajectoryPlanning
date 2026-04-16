from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Event, Queue
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from trajplan.config import load_project_configs  # noqa: E402
from trajplan.runtime.agent_process_crazyflie import CrazyflieAgentProcess  # noqa: E402
from trajplan.runtime.channels import PlannerManagerAgentQueues  # noqa: E402
from trajplan.runtime.planner_manager_process import PlannerManagerProcess  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PlannerManagerProcess with CrazyflieAgentProcess."
    )
    parser.add_argument("--cf-index", type=int, default=0)
    parser.add_argument("--max-runtime", type=float, default=60.0)
    parser.add_argument("--execution-interval", type=float, default=0.01)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--takeoff-height", type=float, default=0.6)
    parser.add_argument("--takeoff-duration", type=float, default=None)
    parser.add_argument("--landing-height", type=float, default=0.04)
    parser.add_argument("--landing-duration", type=float, default=2.0)
    parser.add_argument("--wait-for-state-timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )

    queues = PlannerManagerAgentQueues(
        pm_to_agent=Queue(maxsize=1),
        agent_to_pm=Queue(maxsize=1),
    )
    stop_event = Event()

    planner_process = PlannerManagerProcess(
        planner_manager_agent_queues=queues,
        quadrotor_cfg=quadrotor_cfg,
        planning_cfg=planning_cfg,
        stop_event=stop_event,
        planning_interval=float(
            planning_cfg["planner"]["pm_process_planning_interval"]
        ),
        idle_sleep_time=float(planning_cfg["planner"]["pm_process_idle_sleep_time"]),
    )
    agent_process = CrazyflieAgentProcess(
        planner_manager_agent_queues=queues,
        stop_event=stop_event,
        execution_interval=float(args.execution_interval),
        idle_sleep_time=float(planning_cfg["planner"]["pm_process_idle_sleep_time"]),
        cf_index=int(args.cf_index),
        ema_alpha=float(args.ema_alpha),
        takeoff_height=float(args.takeoff_height),
        takeoff_duration=args.takeoff_duration,
        landing_height=float(args.landing_height),
        landing_duration=float(args.landing_duration),
        wait_for_state_timeout=float(args.wait_for_state_timeout),
        arm_before_takeoff=True,
        disarm_after_landing=True,
    )

    start_time = time.monotonic()
    planner_process.start()
    agent_process.start()
    print("[Main] PlannerManagerProcess and CrazyflieAgentProcess started.", flush=True)

    try:
        while not stop_event.is_set():
            elapsed = time.monotonic() - start_time
            if elapsed >= args.max_runtime:
                print(f"[Main] timeout after {elapsed:.3f}s.", flush=True)
                stop_event.set()
                break

            if not planner_process.is_alive():
                print(
                    "[Main] planner process exited early, "
                    f"exitcode={planner_process.exitcode}",
                    flush=True,
                )
                stop_event.set()
                break

            if not agent_process.is_alive():
                print(
                    "[Main] crazyflie agent process exited early, "
                    f"exitcode={agent_process.exitcode}",
                    flush=True,
                )
                stop_event.set()
                break

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[Main] keyboard interrupt, stopping.", flush=True)
        stop_event.set()
    finally:
        stop_event.set()

        planner_process.join(timeout=5.0)
        if planner_process.is_alive():
            planner_process.terminate()
            planner_process.join(timeout=5.0)

        agent_process.join(timeout=5.0)
        if agent_process.is_alive():
            agent_process.terminate()
            agent_process.join(timeout=5.0)

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
