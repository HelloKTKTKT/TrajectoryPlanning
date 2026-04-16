from __future__ import annotations

import sys
import time
from multiprocessing import Event, Queue
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from trajplan.config import load_project_configs  # noqa: E402
from trajplan.runtime.agent_process import AgentProcess  # noqa: E402
from trajplan.runtime.channels import PlannerManagerAgentQueues  # noqa: E402
from trajplan.runtime.planner_manager_process import PlannerManagerProcess  # noqa: E402


def main() -> None:
    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )

    queues = PlannerManagerAgentQueues(
        pm_to_agent=Queue(maxsize=1),
        agent_to_pm=Queue(maxsize=1),
    )
    stop_event = Event()
    trajectory_log_path = (
        PROJECT_ROOT / "outputs" / "debug_agent_trajectory.csv"
    )

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
    agent_process = AgentProcess(
        planner_manager_agent_queues=queues,
        quadrotor_cfg=quadrotor_cfg,
        stop_event=stop_event,
        control_interval=float(planning_cfg["planner"]["pm_process_planning_interval"]),
        idle_sleep_time=float(planning_cfg["planner"]["pm_process_idle_sleep_time"]),
        trajectory_log_path=trajectory_log_path,
    )

    start_time = time.monotonic()
    planner_process.start()
    agent_process.start()
    print("[Main] PlannerManagerProcess and AgentProcess started.", flush=True)

    timeout_s = 10.0
    try:
        while not stop_event.is_set():
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout_s:
                print(f"[Main] timeout after {elapsed:.3f}s.", flush=True)
                stop_event.set()
                break
            time.sleep(0.1)
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
        f"agent_exitcode={agent_process.exitcode}, "
        f"trajectory_log={trajectory_log_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
