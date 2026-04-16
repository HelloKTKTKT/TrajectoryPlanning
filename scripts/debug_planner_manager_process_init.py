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

    process = PlannerManagerProcess(
        planner_manager_agent_queues=queues,
        quadrotor_cfg=quadrotor_cfg,
        planning_cfg=planning_cfg,
        stop_event=stop_event,
        planning_interval=float(
            planning_cfg["planner"]["pm_process_planning_interval"]
        ),
        idle_sleep_time=float(planning_cfg["planner"]["pm_process_idle_sleep_time"]),
    )

    start_time = time.monotonic()
    process.start()
    print("[Main] PlannerManagerProcess started.", flush=True)

    try:
        time.sleep(10.0)
    finally:
        stop_event.set()
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

    elapsed = time.monotonic() - start_time
    print(
        f"[Main] done. elapsed={elapsed:.3f}s, exitcode={process.exitcode}",
        flush=True,
    )


if __name__ == "__main__":
    main()
