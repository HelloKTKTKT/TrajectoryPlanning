from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from trajplan.planning.local_planner import LocalPlanner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def format_vec(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=4, suppress_small=True)


def run_one_case(
    planner: LocalPlanner,
    case_name: str,
    local_start_pva: np.ndarray,
    local_target_pva: np.ndarray,
    current_time: float = 0.0,
) -> None:
    print("=" * 80)
    print(f"Case: {case_name}")

    trajectory = planner.plan(
        active_command=None,
        local_start_pva=local_start_pva,
        local_target_pva=local_target_pva,
        current_time=current_time,
    )

    if trajectory is None:
        print("Result: planning failed, trajectory is None")
        return

    print("Result: planning succeeded")
    print(f"delta_t            : {trajectory.delta_t}")
    print(f"num_control_points : {trajectory.num_control_points}")
    print(f"num_segments       : {trajectory.num_segments}")
    print(f"duration           : {trajectory.duration:.4f} s")

    pva_start_eval = trajectory.evaluate_pva(0.0)
    pva_end_eval = trajectory.evaluate_pva(trajectory.duration)

    start_error = np.linalg.norm(pva_start_eval - local_start_pva)
    end_error = np.linalg.norm(pva_end_eval - local_target_pva)

    print(f"local_start_pva    : {format_vec(local_start_pva)}")
    print(f"local_target_pva   : {format_vec(local_target_pva)}")
    print(f"eval @ 0           : {format_vec(pva_start_eval)}")
    print(f"eval @ T           : {format_vec(pva_end_eval)}")
    print(f"start error norm   : {start_error:.6e}")
    print(f"end error norm     : {end_error:.6e}")

    sample_times = np.linspace(0.0, trajectory.duration, num=5)
    print("sampled pva:")
    for t in sample_times:
        pva = trajectory.evaluate_pva(float(t))
        print(f"  t={t:8.4f} -> {format_vec(pva)}")


def main() -> None:
    planner = LocalPlanner(delta_t=0.5)

    test_cases: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "rest_to_rest_short",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ),
        (
            "moving_start_to_rest_target",
            np.array([0.0, 0.0, 0.0, 0.8, -0.3, 0.2, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ),
        (
            "rest_to_moving_target",
            np.array([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([4.0, 0.0, 1.5, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ),
    ]

    print("Debug LocalPlanner start")
    for case_name, local_start_pva, local_target_pva in test_cases:
        run_one_case(
            planner=planner,
            case_name=case_name,
            local_start_pva=local_start_pva,
            local_target_pva=local_target_pva,
            current_time=0.0,
        )


if __name__ == "__main__":
    main()
