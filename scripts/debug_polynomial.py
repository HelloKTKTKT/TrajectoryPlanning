from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from trajplan.trajectory.polynomial import OneSegmentMinimumJerkTrajectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def format_vec(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=4, suppress_small=True)


def run_one_case(
    case_name: str,
    start_position: np.ndarray,
    end_position: np.ndarray,
    start_velocity: np.ndarray,
    end_velocity: np.ndarray,
    start_acceleration: np.ndarray,
    end_acceleration: np.ndarray,
    duration: float,
) -> None:
    print("=" * 80)
    print(f"Case: {case_name}")

    traj = OneSegmentMinimumJerkTrajectory(
        start_position=start_position,
        end_position=end_position,
        start_velocity=start_velocity,
        end_velocity=end_velocity,
        start_acceleration=start_acceleration,
        end_acceleration=end_acceleration,
        duration=duration,
    )

    pva_0 = traj.evaluate_pva(0.0)
    pva_T = traj.evaluate_pva(duration)

    start_pva_expected = np.hstack((start_position, start_velocity, start_acceleration))
    end_pva_expected = np.hstack((end_position, end_velocity, end_acceleration))

    start_error = np.linalg.norm(pva_0 - start_pva_expected)
    end_error = np.linalg.norm(pva_T - end_pva_expected)

    print(f"duration          : {duration:.4f} s")
    print(f"start expected    : {format_vec(start_pva_expected)}")
    print(f"eval @ 0          : {format_vec(pva_0)}")
    print(f"start error norm  : {start_error:.6e}")
    print(f"end expected      : {format_vec(end_pva_expected)}")
    print(f"eval @ T          : {format_vec(pva_T)}")
    print(f"end error norm    : {end_error:.6e}")

    sample_times = np.linspace(0.0, duration, num=5)
    print("sampled pva:")
    for t in sample_times:
        pva = traj.evaluate_pva(float(t))
        print(f"  t={t:8.4f} -> {format_vec(pva)}")

    # Check clamp behavior
    pva_before = traj.evaluate_pva(-1.0)
    pva_after = traj.evaluate_pva(duration + 1.0)

    print("clamp check:")
    print(f"  eval @ -1.0     : {format_vec(pva_before)}")
    print(f"  eval @ T+1      : {format_vec(pva_after)}")


def main() -> None:
    test_cases = [
        (
            "rest_to_rest",
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([2.0, 1.0, 0.5], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            2.0,
        ),
        (
            "moving_start_to_rest_end",
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([3.0, 2.0, 1.0], dtype=np.float64),
            np.array([0.8, -0.3, 0.2], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            2.5,
        ),
        (
            "rest_start_to_moving_end",
            np.array([1.0, -1.0, 0.5], dtype=np.float64),
            np.array([4.0, 0.0, 1.5], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.5, 0.2, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            3.0,
        ),
    ]

    print("Debug OneSegmentMinimumJerkTrajectory start")
    for case in test_cases:
        run_one_case(*case)


if __name__ == "__main__":
    main()
