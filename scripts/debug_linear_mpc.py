from __future__ import annotations
import numpy as np

from trajplan.config import build_linear_mpc_config, load_project_configs
from trajplan.trajectory.linear_mpc import LinearMpcTrajectory

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _format_vector(x: np.ndarray) -> str:
    return np.array2string(x, precision=4, suppress_small=True)


def run_one_case(
    case_name: str,
    trajectory: LinearMpcTrajectory,
    start_state: np.ndarray,
    target_state: np.ndarray,
) -> None:
    print("=" * 80)
    print(f"Case: {case_name}")

    trajectory.update_traj_info(
        start_state=start_state,
        target_state=target_state,
    )

    print(f"status      : {trajectory.status}")
    print(f"duration    : {trajectory.duration:.4f} s")
    print(
        f"num_samples : {0 if trajectory.Xpred is None else trajectory.Xpred.shape[0]}"
    )

    if trajectory.status != "optimal" or trajectory.Xpred is None:
        print("MPC did not solve successfully.")
        return

    pva_start_eval = trajectory.evaluate_pva(0.0)
    pva_end_eval = trajectory.evaluate_pva(trajectory.duration)

    start_error = np.linalg.norm(pva_start_eval - start_state.reshape(-1))
    end_error = np.linalg.norm(pva_end_eval - target_state.reshape(-1))

    print(f"start_state : {_format_vector(start_state.reshape(-1))}")
    print(f"target_state: {_format_vector(target_state.reshape(-1))}")
    print(f"eval @ 0    : {_format_vector(pva_start_eval)}")
    print(f"eval @ T    : {_format_vector(pva_end_eval)}")
    print(f"start error : {start_error:.6e}")
    print(f"end error   : {end_error:.6e}")

    sample_times = np.linspace(
        0.0, trajectory.duration, num=min(5, max(2, trajectory.Xpred.shape[0]))
    )
    print("sampled pva :")
    for t in sample_times:
        pva = trajectory.evaluate_pva(float(t))
        print(f"  t={t:8.4f} -> {_format_vector(pva)}")


def main() -> None:
    quadrotor_yaml_path = PROJECT_ROOT / "configs" / "quadrotor.yaml"
    planning_yaml_path = PROJECT_ROOT / "configs" / "planning.yaml"

    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=quadrotor_yaml_path,
        planning_yaml_path=planning_yaml_path,
    )

    lmpc_config = build_linear_mpc_config(
        quadrotor_cfg=quadrotor_cfg,
        planning_cfg=planning_cfg,
    )

    trajectory = LinearMpcTrajectory(config=lmpc_config)

    test_cases: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "rest_to_rest_short",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ),
        (
            "rest_to_rest_longer",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([5.0, -3.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ),
        (
            "moving_start_to_rest_target",
            np.array([0.0, 0.0, 0.0, 0.8, -0.3, 0.2, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ),
    ]

    print("Loaded configs successfully.")
    print(f"MPC ts                  : {lmpc_config.ts}")
    print(f"MPC planning horizon    : {lmpc_config.planning_horizon_steps}")
    print(f"Max velocity 3D         : {lmpc_config.max_velocity_3d}")
    print(f"Max acceleration 3D     : {lmpc_config.max_acceleration_3d}")
    print(f"Max jerk 3D             : {lmpc_config.max_jerk_3d}")

    for case_name, start_state, target_state in test_cases:
        run_one_case(
            case_name=case_name,
            trajectory=trajectory,
            start_state=start_state,
            target_state=target_state,
        )


if __name__ == "__main__":
    main()
