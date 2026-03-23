from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from trajplan.shared_types import Matrix
from trajplan.trajectory.linear_mpc import LinearMpcConfig
from trajplan.planning.bspline_optimizer import BsplineOptimizerConfig


def load_yaml(yaml_path: str | Path) -> dict[str, Any]:
    path = Path(yaml_path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected top-level YAML content to be a dict, but got {type(data)}."
        )

    return data


def load_project_configs(
    quadrotor_yaml_path: str | Path,
    planning_yaml_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    quadrotor_cfg = load_yaml(quadrotor_yaml_path)
    planning_cfg = load_yaml(planning_yaml_path)
    return quadrotor_cfg, planning_cfg


def _as_diag_matrix_from_3x3_weights(
    w_position: list[float] | tuple[float, float, float],
    w_velocity: list[float] | tuple[float, float, float],
    w_acceleration: list[float] | tuple[float, float, float],
) -> Matrix:
    wp = np.asarray(w_position, dtype=np.float64).reshape(-1)
    wv = np.asarray(w_velocity, dtype=np.float64).reshape(-1)
    wa = np.asarray(w_acceleration, dtype=np.float64).reshape(-1)

    if wp.shape != (3,):
        raise ValueError(f"Expected 3 position weights, but got shape {wp.shape}.")
    if wv.shape != (3,):
        raise ValueError(f"Expected 3 velocity weights, but got shape {wv.shape}.")
    if wa.shape != (3,):
        raise ValueError(f"Expected 3 acceleration weights, but got shape {wa.shape}.")

    return np.diag(np.hstack((wp, wv, wa)))


def _build_triple_integrator_discrete_model(ts: float) -> tuple[Matrix, Matrix]:
    if ts <= 0.0:
        raise ValueError(f"Expected ts > 0, but got {ts}.")

    I3 = np.eye(3, dtype=np.float64)
    Z3 = np.zeros((3, 3), dtype=np.float64)

    Ad = np.block(
        [
            [I3, ts * I3, 0.5 * ts**2 * I3],
            [Z3, I3, ts * I3],
            [Z3, Z3, I3],
        ]
    )

    Bd = np.vstack(
        (
            (ts**3 / 6.0) * I3,
            (ts**2 / 2.0) * I3,
            ts * I3,
        )
    )

    return Ad, Bd


def build_linear_mpc_config(
    quadrotor_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
) -> LinearMpcConfig:
    if "limits" not in quadrotor_cfg:
        raise ValueError("quadrotor config must contain 'limits'.")

    if "global_linear_mpc" not in planning_cfg:
        raise ValueError("planning config must contain 'global_linear_mpc'.")

    limits_cfg = quadrotor_cfg["limits"]
    mpc_cfg = planning_cfg["global_linear_mpc"]

    ts = float(mpc_cfg["ts"])
    planning_horizon_steps = int(mpc_cfg["planning_horizon_steps"])

    max_velocity_3d = np.asarray(
        limits_cfg["max_velocity_3d"],
        dtype=np.float64,
    ).reshape(-1)
    max_acceleration_3d = np.asarray(
        limits_cfg["max_acceleration_3d"],
        dtype=np.float64,
    ).reshape(-1)
    max_jerk_3d = np.asarray(
        limits_cfg["max_jerk_3d"],
        dtype=np.float64,
    ).reshape(-1)

    if max_velocity_3d.shape != (3,):
        raise ValueError(
            f"Expected max_velocity_3d shape (3,), but got {max_velocity_3d.shape}."
        )
    if max_acceleration_3d.shape != (3,):
        raise ValueError(
            "Expected max_acceleration_3d shape (3,), "
            f"but got {max_acceleration_3d.shape}."
        )
    if max_jerk_3d.shape != (3,):
        raise ValueError(
            f"Expected max_jerk_3d shape (3,), but got {max_jerk_3d.shape}."
        )

    Ad, Bd = _build_triple_integrator_discrete_model(ts)

    Q = _as_diag_matrix_from_3x3_weights(
        w_position=mpc_cfg["q_position"],
        w_velocity=mpc_cfg["q_velocity"],
        w_acceleration=mpc_cfg["q_acceleration"],
    )
    QN = _as_diag_matrix_from_3x3_weights(
        w_position=mpc_cfg["qn_position"],
        w_velocity=mpc_cfg["qn_velocity"],
        w_acceleration=mpc_cfg["qn_acceleration"],
    )

    r_jerk = np.asarray(mpc_cfg["r_jerk"], dtype=np.float64).reshape(-1)
    if r_jerk.shape != (3,):
        raise ValueError(f"Expected r_jerk shape (3,), but got {r_jerk.shape}.")
    R = np.diag(r_jerk)

    return LinearMpcConfig(
        ts=ts,
        planning_horizon_steps=planning_horizon_steps,
        Ad=Ad,
        Bd=Bd,
        Q=Q,
        QN=QN,
        R=R,
        max_velocity_3d=max_velocity_3d,
        max_acceleration_3d=max_acceleration_3d,
        max_jerk_3d=max_jerk_3d,
    )


def build_local_optimizer_config(
    quadrotor_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
) -> BsplineOptimizerConfig:
    raise NotImplementedError("Local optimizer config builder is not implemented yet.")
