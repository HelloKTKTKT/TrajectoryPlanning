from __future__ import annotations

from pathlib import Path
from typing import Any

from trajplan.planning.bspline_optimizer import BsplineOptimizerConfig
from trajplan.planning.local_planner import LocalPlannerConfig
from trajplan.planning.planner_manager import PlannerManagerConfig
from trajplan.trajectory.linear_mpc import LinearMpcConfig

from trajplan.quadrotor.model import (
    QuadrotorPhysicalConfig,
    compute_inertia_matrix,
)
from trajplan.controller import DifferentialFlatnessControllerConfig


from trajplan.shared_types import Matrix

import numpy as np
import yaml


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


def build_bspline_optimizer_config(
    quadrotor_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
) -> BsplineOptimizerConfig:
    max_vel = float(quadrotor_cfg["limits"]["max_velocity_3d"][0])
    max_acc = float(quadrotor_cfg["limits"]["max_acceleration_3d"][0])
    safe_dist = float(planning_cfg["bspline_optimizer"]["safe_dist"])
    lambda_smooth = float(planning_cfg["bspline_optimizer"]["lambda_smooth"])
    lambda_dist = float(planning_cfg["bspline_optimizer"]["lambda_dist"])
    lambda_feasibility = float(planning_cfg["bspline_optimizer"]["lambda_feasibility"])
    lambda_fitness = float(planning_cfg["bspline_optimizer"]["lambda_fitness"])
    tol = float(planning_cfg["bspline_optimizer"]["tol"])
    config_build = BsplineOptimizerConfig(
        max_vel=max_vel,
        max_acc=max_acc,
        safe_dist=safe_dist,
        lambda_smooth=lambda_smooth,
        lambda_dist=lambda_dist,
        lambda_feasibility=lambda_feasibility,
        lambda_fitness=lambda_fitness,
        tol=tol,
    )
    return config_build


def build_local_planner_config(
    quadrotor_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
) -> LocalPlannerConfig:
    max_vel = float(quadrotor_cfg["limits"]["max_velocity_3d"][0])
    max_acc = float(quadrotor_cfg["limits"]["max_acceleration_3d"][0])
    ctrl_pt_dist = float(planning_cfg["local_planner"]["ctrl_pt_dist"])
    planning_horizon = float(planning_cfg["local_planner"]["planning_horizon"])
    collision_weight_increase_factor = float(
        planning_cfg["local_planner"]["collision_weight_increase_factor"]
    )
    max_restart_num = int(planning_cfg["local_planner"]["max_restart_num"])
    goal_tol = float(planning_cfg["goal"]["tolerance"])
    lambda_dist_initial = float(planning_cfg["bspline_optimizer"]["lambda_dist"])
    config_build = LocalPlannerConfig(
        max_vel=max_vel,
        max_acc=max_acc,
        ctrl_pt_dist=ctrl_pt_dist,
        planning_horizon=planning_horizon,
        collision_weight_increase_factor=collision_weight_increase_factor,
        max_restart_num=max_restart_num,
        goal_tol=goal_tol,
        lambda_dist_initial=lambda_dist_initial,
    )
    return config_build


def build_planner_manager_config(
    planning_cfg: dict[str, Any],
) -> PlannerManagerConfig:
    goal_tol = float(planning_cfg["goal"]["tolerance"])
    planning_horizon = float(planning_cfg["local_planner"]["planning_horizon"])
    config_build = PlannerManagerConfig(
        goal_tol=goal_tol,
        planning_horizon=planning_horizon,
    )
    return config_build


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


def build_quadrotor_physical_config(
    quadrotor_cfg: dict[str, Any],
) -> QuadrotorPhysicalConfig:
    if "physical" not in quadrotor_cfg:
        raise ValueError("quadrotor config must contain 'physical'.")

    physical_cfg = quadrotor_cfg["physical"]

    rotor_count = int(physical_cfg["rotor_count"])

    mass = float(physical_cfg["mass"])
    arm_length = float(physical_cfg["arm_length"])
    motor_weight = float(physical_cfg["motor_weight"])

    body_box_size = np.asarray(
        physical_cfg["body_box_size"],
        dtype=np.float64,
    ).reshape(-1)
    if body_box_size.shape != (3,):
        raise ValueError(
            f"Expected body_box_size shape (3,), but got {body_box_size.shape}."
        )
    box_x, box_y, box_z = body_box_size.tolist()

    body_real_z = float(physical_cfg["body_real_z"])
    align_angle_rad = float(physical_cfg["align_angle_rad"])

    air_density = float(physical_cfg["air_density"])
    propeller_diameter = float(physical_cfg["propeller_diameter"])
    thrust_coefficient = float(physical_cfg["thrust_coefficient"])
    power_coefficient = float(physical_cfg["power_coefficient"])
    max_rpm = float(physical_cfg["max_rpm"])

    if rotor_count != 4:
        raise ValueError(
            f"Current implementation expects rotor_count == 4, but got {rotor_count}."
        )
    if mass <= 0.0:
        raise ValueError(f"Expected mass > 0, but got {mass}.")
    if arm_length <= 0.0:
        raise ValueError(f"Expected arm_length > 0, but got {arm_length}.")
    if motor_weight < 0.0:
        raise ValueError(f"Expected motor_weight >= 0, but got {motor_weight}.")
    if np.any(body_box_size <= 0.0):
        raise ValueError(
            f"Expected positive body_box_size entries, but got {body_box_size}."
        )
    if air_density <= 0.0:
        raise ValueError(f"Expected air_density > 0, but got {air_density}.")
    if propeller_diameter <= 0.0:
        raise ValueError(
            f"Expected propeller_diameter > 0, but got {propeller_diameter}."
        )
    if thrust_coefficient <= 0.0:
        raise ValueError(
            f"Expected thrust_coefficient > 0, but got {thrust_coefficient}."
        )
    if power_coefficient <= 0.0:
        raise ValueError(
            f"Expected power_coefficient > 0, but got {power_coefficient}."
        )
    if max_rpm <= 0.0:
        raise ValueError(f"Expected max_rpm > 0, but got {max_rpm}.")

    max_rps = max_rpm / 60.0
    max_force = thrust_coefficient * air_density * propeller_diameter**4 * max_rps**2
    max_rotor_torque = (
        power_coefficient
        * air_density
        * propeller_diameter**5
        * max_rps**2
        / (2.0 * np.pi)
    )
    torque_force_ratio = max_rotor_torque / max_force

    box_mass = mass - rotor_count * motor_weight
    if box_mass <= 0.0:
        raise ValueError(
            f"Expected positive box_mass, but got {box_mass}. "
            "Check mass and motor_weight."
        )

    proj_len = arm_length * np.sin(align_angle_rad)
    quad_poses = np.array(
        [
            [proj_len, proj_len, body_real_z],
            [-proj_len, -proj_len, body_real_z],
            [proj_len, -proj_len, body_real_z],
            [-proj_len, proj_len, body_real_z],
        ],
        dtype=np.float64,
    )

    inertia_matrix = compute_inertia_matrix(
        box_x=box_x,
        box_y=box_y,
        box_z=box_z,
        box_mass=box_mass,
        motor_weight=motor_weight,
        quad_poses=quad_poses,
    )
    inertia_inv = np.linalg.inv(inertia_matrix)

    rforce2pseudo = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [proj_len, -proj_len, proj_len, -proj_len],
            [-proj_len, proj_len, proj_len, -proj_len],
            [
                -torque_force_ratio,
                -torque_force_ratio,
                torque_force_ratio,
                torque_force_ratio,
            ],
        ],
        dtype=np.float64,
    )
    pseudo2rforce = np.linalg.inv(rforce2pseudo)

    return QuadrotorPhysicalConfig(
        rotor_count=rotor_count,
        mass=mass,
        arm_length=arm_length,
        motor_weight=motor_weight,
        body_box_size=body_box_size,
        body_real_z=body_real_z,
        align_angle_rad=align_angle_rad,
        air_density=air_density,
        propeller_diameter=propeller_diameter,
        thrust_coefficient=thrust_coefficient,
        power_coefficient=power_coefficient,
        max_rpm=max_rpm,
        max_rps=max_rps,
        max_force=max_force,
        max_rotor_torque=max_rotor_torque,
        torque_force_ratio=torque_force_ratio,
        box_mass=box_mass,
        quad_poses=quad_poses,
        inertia_matrix=inertia_matrix,
        inertia_inv=inertia_inv,
        rforce2pseudo=rforce2pseudo,
        pseudo2rforce=pseudo2rforce,
    )


def build_differential_flatness_controller_config(
    quadrotor_cfg: dict[str, Any],
) -> DifferentialFlatnessControllerConfig:
    if "controller" not in quadrotor_cfg:
        raise ValueError("quadrotor config must contain 'controller'.")

    controller_cfg = quadrotor_cfg["controller"]

    if "differential_flatness" not in controller_cfg:
        raise ValueError(
            "quadrotor config['controller'] must contain 'differential_flatness'."
        )

    df_cfg = controller_cfg["differential_flatness"]

    kp = _as_diag_matrix_from_3_weights(df_cfg["kp"])
    kv = _as_diag_matrix_from_3_weights(df_cfg["kv"])
    ka = _as_diag_matrix_from_3_weights(df_cfg["ka"])
    kvi = _as_diag_matrix_from_3_weights(df_cfg["kvi"])

    kq = _as_diag_matrix_from_3_weights(df_cfg["kq"])
    kw = _as_diag_matrix_from_3_weights(df_cfg["kw"])

    bw_max = np.asarray(df_cfg["bw_max"], dtype=np.float64).reshape(-1)
    if bw_max.shape != (3,):
        raise ValueError(f"Expected bw_max shape (3,), but got {bw_max.shape}.")

    qp_weight_diag = np.asarray(df_cfg["qp_weight"], dtype=np.float64).reshape(-1)
    if qp_weight_diag.shape != (4,):
        raise ValueError(
            f"Expected qp_weight shape (4,), but got {qp_weight_diag.shape}."
        )
    qp_weight = np.diag(qp_weight_diag)

    return DifferentialFlatnessControllerConfig(
        kp=kp,
        kv=kv,
        ka=ka,
        kvi=kvi,
        kq=kq,
        kw=kw,
        ep_max=float(df_cfg["ep_max"]),
        ev_max=float(df_cfg["ev_max"]),
        bw_max=bw_max,
        sat_int_ev=float(df_cfg["sat_int_ev"]),
        minimum_thrust=float(df_cfg["minimum_thrust"]),
        maximum_thrust=float(df_cfg["maximum_thrust"]),
        epsilon=float(df_cfg["epsilon"]),
        rotor_force_min=float(df_cfg["rotor_force_min"]),
        rotor_force_max_scale=float(df_cfg["rotor_force_max_scale"]),
        qp_weight=qp_weight,
    )


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


def _as_diag_matrix_from_3_weights(
    weights: list[float] | tuple[float, float, float],
) -> Matrix:
    weights_array = np.asarray(weights, dtype=np.float64).reshape(-1)

    if weights_array.shape != (3,):
        raise ValueError(f"Expected 3 weights, but got shape {weights_array.shape}.")

    return np.diag(weights_array)
