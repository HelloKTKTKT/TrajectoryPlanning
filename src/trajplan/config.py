from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from pathlib import Path

import numpy as np
import yaml

from trajplan.quadrotor.model import QuadrotorPhysicalConfig, compute_inertia_matrix
from trajplan.planning.bspline_optimizer import BsplineOptimizerConfig
from trajplan.planning.local_planner import LocalPlannerConfig
from trajplan.planning.planner_manager import PlannerManagerConfig
from trajplan.map.grid_map import ObstacleConfig, GridMapConfig
from trajplan.shared_types import Matrix
from trajplan.df_controller import DifferentialFlatnessControllerConfig


@dataclass(slots=True)
class SwarmAgentScenario:
    agent_id: int
    initial_position: np.ndarray
    goal_positions: list[np.ndarray]


@dataclass(slots=True)
class SwarmScenarioConfig:
    agents: list[SwarmAgentScenario]


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

    return cast(dict[str, Any], data)


def load_project_configs(
    quadrotor_yaml_path: str | Path,
    planning_yaml_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    quadrotor_cfg = load_yaml(quadrotor_yaml_path)
    planning_cfg = load_yaml(planning_yaml_path)
    return quadrotor_cfg, planning_cfg


def load_swarm_scenario_config(yaml_path: str | Path) -> SwarmScenarioConfig:
    scenario_cfg = load_yaml(yaml_path)
    raw_agents = scenario_cfg.get("agents", [])
    if not isinstance(raw_agents, list):
        raise ValueError("Expected swarm scenario field 'agents' to be a list.")

    agents: list[SwarmAgentScenario] = []
    seen_agent_ids: set[int] = set()

    for raw_agent in raw_agents:
        if not isinstance(raw_agent, dict):
            raise ValueError("Expected each swarm agent entry to be a dict.")

        agent_id = int(raw_agent["agent_id"])
        if agent_id in seen_agent_ids:
            raise ValueError(f"Duplicated agent_id detected in scenario config: {agent_id}")
        seen_agent_ids.add(agent_id)

        initial_position = np.asarray(
            raw_agent["initial_position"],
            dtype=np.float64,
        ).reshape(-1)
        if initial_position.shape != (3,):
            raise ValueError(
                "Expected initial_position to have shape (3,), "
                f"but got {initial_position.shape} for agent_id={agent_id}."
            )

        raw_goal_positions = raw_agent.get("goal_positions", [])
        if not isinstance(raw_goal_positions, list):
            raise ValueError(
                f"Expected goal_positions to be a list for agent_id={agent_id}."
            )

        goal_positions: list[np.ndarray] = []
        for goal_position in raw_goal_positions:
            goal = np.asarray(goal_position, dtype=np.float64).reshape(-1)
            if goal.shape != (3,):
                raise ValueError(
                    "Expected each goal_position to have shape (3,), "
                    f"but got {goal.shape} for agent_id={agent_id}."
                )
            goal_positions.append(goal)

        agents.append(
            SwarmAgentScenario(
                agent_id=agent_id,
                initial_position=initial_position,
                goal_positions=goal_positions,
            )
        )

    return SwarmScenarioConfig(agents=agents)


def build_grid_map_config(
    planning_cfg: dict[str, Any],
) -> GridMapConfig:
    grid_map_cfg = planning_cfg["grid_map"]

    obstacles: list[ObstacleConfig] = []
    for obstacle_cfg in grid_map_cfg["obstacles"]:
        obstacles.append(
            ObstacleConfig(
                obs_start_index=np.asarray(
                    obstacle_cfg["obs_start_index"], dtype=np.int64
                ).reshape(3),
                obs_size=np.asarray(obstacle_cfg["obs_size"], dtype=np.int64).reshape(
                    3
                ),
            )
        )

    return GridMapConfig(
        resolution=float(grid_map_cfg["resolution"]),
        map_size=np.asarray(grid_map_cfg["map_size"], dtype=np.int64).reshape(3),
        origin=np.asarray(grid_map_cfg["origin"], dtype=np.float64).reshape(3),
        obstacles=obstacles,
    )


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
    lambda_swarm = float(planning_cfg["bspline_optimizer"]["lambda_swarm"])
    tol = float(planning_cfg["bspline_optimizer"]["tol"])
    swarm_clearance = float(planning_cfg["planner"]["swarm_clearance"])
    swarm_sample_divisor = int(
        planning_cfg["bspline_optimizer"]["swarm_sample_divisor"]
    )
    config_build = BsplineOptimizerConfig(
        max_vel=max_vel,
        max_acc=max_acc,
        safe_dist=safe_dist,
        lambda_smooth=lambda_smooth,
        lambda_dist=lambda_dist,
        lambda_feasibility=lambda_feasibility,
        lambda_fitness=lambda_fitness,
        lambda_swarm=lambda_swarm,
        tol=tol,
        swarm_clearance=swarm_clearance,
        swarm_sample_divisor=swarm_sample_divisor,
    )
    return config_build


def build_planner_manager_config(
    planning_cfg: dict[str, Any],
) -> PlannerManagerConfig:
    goal_tol = float(planning_cfg["planner"]["goal_tolerance"])
    planning_horizon = float(planning_cfg["planner"]["planning_horizon"])
    global_interpoint_dist_thresh = float(
        planning_cfg["planner"]["global_interpoint_dist_thresh"]
    )
    replan_thresh = float(planning_cfg["planner"]["replan_thresh"])
    no_replan_thresh = float(planning_cfg["planner"]["no_replan_thresh"])
    swarm_clearance = float(planning_cfg["planner"]["swarm_clearance"])
    swarm_replan_cooldown = float(planning_cfg["planner"]["swarm_replan_cooldown"])
    swarm_priority_enabled = bool(
        planning_cfg["planner"].get("swarm_priority_enabled", False)
    )
    safety_check_interval = float(planning_cfg["planner"]["safety_check_interval"])
    safety_sample_dt = float(planning_cfg["planner"]["safety_sample_dt"])
    emergency_time = float(planning_cfg["planner"]["emergency_time"])
    emergency_vel_threshold = float(planning_cfg["planner"]["emergency_vel_threshold"])
    emergency_stop_duration = float(planning_cfg["planner"]["emergency_stop_duration"])
    config_build = PlannerManagerConfig(
        goal_tol=goal_tol,
        planning_horizon=planning_horizon,
        global_interpoint_dist_thresh=global_interpoint_dist_thresh,
        replan_thresh=replan_thresh,
        no_replan_thresh=no_replan_thresh,
        swarm_clearance=swarm_clearance,
        swarm_replan_cooldown=swarm_replan_cooldown,
        swarm_priority_enabled=swarm_priority_enabled,
        safety_check_interval=safety_check_interval,
        safety_sample_dt=safety_sample_dt,
        emergency_time=emergency_time,
        emergency_vel_threshold=emergency_vel_threshold,
        emergency_stop_duration=emergency_stop_duration,
    )
    return config_build


def build_local_planner_config(
    quadrotor_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
) -> LocalPlannerConfig:
    max_vel = float(quadrotor_cfg["limits"]["max_velocity_3d"][0])
    max_acc = float(quadrotor_cfg["limits"]["max_acceleration_3d"][0])
    ctrl_pt_dist = float(planning_cfg["planner"]["ctrl_pt_dist"])
    planning_horizon = float(planning_cfg["planner"]["planning_horizon"])
    collision_weight_increase_factor = float(
        planning_cfg["planner"]["collision_weight_increase_factor"]
    )
    max_restart_num = int(planning_cfg["planner"]["max_restart_num"])
    goal_tol = float(planning_cfg["planner"]["goal_tolerance"])
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


def build_quadrotor_physical_config(
    quadrotor_cfg: dict[str, Any],
) -> QuadrotorPhysicalConfig:
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
            f"Expected positive box_mass, but got {box_mass}. Check mass and motor_weight."
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
    inertia_inv = np.asarray(
        np.linalg.inv(inertia_matrix),
        dtype=np.float64,
    )

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
    pseudo2rforce = np.asarray(
        np.linalg.inv(rforce2pseudo),
        dtype=np.float64,
    )

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


# def _as_diag_matrix_from_3x3_weights(
#     w_position: list[float] | tuple[float, float, float],
#     w_velocity: list[float] | tuple[float, float, float],
#     w_acceleration: list[float] | tuple[float, float, float],
# ) -> Matrix:
#     wp = np.asarray(w_position, dtype=np.float64).reshape(-1)
#     wv = np.asarray(w_velocity, dtype=np.float64).reshape(-1)
#     wa = np.asarray(w_acceleration, dtype=np.float64).reshape(-1)

#     if wp.shape != (3,):
#         raise ValueError(f"Expected 3 position weights, but got shape {wp.shape}.")
#     if wv.shape != (3,):
#         raise ValueError(f"Expected 3 velocity weights, but got shape {wv.shape}.")
#     if wa.shape != (3,):
#         raise ValueError(f"Expected 3 acceleration weights, but got shape {wa.shape}.")

#     return np.diag(np.hstack((wp, wv, wa)))


def _as_diag_matrix_from_3_weights(
    weights: list[float] | tuple[float, float, float],
) -> Matrix:
    weights_array = np.asarray(weights, dtype=np.float64).reshape(-1)

    if weights_array.shape != (3,):
        raise ValueError(f"Expected 3 weights, but got shape {weights_array.shape}.")

    return np.diag(weights_array)
