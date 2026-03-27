from __future__ import annotations
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from trajplan.config import (
    load_project_configs,
    build_bspline_optimizer_config,
    build_local_planner_config,
    build_planner_manager_config,
    build_linear_mpc_config,
)
from trajplan.planning.bspline_optimizer import BsplineOptimizer
from trajplan.planning.local_planner import LocalPlanner
from trajplan.planning.planner_manager import PlannerManager
from trajplan.map.grid_map import GridMap

from trajplan.controller import Controller
from trajplan.quadrotor.agent import QuadrotorAgent
from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.sim_backend import SimBackend
from trajplan.runtime.sim_state_provider import SimStateProvider


# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# SRC_ROOT = PROJECT_ROOT / "src"
# if str(SRC_ROOT) not in sys.path:
#     sys.path.insert(0, str(SRC_ROOT))


def format_vec(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=4, suppress_small=True)


def _show_plan_step(
    grid_map,
    agent_pos: np.ndarray,
    global_target: np.ndarray | None,
    new_command,
    sim_time: float,
) -> None:
    """Pop-up a blocking 3D figure every time a new command is issued."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    grid_map.plot_3d(ax=ax)

    # Current agent position
    ax.scatter(
        *agent_pos, s=120, marker="o", color="dodgerblue", zorder=5, label="agent"
    )

    # Active global target
    if global_target is not None:
        ax.scatter(
            *global_target, s=160, marker="*", color="orange", zorder=5, label="target"
        )

    # Local B-spline trajectory: sample from t=0 to duration
    if (
        new_command is not None
        and new_command.is_track
        and new_command.trajectory is not None
    ):
        traj = new_command.trajectory
        ts = np.arange(0.0, traj.duration + 1e-3, 0.05)
        pts = np.array([traj.evaluate_pva(float(t))[:3] for t in ts])
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color="limegreen",
            linewidth=2,
            label="local traj",
        )
        ax.scatter(*pts[-1], s=80, marker="D", color="limegreen", zorder=5)

    ax.set_title(f"Plan update  t = {sim_time:.2f} s")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()  # blocking — simulation pauses until window is closed


def _show_final_trajectory(
    grid_map,
    pos_history: list[np.ndarray],
) -> None:
    """Show the full flown path overlaid on the obstacle map."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    grid_map.plot_3d(ax=ax)

    path = np.vstack(pos_history)  # (N, 3)
    ax.plot(
        path[:, 0],
        path[:, 1],
        path[:, 2],
        color="tomato",
        linewidth=2,
        label="flown path",
    )
    ax.scatter(*path[0], s=120, marker="o", color="limegreen", zorder=5, label="start")
    ax.scatter(*path[-1], s=120, marker="D", color="orange", zorder=5, label="end")

    ax.set_title("Full flown trajectory")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


def build_bspline_optimizer(
    quadrotor_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
) -> BsplineOptimizer:
    bspline_optimizer_cfg = build_bspline_optimizer_config(
        quadrotor_cfg=quadrotor_cfg,
        planning_cfg=planning_cfg,
    )
    return BsplineOptimizer(config=bspline_optimizer_cfg)


def build_grid_map(planning_cfg: dict[str, Any]) -> GridMap:
    resolution = float(planning_cfg["grid_map"]["resolution"])
    origin = np.zeros(3, dtype=np.float64)
    size = np.asarray(planning_cfg["grid_map"]["map_size"], dtype=np.int64)
    grid_map = GridMap(resolution=resolution, origin=origin, map_size=size)

    grid_map.add_obstacle(
        obs_start_index=np.array([10, 30, 0], dtype=np.int64),
        obs_size=np.array([30, 10, 30], dtype=np.int64),
    )

    grid_map.add_obstacle(
        obs_start_index=np.array([20, 10, 0], dtype=np.int64),
        obs_size=np.array([20, 10, 30], dtype=np.int64),
    )

    return grid_map


def main() -> None:
    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )

    grid_map = build_grid_map(planning_cfg)

    bspline_optimizer = build_bspline_optimizer(quadrotor_cfg, planning_cfg)

    local_planner_cfg = build_local_planner_config(quadrotor_cfg, planning_cfg)
    local_planner = LocalPlanner(config=local_planner_cfg, optimizer=bspline_optimizer)

    planner_manager_cfg = build_planner_manager_config(planning_cfg)

    linear_mpc_cfg = build_linear_mpc_config(quadrotor_cfg, planning_cfg)

    planner_manager = PlannerManager(
        local_planner=local_planner,
        global_lmpc_config=linear_mpc_cfg,
        grid_map=grid_map,
        config=planner_manager_cfg,
    )

    controller = Controller(
        kp_position=2.0,
        kv_velocity=1.5,
        max_acceleration_norm=float(
            np.max(
                np.asarray(
                    quadrotor_cfg["limits"]["max_acceleration_3d"], dtype=np.float64
                )
            )
        ),
    )

    initial_state = QuadrotorState(
        pva=np.array(
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
    )

    state_provider = SimStateProvider(initial_state=initial_state)

    sim_backend = SimBackend(
        state_provider=state_provider,
        max_acceleration_norm=float(
            np.max(
                np.asarray(
                    quadrotor_cfg["limits"]["max_acceleration_3d"], dtype=np.float64
                )
            )
        ),
    )

    agent = QuadrotorAgent(
        controller=controller,
        state_provider=state_provider,
        backend=sim_backend,
    )

    planner_manager.add_goal(np.array([4.5, 4.5, 1.0], dtype=np.float64))
    planner_manager.add_goal(np.array([1.0, 2.0, 0.5], dtype=np.float64))

    # ------------------------------------------------------------------
    # Main loop settings
    # ------------------------------------------------------------------
    dt_exec = 0.01
    dt_plan = 1.0

    sim_time = 0.0
    max_sim_time = 30.0

    next_plan_time = 0.0
    step_count = 0
    log_every_n_steps = 10

    print("=" * 100)
    print("Start minimal simulation demo")
    print(f"dt_exec     : {dt_exec}")
    print(f"dt_plan     : {dt_plan}")
    print(f"max_sim_time: {max_sim_time}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    pos_history: list[np.ndarray] = []

    while not agent.is_finished and sim_time <= max_sim_time:
        current_state = state_provider.get_state()
        pos_history.append(current_state.pva[:3].copy())

        if sim_time >= next_plan_time - 1e-12:
            new_command = planner_manager.update_and_get_command(
                current_state=current_state,
                current_time=sim_time,
                active_command=agent.active_command,
            )
            agent.set_active_command(new_command)
            next_plan_time += dt_plan

            if new_command.mode.value == "track":
                print(f"new traj duration: {new_command.trajectory.duration:.2f} s")
                pva_start = new_command.trajectory.evaluate_pva(0.0)
                print(f"new traj start pva: {format_vec(pva_start)}")

            print(f"new command: {new_command.mode}")

            _show_plan_step(
                grid_map=grid_map,
                agent_pos=current_state.pva[:3].copy(),
                global_target=planner_manager.active_goal_position,
                new_command=new_command,
                sim_time=sim_time,
            )

        agent.step(dt=dt_exec, now_time=sim_time)

        sim_time += dt_exec
        step_count += 1

        if step_count % log_every_n_steps == 0:
            state_now = state_provider.get_state()
            position = state_now.pva[:3]
            velocity = state_now.pva[3:6]

            if agent.active_command is None:
                command_mode = "none"
            else:
                command_mode = agent.active_command.mode.value

            active_goal = planner_manager.active_goal_position
            active_goal_str = "None" if active_goal is None else format_vec(active_goal)

            print(
                f"[t={sim_time:6.2f}s] "
                f"pos={format_vec(position)} "
                f"vel={format_vec(velocity)} "
                f"cmd={command_mode} "
                f"goal={active_goal_str} "
                f"global_progress={planner_manager.global_progress:.2f}"
            )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    final_state = state_provider.get_state()

    print("\n" + "=" * 100)
    print("Simulation finished")
    print(f"agent.is_finished : {agent.is_finished}")
    print(f"sim_time          : {sim_time:.2f}")
    print(f"final position    : {format_vec(final_state.pva[:3])}")
    print(f"final velocity    : {format_vec(final_state.pva[3:6])}")
    print(f"remaining goals   : {len(planner_manager.goal_position_list)}")
    print(
        "active goal       :",
        "None"
        if planner_manager.active_goal_position is None
        else format_vec(planner_manager.active_goal_position),
    )
    print("=" * 100)

    _show_final_trajectory(grid_map, pos_history)


if __name__ == "__main__":
    main()
