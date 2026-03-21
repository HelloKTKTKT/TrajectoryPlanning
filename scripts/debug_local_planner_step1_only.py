from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from trajplan.config import build_linear_mpc_config, load_project_configs
from trajplan.controller import Controller
from trajplan.map.grid_map import GridMap
from trajplan.planning.local_planner import LocalPlanner, LocalPlanningResult
from trajplan.planning.planner_manager import PlannerManager
from trajplan.quadrotor.agent import QuadrotorAgent
from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.sim_backend import SimBackend
from trajplan.runtime.sim_state_provider import SimStateProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class Step1OnlyLocalPlanner(LocalPlanner):
    """
    Debug-only planner that runs only Step 1:
    build initial bspline, then return immediately.
    """

    def plan(
        self,
        grid_map: GridMap,
        active_command,
        local_start_pva,
        local_target_pva,
        current_time: float,
    ) -> LocalPlanningResult:
        _ = grid_map

        dist_to_target = np.linalg.norm(local_target_pva[:3] - local_start_pva[:3])
        if dist_to_target < 0.2:
            return LocalPlanningResult(
                status="target_close",
                trajectory=None,
                message="Local target is already very close.",
            )

        initial_bspline = self._build_initial_bspline(
            active_command=active_command,
            local_start_pva=local_start_pva,
            local_target_pva=local_target_pva,
            current_time=current_time,
        )

        if initial_bspline is None:
            return LocalPlanningResult(
                status="failure",
                trajectory=None,
                message="Initial B-spline generation failed.",
            )

        return LocalPlanningResult(
            status="success",
            trajectory=initial_bspline,
            message="Step-1-only initial B-spline generated successfully.",
        )


def format_vec(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=4, suppress_small=True)


def main() -> None:
    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )
    global_lmpc_config = build_linear_mpc_config(
        quadrotor_cfg=quadrotor_cfg,
        planning_cfg=planning_cfg,
    )

    planning_horizon = float(planning_cfg["local_planner"]["planning_horizon"])
    ctrl_pt_dist = float(planning_cfg["local_planner"]["ctrl_pt_dist"])
    goal_tolerance = float(planning_cfg["goal"]["tolerance"])

    max_v = float(
        np.max(np.asarray(quadrotor_cfg["limits"]["max_velocity_3d"], dtype=np.float64))
    )
    max_a = float(
        np.max(
            np.asarray(quadrotor_cfg["limits"]["max_acceleration_3d"], dtype=np.float64)
        )
    )

    # Empty grid map. No obstacles in this debug.
    grid_map = GridMap(
        map_size=np.array([200, 200, 50], dtype=np.int64),
        resolution=0.2,
        origin=np.array([-5.0, -5.0, -1.0], dtype=np.float64),
    )

    local_planner = Step1OnlyLocalPlanner(
        ctrl_pt_dist=ctrl_pt_dist,
        planning_horizon=planning_horizon,
        max_v=max_v,
        max_a=max_a,
    )

    planner_manager = PlannerManager(
        local_planner=local_planner,
        global_lmpc_config=global_lmpc_config,
        grid_map=grid_map,
        goal_tolerance=goal_tolerance,
        planning_horizon=planning_horizon,
    )

    controller = Controller(
        kp_position=2.0,
        kv_velocity=1.5,
        max_acceleration_norm=max_a,
    )

    initial_state = QuadrotorState(
        pva=np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
    )

    state_provider = SimStateProvider(initial_state=initial_state)
    sim_backend = SimBackend(
        state_provider=state_provider,
        max_acceleration_norm=max_a,
    )
    agent = QuadrotorAgent(
        controller=controller,
        state_provider=state_provider,
        backend=sim_backend,
    )

    # Two goals, each separated by more than planning_horizon.
    goal_1 = np.array([2.0 * planning_horizon, 0.0, 0.0], dtype=np.float64)
    goal_2 = np.array([4.0 * planning_horizon, 0.0, 0.0], dtype=np.float64)

    planner_manager.add_goal(goal_1)
    planner_manager.add_goal(goal_2)

    dt_exec = 0.02
    dt_plan = 0.20
    sim_time = 0.0
    max_sim_time = 30.0
    next_plan_time = 0.0

    step_count = 0
    log_every_n_steps = 10

    print("=" * 100)
    print("Debug Step-1-only LocalPlanner")
    print(f"planning_horizon : {planning_horizon}")
    print(f"ctrl_pt_dist     : {ctrl_pt_dist}")
    print(f"max_v            : {max_v}")
    print(f"max_a            : {max_a}")
    print(f"goal_1           : {format_vec(goal_1)}")
    print(f"goal_2           : {format_vec(goal_2)}")
    print("=" * 100)

    while not agent.is_finished and sim_time <= max_sim_time:
        current_state = state_provider.get_state()

        if sim_time >= next_plan_time - 1e-12:
            new_command = planner_manager.update_and_get_command(
                current_state=current_state,
                current_time=sim_time,
                active_command=agent.active_command,
            )
            agent.set_active_command(new_command)
            next_plan_time += dt_plan

        agent.step(dt=dt_exec)

        sim_time += dt_exec
        step_count += 1

        if step_count % log_every_n_steps == 0:
            state_now = state_provider.get_state()
            position = state_now.pva[:3]
            velocity = state_now.pva[3:6]

            if agent.active_command is None:
                command_mode = "none"
                traj_duration = None
            else:
                command_mode = agent.active_command.mode.value
                traj_duration = (
                    None
                    if agent.active_command.trajectory is None
                    else agent.active_command.trajectory.duration
                )

            active_goal = planner_manager.active_goal_position
            active_goal_str = "None" if active_goal is None else format_vec(active_goal)

            print(
                f"[t={sim_time:6.2f}s] "
                f"pos={format_vec(position)} "
                f"vel={format_vec(velocity)} "
                f"cmd={command_mode} "
                f"traj_T={traj_duration} "
                f"goal={active_goal_str} "
                f"global_progress={planner_manager.global_progress:.2f}"
            )

    final_state = state_provider.get_state()

    print("\n" + "=" * 100)
    print("Debug finished")
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


if __name__ == "__main__":
    main()
