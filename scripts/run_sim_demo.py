from __future__ import annotations

from pathlib import Path
import sys
# import time

import numpy as np


from trajplan.config import build_linear_mpc_config, load_project_configs
from trajplan.controller import Controller
from trajplan.planning.local_planner import LocalPlanner
from trajplan.planning.planner_manager import PlannerManager
from trajplan.quadrotor.agent import QuadrotorAgent
from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.sim_backend import SimBackend
from trajplan.runtime.sim_state_provider import SimStateProvider
from trajplan.map.grid_map import grid_map


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def format_vec(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=4, suppress_small=True)


def main() -> None:
    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )
    global_lmpc_config = build_linear_mpc_config(
        quadrotor_cfg=quadrotor_cfg,
        planning_cfg=planning_cfg,
    )

    # ------------------------------------------------------------------
    # Build modules
    # ------------------------------------------------------------------
    local_planner = LocalPlanner(
        ctrl_pt_dist=float(planning_cfg["local_planner"]["ctrl_pt_dist"]),
        planning_horizon=float(planning_cfg["local_planner"]["planning_horizon"]),
        max_v=float(
            np.max(
                np.asarray(quadrotor_cfg["limits"]["max_velocity_3d"], dtype=np.float64)
            )
        ),
        max_a=float(
            np.max(
                np.asarray(
                    quadrotor_cfg["limits"]["max_acceleration_3d"], dtype=np.float64
                )
            )
        ),
    )

    planner_manager = PlannerManager(
        local_planner=local_planner,
        global_lmpc_config=global_lmpc_config,
        grid_map=grid_map,
        goal_tolerance=float(planning_cfg["goal"]["tolerance"]),
        planning_horizon=float(planning_cfg["local_planner"]["planning_horizon"]),
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
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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

    # ------------------------------------------------------------------
    # Mission goals
    # ------------------------------------------------------------------
    planner_manager.add_goal(np.array([3.0, 0.0, 0.0], dtype=np.float64))
    planner_manager.add_goal(np.array([6.0, 2.0, 0.0], dtype=np.float64))

    # ------------------------------------------------------------------
    # Main loop settings
    # ------------------------------------------------------------------
    dt_exec = 0.02
    dt_plan = 0.20

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


if __name__ == "__main__":
    main()
