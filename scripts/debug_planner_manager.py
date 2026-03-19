from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from trajplan.config import build_linear_mpc_config, load_project_configs
from trajplan.planning.planner_manager import PlannerManager
from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.quadrotor.state import QuadrotorState
from trajplan.shared_types import Vector
from trajplan.trajectory.bspline import UniformBSpline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FakeLocalPlanner:
    """
    A fake local planner for debugging PlannerManager.

    It records the last planning request and always returns a very simple
    valid UniformBSpline, so that PlannerManager can keep producing TRACK
    commands.
    """

    def __init__(self) -> None:
        self.call_count: int = 0
        self.last_call: dict | None = None

    def plan(
        self,
        active_command: QuadrotorCommand | None,
        local_start_pva: Vector,
        local_target_pva: Vector,
        current_time: float,
    ) -> UniformBSpline | None:
        self.call_count += 1

        self.last_call = {
            "call_count": self.call_count,
            "active_command_is_none": active_command is None,
            "active_command_mode": None
            if active_command is None
            else active_command.mode.value,
            "local_start_pva": np.asarray(local_start_pva, dtype=np.float64).copy(),
            "local_target_pva": np.asarray(local_target_pva, dtype=np.float64).copy(),
            "current_time": float(current_time),
        }

        start_pos = np.asarray(local_start_pva[:3], dtype=np.float64)
        target_pos = np.asarray(local_target_pva[:3], dtype=np.float64)

        # A minimal cubic spline with 4 control points.
        # This is only for debug, not for real planning quality.
        control_points = np.vstack(
            (
                start_pos,
                start_pos,
                target_pos,
                target_pos,
            )
        )

        return UniformBSpline(control_points=control_points, delta_t=1.0)


def make_state(
    position: list[float] | np.ndarray,
    velocity: list[float] | np.ndarray | None = None,
    acceleration: list[float] | np.ndarray | None = None,
) -> QuadrotorState:
    position = np.asarray(position, dtype=np.float64).reshape(-1)

    if position.shape != (3,):
        raise ValueError(f"Expected position shape (3,), but got {position.shape}.")

    if velocity is None:
        velocity = np.zeros(3, dtype=np.float64)
    else:
        velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)

    if acceleration is None:
        acceleration = np.zeros(3, dtype=np.float64)
    else:
        acceleration = np.asarray(acceleration, dtype=np.float64).reshape(-1)

    if velocity.shape != (3,):
        raise ValueError(f"Expected velocity shape (3,), but got {velocity.shape}.")
    if acceleration.shape != (3,):
        raise ValueError(
            f"Expected acceleration shape (3,), but got {acceleration.shape}."
        )

    pva = np.hstack((position, velocity, acceleration))
    return QuadrotorState(pva=pva)


def format_vec(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=4, suppress_small=True)


def print_manager_state(manager: PlannerManager, title: str) -> None:
    print(f"\n--- {title} ---")
    print(
        "active_goal_position:",
        None
        if manager.active_goal_position is None
        else format_vec(manager.active_goal_position),
    )
    print("num_pending_goals   :", len(manager.goal_position_list))
    print("global_progress     :", f"{manager.global_progress:.4f}")
    if manager.global_trajectory is None:
        print("global_traj_status  : None")
        print("global_traj_duration: None")
    else:
        print("global_traj_status  :", manager.global_trajectory.status)
        print("global_traj_duration:", f"{manager.global_trajectory.duration:.4f}")


def print_command(command: QuadrotorCommand, title: str) -> None:
    print(f"\n[{title}]")
    print("mode      :", command.mode.value)
    print("start_time:", f"{command.start_time:.4f}")
    print("message   :", command.message)
    print("has_traj  :", command.trajectory is not None)


def print_last_local_planner_call(fake_local_planner: FakeLocalPlanner) -> None:
    call = fake_local_planner.last_call

    if call is None:
        print("Local planner was not called.")
        return

    print("\nLocal planner last call")
    print("call_count             :", call["call_count"])
    print("active_command_is_none :", call["active_command_is_none"])
    print("active_command_mode    :", call["active_command_mode"])
    print("local_start_pva        :", format_vec(call["local_start_pva"]))
    print("local_target_pva       :", format_vec(call["local_target_pva"]))
    print("current_time           :", f"{call['current_time']:.4f}")


def main() -> None:
    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )
    global_lmpc_config = build_linear_mpc_config(
        quadrotor_cfg=quadrotor_cfg,
        planning_cfg=planning_cfg,
    )

    fake_local_planner = FakeLocalPlanner()
    manager = PlannerManager(
        local_planner=fake_local_planner,
        global_lmpc_config=global_lmpc_config,
        goal_tolerance=float(planning_cfg["goal"]["tolerance"]),
        local_target_distance=3.0,
    )

    goal_1 = np.array([3.0, 0.0, 0.0], dtype=np.float64)
    goal_2 = np.array([6.0, 2.0, 0.0], dtype=np.float64)

    manager.add_goal(goal_1)
    manager.add_goal(goal_2)

    print("=" * 80)
    print("PlannerManager debug start")

    # ------------------------------------------------------------------
    # Scenario 1
    # No active goal yet, goal list non-empty.
    # Expectation:
    # - activate first goal
    # - build global trajectory
    # - call local planner
    # - return TRACK command
    # ------------------------------------------------------------------
    state_1 = make_state(position=[0.0, 0.0, 0.0])
    time_1 = 0.0
    command_1 = manager.update_and_get_command(
        current_state=state_1,
        current_time=time_1,
        active_command=None,
    )

    print_manager_state(manager, "After Scenario 1")
    print_command(command_1, "Scenario 1 command")
    print_last_local_planner_call(fake_local_planner)

    # ------------------------------------------------------------------
    # Scenario 2
    # Same goal, not reached yet, and there is an old active TRACK command.
    # Expectation:
    # - keep same active goal
    # - do NOT switch goal
    # - still replan
    # - local planner receives old active_command, not None
    # ------------------------------------------------------------------
    state_2 = make_state(position=[0.8, 0.0, 0.0], velocity=[0.2, 0.0, 0.0])
    time_2 = 0.2
    command_2 = manager.update_and_get_command(
        current_state=state_2,
        current_time=time_2,
        active_command=command_1,
    )

    print_manager_state(manager, "After Scenario 2")
    print_command(command_2, "Scenario 2 command")
    print_last_local_planner_call(fake_local_planner)

    # ------------------------------------------------------------------
    # Scenario 3
    # Current goal reached, but there is still one pending goal.
    # Expectation:
    # - switch to the next goal
    # - reset and rebuild global trajectory
    # - local planner receives active_command=None because goal switched
    # ------------------------------------------------------------------
    state_3 = make_state(position=goal_1, velocity=[0.0, 0.0, 0.0])
    time_3 = 1.0
    command_3 = manager.update_and_get_command(
        current_state=state_3,
        current_time=time_3,
        active_command=command_2,
    )

    print_manager_state(manager, "After Scenario 3")
    print_command(command_3, "Scenario 3 command")
    print_last_local_planner_call(fake_local_planner)

    # ------------------------------------------------------------------
    # Scenario 4
    # Current goal reached and no pending goals remain.
    # Expectation:
    # - return FINISH command
    # ------------------------------------------------------------------
    manager.goal_position_list.clear()
    state_4 = make_state(position=goal_2, velocity=[0.0, 0.0, 0.0])
    time_4 = 2.0
    command_4 = manager.update_and_get_command(
        current_state=state_4,
        current_time=time_4,
        active_command=command_3,
    )

    print_manager_state(manager, "After Scenario 4")
    print_command(command_4, "Scenario 4 command")

    print("\n" + "=" * 80)
    print("Expected checks")
    print(
        "1. Scenario 1 should be TRACK and local planner active_command_is_none should be True."
    )
    print(
        "2. Scenario 2 should be TRACK and local planner active_command_is_none should be False."
    )
    print(
        "3. Scenario 3 should be TRACK and local planner active_command_is_none should be True."
    )
    print("4. Scenario 4 should be FINISH.")
    print("=" * 80)


if __name__ == "__main__":
    main()
