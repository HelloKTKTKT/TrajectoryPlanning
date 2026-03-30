from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from trajplan.map.grid_map import GridMap
from trajplan.planning.local_planner import LocalPlanner
from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.quadrotor.state import QuadrotorState
from trajplan.shared_types import Vector
from trajplan.trajectory.linear_mpc import LinearMpcConfig, LinearMpcTrajectory


@dataclass(slots=True)
class PlannerManagerConfig:
    """
    Configuration for PlannerManager.
    """

    goal_tol: float
    planning_horizon: float


class PlannerManager:
    """
    Planner manager.

    Current responsibilities
    ------------------------
    - maintain an active global goal and a queued goal list
    - maintain a global LMPC trajectory and its progress
    - decide whether to finish, hover, keep the old command, or replan
    - decide local start pva and local target pva
    - select local target pva based on planning_horizon threshold
    - call the local planner to generate a new local trajectory
    """

    def __init__(
        self,
        local_planner: LocalPlanner,
        global_lmpc_config: LinearMpcConfig,
        config: PlannerManagerConfig,
        grid_map: GridMap,
    ) -> None:
        self.local_planner = local_planner
        self.global_lmpc_config = global_lmpc_config
        self.grid_map = grid_map
        self.config = config

        # self.goal_tol = float(config.goal_tol)
        # self.planning_horizon = float(config.planning_horizon)

        self.active_goal_position: Vector | None = None
        self.goal_position_list: list[Vector] = []

        self.global_trajectory: LinearMpcTrajectory | None = None
        self.global_progress: float = 0.0

    # ------------------------------------------------------------------
    # goal management
    # ------------------------------------------------------------------

    def _validate_goal_position(self, goal_position: Vector) -> Vector:
        goal = np.asarray(goal_position, dtype=np.float64).reshape(-1)

        if goal.shape != (3,):
            raise ValueError(
                f"Expected goal_position to have shape (3,), but got {goal.shape}."
            )

        return goal

    def add_goal(self, goal_position: Vector) -> None:
        goal = self._validate_goal_position(goal_position)
        self.goal_position_list.append(goal)

    def add_goals(self, goal_positions: list[Vector]) -> None:
        for goal_position in goal_positions:
            self.add_goal(goal_position)

    def clear_goals(self) -> None:
        self.active_goal_position = None
        self.goal_position_list.clear()
        self._reset_global_guidance()

    def has_active_goal(self) -> bool:
        return self.active_goal_position is not None

    def has_pending_goals(self) -> bool:
        return len(self.goal_position_list) > 0

    def is_goal_reached(self, current_state: QuadrotorState) -> bool:
        """
        Check goal reaching using the first 6 dimensions:
        position + velocity.

        The goal is interpreted as:
        - target position = active_goal_position
        - target velocity = zero
        """
        if self.active_goal_position is None:
            return False

        goal_pv = np.hstack((self.active_goal_position, np.zeros(3, dtype=np.float64)))
        current_pv = current_state.pva[:6]

        error_norm = np.linalg.norm(current_pv[:3] - goal_pv[:3])
        # print(
        #     f"error_norm: {error_norm}, curent p: {current_pv[:3]}, goal p: {goal_pv[:3]}"
        # )
        return bool(error_norm <= self.config.goal_tol)

    # ------------------------------------------------------------------
    # global guidance
    # ------------------------------------------------------------------

    def _reset_global_guidance(self) -> None:
        self.global_trajectory = None
        self.global_progress = 0.0

    def _build_global_trajectory(
        self,
        current_state: QuadrotorState,
    ) -> LinearMpcTrajectory | None:
        """
        Build a global LMPC trajectory from current state to active goal.
        """
        if self.active_goal_position is None:
            return None

        start_state = current_state.pva.copy()
        target_state = np.hstack(
            (self.active_goal_position, np.zeros(6, dtype=np.float64))
        )

        global_trajectory = LinearMpcTrajectory(config=self.global_lmpc_config)
        global_trajectory.update_traj_info(
            start_state=start_state,
            target_state=target_state,
        )

        if global_trajectory.status != "optimal":
            return None

        return global_trajectory

    # ------------------------------------------------------------------
    # local start / local target selection
    # ------------------------------------------------------------------

    def _select_local_start_pva(
        self,
        current_state: QuadrotorState,
        current_time: float,
        active_command: QuadrotorCommand | None,
    ) -> Vector:
        """
        Select the local trajectory start pva.

        Current default rule
        --------------------
        - if there is a valid active track command, use its evaluated pva
        - otherwise use current odometry state directly
        """
        if (
            active_command is not None
            and active_command.is_track
            and active_command.trajectory
            is not None  # if not None, this should be a valid uniform bspline
        ):
            return active_command.sample_pva(current_time)

        pva = np.zeros(9)
        pva[:3] = current_state.pva.copy()[:3]
        return pva

    def _get_local_target_update_global_progress(
        self,
        local_start_pva: Vector,
    ) -> Vector:
        """
        Select the local target pva from the maintained global trajectory.

        Logic
        -----
        1. Starting from current global_progress, find the point on the global
           trajectory whose position is closest to local_start_pva[:3].
        2. Update global_progress to that closest time.
        3. Continue forward until the distance from the local start position
           exceeds planning_horizon.
        4. Return the corresponding pva as local target.
        """
        if self.global_trajectory is None:
            raise ValueError("global_trajectory is not available.")

        start_position = local_start_pva[:3]

        t_min = self.global_progress
        t_max = self.global_trajectory.duration
        dt = self.global_lmpc_config.ts

        # Step 1: find the closest point from current global_progress onward.
        dist_min = np.inf
        dist_min_t = t_min

        t = t_min
        while t <= t_max + 1e-9:
            pva = self.global_trajectory.evaluate_pva(t)
            position = pva[:3]
            distance = np.linalg.norm(position - start_position)

            if distance < dist_min:
                dist_min = distance
                dist_min_t = t

            t += dt

        self.global_progress = dist_min_t
        self.global_trajectory.current_progress = dist_min_t

        # Step 2: search forward for the local target.
        t = dist_min_t
        local_target_pva = self.global_trajectory.evaluate_pva(t_max)

        while t <= t_max + 1e-9:
            pva = self.global_trajectory.evaluate_pva(t)
            position = pva[:3]
            distance = np.linalg.norm(position - start_position)

            if distance >= self.config.planning_horizon:
                local_target_pva = pva
                break

            local_target_pva = pva
            t += dt

        return local_target_pva

    # ------------------------------------------------------------------
    # main update interface
    # ------------------------------------------------------------------

    def update_and_get_command(
        self,
        current_state: QuadrotorState | None,
        current_time: float,
        active_command: QuadrotorCommand | None,
    ) -> QuadrotorCommand:
        """
        Main update interface.

        This is the main function to call at each control step.

        It will internally decide which command should be active at the current moment.
        """
        if current_state is None:
            return QuadrotorCommand.hover(
                start_time=current_time,
                message="No state available.",
            )

        goal_switched = False

        # --------------------------------------------------------------
        # Step 1: resolve active goal from current active goal + goal list
        # --------------------------------------------------------------
        if self.active_goal_position is not None:
            if self.is_goal_reached(current_state):
                self.active_goal_position = None
                self._reset_global_guidance()

        if self.active_goal_position is None:
            if len(self.goal_position_list) == 0:
                return QuadrotorCommand.finish(
                    start_time=current_time,
                    message="All goals are completed.",
                )
            self.active_goal_position = self.goal_position_list.pop(0)
            self._reset_global_guidance()  # global traj = None, global progress = 0
            goal_switched = True

        # --------------------------------------------------------------
        # Step 2: ensure global trajectory exists for current active goal
        # --------------------------------------------------------------
        if self.global_trajectory is None:
            self.global_trajectory = self._build_global_trajectory(
                current_state=current_state,
            )
            self.global_progress = 0.0

        if self.global_trajectory is None:
            return QuadrotorCommand.hover(
                start_time=current_time,
                message="Failed to build global trajectory.",
            )

        # --------------------------------------------------------------
        # Step 3: decide whether to keep old command or replan: currently always replan when not finished
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # Step 4: build local planning request
        # --------------------------------------------------------------

        local_start_pva = self._select_local_start_pva(
            current_state=current_state,
            current_time=current_time,
            active_command=active_command,
        )

        local_target_pva = self._get_local_target_update_global_progress(
            local_start_pva=local_start_pva,
        )

        command_for_local_planner = None if goal_switched else active_command

        planning_result = self.local_planner.plan(
            grid_map=self.grid_map,
            active_command=command_for_local_planner,
            local_start_pva=local_start_pva,
            local_target_pva=local_target_pva,
            current_time=current_time,
        )

        if planning_result.status == "failure":
            return QuadrotorCommand.hover(
                start_time=current_time,
                message=planning_result.message,
            )

        if planning_result.status == "target_close":
            if active_command is not None and active_command.is_track:
                return active_command
            return QuadrotorCommand.hover(
                start_time=current_time,
                message=planning_result.message,
            )

        if planning_result.status != "success" or planning_result.trajectory is None:
            return QuadrotorCommand.hover(
                start_time=current_time,
                message="Unexpected local planning result.",
            )

        return QuadrotorCommand.track(
            trajectory=planning_result.trajectory,
            start_time=current_time,
            message=planning_result.message,
        )
