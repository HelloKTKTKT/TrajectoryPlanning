from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from trajplan.planning.local_planner import LocalPlanningStatus
from trajplan.shared_types import Vector
from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.quadrotor.state import QuadrotorState
from trajplan.map.grid_map import GridMap
from trajplan.trajectory.bspline import UniformBSpline
from trajplan.trajectory.polynomial import MinicostTraj
from trajplan.planning.local_planner import LocalPlanner

from enum import Enum


@dataclass(slots=True)
class PlannerManagerConfig:
    """
    Configuration for PlannerManager.
    """

    goal_tol: float
    planning_horizon: float
    global_interpoint_dist_thresh: float
    replan_thresh: float
    no_replan_thresh: float
    safety_check_interval: float
    safety_sample_dt: float
    emergency_time: float
    emergency_vel_threshold: float
    emergency_stop_duration: float


@dataclass(slots=True)
class PlannerManagerUpdateResult:
    """
    One planner-manager update result.

    Notes
    -----
    - `command` is always populated.
    - `should_publish=False` means the caller should keep the currently active
      command and not push `command` downstream.
    """

    command: QuadrotorCommand
    should_publish: bool


class SafetyStatus(str, Enum):
    SAFE = "safe"
    REPLAN = "replan"
    EMERGENCY = "emergency"


class PlannerManager:
    """
    High-level planner coordinator.

    Current responsibilities
    ------------------------
    - maintain active / pending global goals
    - maintain global guidance state
    - select local start pva and local target pva
    - decide whether a planning cycle is needed
    - delegate "generate new" vs "try extend old" to the local planner by
      choosing whether to pass the active command through
    """

    def __init__(
        self,
        local_planner: LocalPlanner,
        config: PlannerManagerConfig,
        grid_map: GridMap,
    ) -> None:
        self.local_planner = local_planner
        self.config = config
        self.grid_map = grid_map

        self.active_goal_position: Vector | None = None
        self.goal_position_list: list[Vector] = []

        self.global_trajectory: MinicostTraj | None = None
        self.global_progress: float = 0.0
        self.in_emergency_mode: bool = False
        self.last_safety_check_time: float = time.monotonic()
        self.next_traj_id: int = 1

    @staticmethod
    def _validate_goal_position(goal_position: Vector) -> Vector:
        goal = np.asarray(goal_position, dtype=np.float64).reshape(-1)

        if goal.shape != (3,):
            raise ValueError(
                f"Expected goal_position to have shape (3,), but got {goal.shape}."
            )

        return goal

    def add_goal(self, goal_position: Vector) -> None:
        goal = self._validate_goal_position(goal_position)
        self.goal_position_list.append(goal)

    def add_goals(self, goal_positions: Sequence[Vector]) -> None:
        for goal_position in goal_positions:
            self.add_goal(goal_position)

    def update_and_get_command(
        self,
        current_state: QuadrotorState,
        msg_time: float,
        active_command: QuadrotorCommand | None,
    ) -> PlannerManagerUpdateResult:
        """
        Main planner-manager update entry.

        This method intentionally follows a flattened EGO-style decision flow
        instead of exposing many explicit FSM states:

        1. handle emergency / abnormal conditions
        3. resolve active goal lifecycle
        4. ensure global guidance exists
        5. decide whether replanning is needed
        6. if needed, build local planning inputs and call the local planner
        """

        """
        Step 1: check in emergency mode.
        -> if cannot release yet, should_publish=False, return immediately and the result will not be sent.
        -> if can release, switch off emergency mode, turn the force_generate_new flag on to trigger a new plan generation in the future steps.
        """
        force_generate_new = False
        if self.in_emergency_mode:
            if self._can_release_emergency_mode(current_state):
                self.in_emergency_mode = False
                force_generate_new = True
            else:
                if active_command is None:
                    raise RuntimeError(
                        "Planner is in emergency mode but active_command is None."
                    )
                return PlannerManagerUpdateResult(
                    command=active_command.copy(),
                    should_publish=False,
                )  # Here the command does not matter, we keep should_publish=False, agent will not receive this command, only for structure completeness.

        """
        Step 2: safety check

        Check the safety of the current active command.

        - if safe, continue
        - if replanning is needed, set `safety_forced_replan = True`
        - if emergency stop is needed, build an emergency-stop command,
          turn `in_emergency_mode` on, and return immediately
        """
        safety_forced_replan = False
        if not force_generate_new:
            safety_check_time = time.monotonic()
            should_run_safety_check = self.config.safety_check_interval <= 0.0 or (
                safety_check_time - self.last_safety_check_time
                >= self.config.safety_check_interval
            )
            if should_run_safety_check:
                self.last_safety_check_time = safety_check_time
                safety_status = self._check_safety(
                    msg_time=msg_time,
                    active_command=active_command,
                )
                if safety_status == SafetyStatus.EMERGENCY:
                    self.in_emergency_mode = True
                    emergency_command = self._build_emergency_stop_command(
                        current_state=current_state,
                        commit_time=time.monotonic(),
                    )
                    return PlannerManagerUpdateResult(
                        command=emergency_command,
                        should_publish=True,
                    )
                if safety_status == SafetyStatus.REPLAN:
                    safety_forced_replan = True

        """
        Step 3: goal life cycle

        Resolve the current active goal and the queued goal list.

        - if the current active goal is reached:
          clear it and reset global guidance
        - if there is no active goal:
          fetch the next goal from the queue
        - if no goal is available anymore:
          finish the mission and return immediately
        """
        goal_switched = False
        if self.active_goal_position is not None:
            if self.is_goal_reached(current_state):
                self.active_goal_position = None
                self._reset_global_guidance()

        if self.active_goal_position is None:
            if len(self.goal_position_list) == 0:
                return PlannerManagerUpdateResult(
                    command=QuadrotorCommand.finish(
                        start_time=time.monotonic(),
                        message="All goals are completed.",
                    ),
                    should_publish=True,
                )

            self.active_goal_position = self.goal_position_list.pop(0)
            self._reset_global_guidance()
            goal_switched = True

        """
        Step 4: ensure global guidance exists

        The planner manager maintains a global trajectory for the current active
        goal. If it does not exist yet, build a new one from the current state
        to the current active goal.
        """
        if self.global_trajectory is None:
            self.global_trajectory = self._build_global_trajectory(
                current_state=current_state,
            )
            self.global_progress = 0.0

        """
        Notice: after Step 4 (if code runs to here), we are guaranteed to have a valid global trajectory and the active goal is not None.
        """

        """
        Step 5: decide whether replanning is needed

        There are three strong triggers that bypass the normal ego-v1-style
        replan test:
        - goal_switched
        - force_generate_new
        - safety_forced_replan

        If none of them is active, fall back to the normal EXEC_TRAJ logic:
        - if the current track has finished, planning is needed
        - if the quadrotor is already near the final goal, keep the old command
        - if the quadrotor is still near the start of the current local track,
          keep the old command
        - otherwise, trigger a normal local replan
        """
        if (
            goal_switched
            or force_generate_new
            or active_command is None
            or not active_command.is_track
        ):
            local_start_pva = current_state.pva.copy()
            local_target_pva = self._get_local_target_update_global_progress(
                local_start_pva
            )
            command_for_local_planner = None

        elif safety_forced_replan:
            local_start_pva = active_command.sample_pva(msg_time)
            local_target_pva = self._get_local_target_update_global_progress(
                local_start_pva
            )
            command_for_local_planner = active_command
        else:
            trajectory = active_command.trajectory
            if not isinstance(trajectory, UniformBSpline):
                raise RuntimeError(
                    "Expected TRACK active_command to carry a UniformBSpline trajectory."
                )
            t_cur = min(
                active_command.get_elapsed_time(msg_time),
                float(trajectory.duration),
            )
            traj_duration = float(trajectory.duration)
            current_pos = trajectory.evaluate_pva(t_cur)[:3]
            start_pos = trajectory.evaluate_pva(0.0)[:3]

            traj_finished = t_cur >= traj_duration - 1e-2
            near_goal = (
                np.linalg.norm(self.active_goal_position - current_pos)
                < self.config.no_replan_thresh
            )
            near_start = (
                np.linalg.norm(current_pos - start_pos) < self.config.replan_thresh
            )

            if traj_finished:
                local_start_pva = current_state.pva.copy()
                local_target_pva = self._get_local_target_update_global_progress(
                    local_start_pva
                )
                command_for_local_planner = None

            elif near_goal or near_start:
                return PlannerManagerUpdateResult(
                    command=active_command.copy(),
                    should_publish=False,
                )
            else:
                local_start_pva = active_command.sample_pva(msg_time)
                local_target_pva = self._get_local_target_update_global_progress(
                    local_start_pva
                )
                command_for_local_planner = active_command

        local_plan_res = self.local_planner.plan(
            grid_map=self.grid_map,
            active_command=command_for_local_planner,
            local_start_pva=local_start_pva,
            local_target_pva=local_target_pva,
            msg_time=msg_time,
        )

        if local_plan_res.status == LocalPlanningStatus.SUCCESS:
            trajectory = local_plan_res.trajectory
            if not isinstance(trajectory, UniformBSpline):
                raise RuntimeError(
                    "Local planner returned SUCCESS with a non-UniformBSpline trajectory."
                )
            commit_time = time.monotonic()
            new_command = QuadrotorCommand.track(
                trajectory=trajectory,
                start_time=commit_time,
                traj_id=self.next_traj_id,
                message=local_plan_res.message,
                local_target_pva=local_target_pva.copy(),
            )
            self.next_traj_id += 1

            return PlannerManagerUpdateResult(
                command=new_command,
                should_publish=True,
            )

        if active_command is not None:
            return PlannerManagerUpdateResult(
                command=active_command.copy(),
                should_publish=False,
            )

        return PlannerManagerUpdateResult(
            command=QuadrotorCommand.hover(
                start_time=time.monotonic(),
                message=local_plan_res.message,
            ),
            should_publish=False,
        )

    def _can_release_emergency_mode(self, current_state: QuadrotorState) -> bool:
        current_vel_norm = np.linalg.norm(current_state.velocity)
        return bool(current_vel_norm < self.config.emergency_vel_threshold)

    def _build_emergency_stop_command(
        self,
        current_state: QuadrotorState,
        commit_time: float,
    ) -> QuadrotorCommand:
        zero_vec = np.zeros(3, dtype=np.float64)
        stop_position = current_state.position.copy()
        stop_target_pva = np.hstack((stop_position, zero_vec, zero_vec))

        stop_duration = float(self.config.emergency_stop_duration)
        half_duration = stop_duration / 2.0

        stop_trajectory = MinicostTraj(
            waypoints=np.vstack(
                (
                    stop_position,
                    stop_position,
                    stop_position,
                )
            ),
            ho_start=np.vstack(
                (
                    current_state.velocity,
                    current_state.acceleration,
                )
            ),
            ho_end=np.vstack(
                (
                    zero_vec,
                    zero_vec,
                )
            ),
            s=3,
            time_idx=np.array(
                [0.0, half_duration, stop_duration],
                dtype=np.float64,
            ),
        )

        return QuadrotorCommand.emergency_stop(
            trajectory=stop_trajectory,
            start_time=commit_time,
            local_target_pva=stop_target_pva,
            message="Safety check triggered emergency stop.",
        )

    def _check_safety(
        self,
        msg_time: float,
        active_command: QuadrotorCommand | None,
    ) -> SafetyStatus:
        """
        Determine whether the currently active command is still safe.

        Returns
        -------
        SafetyStatus.SAFE
            No immediate issue is detected.
        SafetyStatus.REPLAN
            The current trajectory should be replaced by a new local plan.
        SafetyStatus.EMERGENCY
            The situation is too risky and emergency mode should be entered.

        Notes
        -----
        This follows the safety-check style of ego-planner-master:
        - only the current TRACK command is checked
        - the active local trajectory is sampled forward in time
        - if current progress is still within the first 2/3 of the trajectory,
          only the first 2/3 partition is considered valid for safety checking
        - collision within `emergency_time` triggers EMERGENCY, otherwise REPLAN
        """
        if active_command is None:
            return SafetyStatus.SAFE

        if not active_command.is_track:
            return SafetyStatus.SAFE

        trajectory = active_command.trajectory
        if not isinstance(trajectory, UniformBSpline):
            raise RuntimeError(
                "Expected TRACK active_command to carry a UniformBSpline trajectory."
            )
        t_cur = active_command.get_elapsed_time(msg_time)
        traj_duration = float(trajectory.duration)
        if traj_duration <= 0.0 or t_cur >= traj_duration:
            return SafetyStatus.SAFE

        check_end_time = traj_duration
        two_thirds_time = (2.0 * traj_duration) / 3.0
        if t_cur < two_thirds_time:
            check_end_time = two_thirds_time

        sample_dt = float(self.config.safety_sample_dt)
        if sample_dt <= 0.0:
            raise ValueError(f"Expected safety_sample_dt > 0, but got {sample_dt}.")

        sample_t = t_cur
        while sample_t <= check_end_time + 1e-9:
            sample_pva = trajectory.evaluate_pva(sample_t)
            sample_pos = sample_pva[:3]
            if self.grid_map.is_occupied_world(sample_pos):
                time_to_collision = max(0.0, sample_t - t_cur)
                if time_to_collision < self.config.emergency_time:
                    return SafetyStatus.EMERGENCY
                return SafetyStatus.REPLAN
            sample_t += sample_dt

        return SafetyStatus.SAFE

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

        error_norm = np.linalg.norm(current_pv - goal_pv)
        return bool(error_norm <= self.config.goal_tol)

    def _build_global_trajectory(
        self,
        current_state: QuadrotorState,
    ) -> MinicostTraj:
        if self.active_goal_position is None:
            raise RuntimeError(
                "_build_global_trajectory() requires an active global goal."
            )

        start_pos = np.asarray(current_state.position, dtype=np.float64).reshape(3)
        start_vel = np.asarray(current_state.velocity, dtype=np.float64).reshape(3)
        start_acc = np.asarray(current_state.acceleration, dtype=np.float64).reshape(3)
        goal_pos = np.asarray(self.active_goal_position, dtype=np.float64).reshape(3)

        max_vel = float(self.local_planner.config.max_vel)
        points = [start_pos, goal_pos]

        inter_points: list[np.ndarray] = []
        dist_thresh = float(self.config.global_interpoint_dist_thresh)
        for point_idx in range(len(points) - 1):
            p0 = points[point_idx]
            p1 = points[point_idx + 1]
            inter_points.append(p0)

            dist = float(np.linalg.norm(p1 - p0))
            if dist > dist_thresh:
                insert_num = int(np.floor(dist / dist_thresh)) + 1
                for insert_idx in range(1, insert_num):
                    alpha = float(insert_idx) / float(insert_num)
                    inter_pt = (1.0 - alpha) * p0 + alpha * p1
                    inter_points.append(inter_pt)

        inter_points.append(points[-1])

        segment_times: list[float] = []
        min_segment_time = 1e-3
        for point_idx in range(len(inter_points) - 1):
            segment_dist = float(
                np.linalg.norm(inter_points[point_idx + 1] - inter_points[point_idx])
            )
            segment_time = max(segment_dist / max_vel, min_segment_time)
            segment_times.append(segment_time)

        segment_times[0] *= 2.0
        segment_times[-1] *= 2.0

        time_idx = [0.0]
        for segment_time in segment_times:
            time_idx.append(time_idx[-1] + segment_time)

        return MinicostTraj(
            waypoints=np.vstack(inter_points),
            ho_start=np.vstack((start_vel, start_acc)),
            ho_end=np.zeros((2, 3), dtype=np.float64),
            s=3,
            time_idx=np.asarray(time_idx, dtype=np.float64),
        )

    def _get_local_target_update_global_progress(
        self,
        local_start_pva: Vector,
    ) -> Vector:
        if self.global_trajectory is None:
            raise RuntimeError(
                "_get_local_target_update_global_progress() requires a valid global trajectory."
            )
        if self.active_goal_position is None:
            raise RuntimeError(
                "_get_local_target_update_global_progress() requires an active global goal."
            )

        start_position = np.asarray(local_start_pva[:3], dtype=np.float64).reshape(3)
        t_min = float(self.global_progress)
        t_max = float(self.global_trajectory.duration)

        max_vel = float(self.local_planner.config.max_vel)
        max_acc = float(self.local_planner.config.max_acc)
        dt = float(self.config.planning_horizon) / 20.0 / max_vel

        dist_min = np.inf
        dist_min_t = t_min
        t = t_min
        local_target_pva = self.global_trajectory.evaluate_pva(t_max)

        while t <= t_max + 1e-9:
            pva = self.global_trajectory.evaluate_pva(t)
            position = pva[:3]
            distance = np.linalg.norm(position - start_position)

            if distance < dist_min:
                dist_min = distance
                dist_min_t = t

            if distance >= self.config.planning_horizon:
                local_target_pva = pva
                self.global_progress = dist_min_t
                break

            local_target_pva = pva
            t += dt
        else:
            self.global_progress = dist_min_t

        local_target_position = local_target_pva[:3]
        stopping_distance = (max_vel * max_vel) / (2.0 * max_acc)

        if (
            np.linalg.norm(self.active_goal_position - local_target_position)
            < stopping_distance
        ):
            local_target_velocity = np.zeros(3, dtype=np.float64)
        else:
            local_target_velocity = np.asarray(
                local_target_pva[3:6], dtype=np.float64
            ).reshape(3)

        local_target_acceleration = np.zeros(3, dtype=np.float64)
        return np.hstack(
            (
                local_target_position,
                local_target_velocity,
                local_target_acceleration,
            )
        )

    def _reset_global_guidance(self) -> None:
        self.global_trajectory = None
        self.global_progress = 0.0
