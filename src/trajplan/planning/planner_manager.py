from __future__ import annotations

import time
from collections import deque
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
from trajplan.planning.messages import (
    NeighborTrajectoryMessage,
    TargetStateMessage,
    TrackingReference,
)

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
    swarm_clearance: float
    swarm_replan_cooldown: float
    swarm_priority_enabled: bool
    safety_check_interval: float
    safety_sample_dt: float
    emergency_time: float
    emergency_vel_threshold: float
    emergency_stop_duration: float
    tracking_enabled: bool
    tracking_history_size: int
    tracking_update_interval: float
    tracking_velocity_deadband: float
    tracking_prediction_time: float
    tracking_goal_update_thresh: float
    tracking_replan_interval: float
    tracking_square_size: float
    tracking_height: float
    tracking_goal_obstacle_margin: float
    tracking_goal_search_step: float
    tracking_goal_search_max_dist: float
    tracking_goal_map_margin: float
    tracking_square_scale_min: float
    tracking_square_scale_max: float
    tracking_square_scale_step: float


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


@dataclass(slots=True)
class SafetyCheckResult:
    status: SafetyStatus
    triggered_by_swarm: bool = False


@dataclass(slots=True)
class NeighborTrajectoryRecord:
    agent_id: int
    traj_id: int
    start_time: float
    trajectory: UniformBSpline
    local_target_pva: Vector | None = None

    @classmethod
    def from_message(
        cls,
        message: NeighborTrajectoryMessage,
    ) -> NeighborTrajectoryRecord:
        local_target_pva = (
            None
            if message.local_target_pva is None
            else np.asarray(message.local_target_pva, dtype=np.float64).copy()
        )
        return cls(
            agent_id=message.agent_id,
            traj_id=message.traj_id,
            start_time=message.start_time,
            trajectory=message.trajectory.copy(),
            local_target_pva=local_target_pva,
        )


@dataclass(slots=True)
class FilteredTargetState:
    timestamp: float
    position: Vector
    velocity: Vector


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
        agent_id: int,
        local_planner: LocalPlanner,
        config: PlannerManagerConfig,
        grid_map: GridMap,
    ) -> None:
        self.agent_id = int(agent_id)
        self.local_planner = local_planner
        self.config = config
        self.grid_map = grid_map

        self.active_goal_position: Vector | None = None
        self.goal_position_list: list[Vector] = []

        self.global_trajectory: MinicostTraj | None = None
        self.global_progress: float = 0.0
        self.in_emergency_mode: bool = False
        self.last_safety_check_time: float = time.monotonic()
        self.last_swarm_replan_time: float = -np.inf
        self.last_tracking_replan_time: float = -np.inf
        self.next_traj_id: int = 1
        self.neighbor_traj_buffer: dict[int, NeighborTrajectoryRecord] = {}
        self.target_state_history: deque[TargetStateMessage] = deque()
        self.filtered_target_state: FilteredTargetState | None = None
        self._last_target_estimate_time: float = -np.inf
        self._last_target_avg_position: Vector | None = None
        self.last_valid_tracking_goal: Vector | None = None
        self.latest_tracking_reference: TrackingReference | None = None

    def update_neighbor_trajectory(
        self,
        message: NeighborTrajectoryMessage,
    ) -> bool:
        if message.agent_id == self.agent_id:
            return False

        current_record = self.neighbor_traj_buffer.get(message.agent_id)
        if current_record is not None:
            if message.traj_id < current_record.traj_id:
                return False
            if (
                message.traj_id == current_record.traj_id
                and message.start_time <= current_record.start_time
            ):
                return False

        self.neighbor_traj_buffer[message.agent_id] = (
            NeighborTrajectoryRecord.from_message(message)
        )
        return True

    def _ingest_target_state(
        self,
        target_state: TargetStateMessage | None,
    ) -> None:
        if target_state is None:
            return

        self.target_state_history.append(target_state.copy())
        while len(self.target_state_history) > self.config.tracking_history_size:
            self.target_state_history.popleft()

        if len(self.target_state_history) == 0:
            return

        latest_timestamp = float(self.target_state_history[-1].timestamp)
        if self.filtered_target_state is not None:
            elapsed = latest_timestamp - self._last_target_estimate_time
            if elapsed < self.config.tracking_update_interval:
                return

        avg_position = np.mean(
            np.vstack([message.position for message in self.target_state_history]),
            axis=0,
        )
        avg_velocity = np.mean(
            np.vstack([message.velocity for message in self.target_state_history]),
            axis=0,
        )

        if np.linalg.norm(avg_velocity) < self.config.tracking_velocity_deadband:
            avg_velocity = np.zeros(3, dtype=np.float64)

        self.filtered_target_state = FilteredTargetState(
            timestamp=latest_timestamp,
            position=np.asarray(avg_position, dtype=np.float64).reshape(3),
            velocity=np.asarray(avg_velocity, dtype=np.float64).reshape(3),
        )
        self._last_target_estimate_time = latest_timestamp
        self._last_target_avg_position = np.asarray(
            avg_position,
            dtype=np.float64,
        ).reshape(3)

    def _clamp_point_to_map_interior(
        self,
        point: Vector,
    ) -> Vector:
        point = np.asarray(point, dtype=np.float64).reshape(3)
        margin = max(
            float(self.config.tracking_goal_map_margin),
            0.5 * float(self.grid_map.resolution),
        )
        origin = np.asarray(self.grid_map.origin, dtype=np.float64).reshape(3)
        max_corner = origin + self.grid_map.map_size.astype(np.float64) * float(
            self.grid_map.resolution
        )
        lower = origin + margin
        upper = max_corner - margin
        return np.minimum(np.maximum(point, lower), upper)

    def _is_tracking_goal_valid(
        self,
        point: Vector,
    ) -> bool:
        point = np.asarray(point, dtype=np.float64).reshape(3)
        clamped = self._clamp_point_to_map_interior(point)
        if not np.allclose(point, clamped, atol=1e-9):
            return False
        obstacle_margin = max(
            float(self.config.tracking_goal_obstacle_margin),
            0.0,
        )
        sample_step = float(self.grid_map.resolution)
        max_index = int(np.ceil(obstacle_margin / sample_step))

        for ix in range(-max_index, max_index + 1):
            for iy in range(-max_index, max_index + 1):
                for iz in range(-max_index, max_index + 1):
                    offset = sample_step * np.array([ix, iy, iz], dtype=np.float64)
                    if offset @ offset > obstacle_margin * obstacle_margin + 1e-9:
                        continue
                    if self.grid_map.is_occupied_world(point + offset):
                        return False
        return True

    def _get_tracking_formation_offset(self, scale: float = 1.0) -> Vector:
        offsets = self._get_all_tracking_formation_offsets(scale=scale)
        return offsets[self.agent_id % len(offsets)].copy()

    def _get_all_tracking_formation_offsets(
        self,
        scale: float = 1.0,
    ) -> tuple[np.ndarray, ...]:
        half_side = 0.5 * float(self.config.tracking_square_size) * float(scale)
        height = float(self.config.tracking_height)
        return (
            np.array([half_side, half_side, height], dtype=np.float64),
            np.array([half_side, -half_side, height], dtype=np.float64),
            np.array([-half_side, -half_side, height], dtype=np.float64),
            np.array([-half_side, half_side, height], dtype=np.float64),
        )

    def _build_tracking_scale_candidates(self) -> list[float]:
        scale_min = float(self.config.tracking_square_scale_min)
        scale_max = float(self.config.tracking_square_scale_max)
        scale_step = max(float(self.config.tracking_square_scale_step), 1e-6)
        base_scale = 1.0

        scales: list[float] = [base_scale]

        scale = base_scale + scale_step
        while scale <= scale_max + 1e-9:
            scales.append(min(scale, scale_max))
            scale += scale_step

        scale = base_scale - scale_step
        while scale >= scale_min - 1e-9:
            scales.append(max(scale, scale_min))
            scale -= scale_step

        deduped: list[float] = []
        for scale_value in scales:
            if any(abs(scale_value - existing) <= 1e-9 for existing in deduped):
                continue
            deduped.append(scale_value)
        return deduped

    def _clamp_tracking_center_to_map_interior(
        self,
        target_center: Vector,
    ) -> Vector:
        target_center = np.asarray(target_center, dtype=np.float64).reshape(3)
        margin = max(
            float(self.config.tracking_goal_map_margin),
            0.5 * float(self.grid_map.resolution),
        )
        offsets = np.vstack(self._get_all_tracking_formation_offsets())
        offset_min = offsets.min(axis=0)
        offset_max = offsets.max(axis=0)

        origin = np.asarray(self.grid_map.origin, dtype=np.float64).reshape(3)
        max_corner = origin + self.grid_map.map_size.astype(np.float64) * float(
            self.grid_map.resolution
        )
        lower = origin + margin - offset_min
        upper = max_corner - margin - offset_max
        return np.minimum(np.maximum(target_center, lower), upper)

    def _build_tracking_reference(self) -> TrackingReference | None:
        if self.filtered_target_state is None:
            return None
        return TrackingReference(
            target_timestamp=self.filtered_target_state.timestamp,
            target_position=self.filtered_target_state.position.copy(),
            target_velocity=self.filtered_target_state.velocity.copy(),
            formation_offset=self._get_tracking_formation_offset(),
            prediction_lead_time=self.config.tracking_prediction_time,
        )

    def _compute_tracking_goal_from_reference(
        self,
        reference: TrackingReference,
    ) -> Vector:
        predicted_target_position = np.asarray(
            reference.target_position
            + reference.target_velocity * reference.prediction_lead_time,
            dtype=np.float64,
        ).reshape(3)
        predicted_target_position = self._clamp_tracking_center_to_map_interior(
            predicted_target_position
        )
        return np.asarray(
            predicted_target_position + reference.formation_offset,
            dtype=np.float64,
        ).reshape(3)

    def _sanitize_tracking_goal(
        self,
        raw_goal: Vector,
        reference: TrackingReference,
    ) -> Vector | None:
        raw_goal = np.asarray(raw_goal, dtype=np.float64).reshape(3)
        raw_goal = self._clamp_point_to_map_interior(raw_goal)
        if self._is_tracking_goal_valid(raw_goal):
            return raw_goal

        predicted_target_position = np.asarray(
            reference.target_position
            + reference.target_velocity * reference.prediction_lead_time,
            dtype=np.float64,
        ).reshape(3)
        predicted_target_position = self._clamp_tracking_center_to_map_interior(
            predicted_target_position
        )

        for scale in self._build_tracking_scale_candidates():
            candidate = predicted_target_position + self._get_tracking_formation_offset(
                scale=scale
            )
            candidate = self._clamp_point_to_map_interior(candidate)
            if self._is_tracking_goal_valid(candidate):
                return candidate

        return None

    def _is_tracking_segment_clear(
        self,
        start_point: Vector,
        end_point: Vector,
    ) -> bool:
        start_point = np.asarray(start_point, dtype=np.float64).reshape(3)
        end_point = np.asarray(end_point, dtype=np.float64).reshape(3)
        delta = end_point - start_point
        length = float(np.linalg.norm(delta))
        if length <= 1e-9:
            return self._is_tracking_goal_valid(end_point)

        step = max(float(self.grid_map.resolution), 0.05)
        num_segments = max(int(np.ceil(length / step)), 1)
        for alpha in np.linspace(0.0, 1.0, num_segments + 1, dtype=np.float64):
            sample = start_point + alpha * delta
            if not self._is_tracking_goal_valid(sample):
                return False
        return True

    def _compute_tracking_planning_goal(
        self,
        reference_goal: Vector,
        reference: TrackingReference,
        current_state: QuadrotorState,
    ) -> Vector:
        reference_goal = np.asarray(reference_goal, dtype=np.float64).reshape(3)
        current_position = np.asarray(
            current_state.position, dtype=np.float64
        ).reshape(3)
        target_velocity = np.asarray(reference.target_velocity, dtype=np.float64).reshape(3)
        target_speed = float(np.linalg.norm(target_velocity))

        if target_speed < self.config.tracking_velocity_deadband:
            return reference_goal.copy()

        min_planning_target_dist = max(0.6 * float(self.config.planning_horizon), 1.0)
        dist_to_reference = float(np.linalg.norm(reference_goal - current_position))
        segment_clear = self._is_tracking_segment_clear(current_position, reference_goal)

        if segment_clear and dist_to_reference >= min_planning_target_dist:
            return reference_goal.copy()

        lead_dir = target_velocity / target_speed
        lead_dist = max(min_planning_target_dist - dist_to_reference, 0.0)
        if not segment_clear:
            lead_dist = max(lead_dist, 0.5 * float(self.config.planning_horizon))

        lead_candidates = (
            lead_dist,
            max(lead_dist, 0.5 * float(self.config.planning_horizon)),
            max(lead_dist, 1.0 * float(self.config.planning_horizon)),
        )
        for candidate_lead in lead_candidates:
            if candidate_lead <= 1e-9:
                continue
            planning_goal = reference_goal + candidate_lead * lead_dir
            planning_goal = self._clamp_point_to_map_interior(planning_goal)
            if self._is_tracking_goal_valid(planning_goal):
                return planning_goal

        return reference_goal.copy()

    def _update_tracking_goal(
        self,
        current_state: QuadrotorState,
        msg_time: float,
        target_update_received: bool,
    ) -> bool:
        self.latest_tracking_reference = self._build_tracking_reference()
        if self.latest_tracking_reference is None:
            return False

        raw_goal = self._compute_tracking_goal_from_reference(
            self.latest_tracking_reference
        )
        sanitized_goal = self._sanitize_tracking_goal(
            raw_goal,
            self.latest_tracking_reference,
        )
        if sanitized_goal is None:
            self.active_goal_position = None
            self._reset_global_guidance()
            return False

        sanitized_goal = np.asarray(sanitized_goal, dtype=np.float64).reshape(3)
        self.last_valid_tracking_goal = sanitized_goal.copy()

        if self.active_goal_position is None:
            self.active_goal_position = sanitized_goal.copy()
            self._reset_global_guidance()
            return True

        goal_shift = np.linalg.norm(sanitized_goal - self.active_goal_position)
        interval_due = (
            msg_time - self.last_tracking_replan_time
            >= self.config.tracking_replan_interval
        )
        should_refresh_goal = (
            goal_shift >= self.config.tracking_goal_update_thresh
            or (target_update_received and interval_due)
        )
        if should_refresh_goal:
            self.active_goal_position = sanitized_goal.copy()
            self._reset_global_guidance()
            return True

        return False

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
        neighbor_update_received: bool = False,
        target_state: TargetStateMessage | None = None,
        target_update_received: bool = False,
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
        swarm_forced_replan = False
        if not force_generate_new:
            safety_check_time = time.monotonic()
            periodic_safety_due = self.config.safety_check_interval <= 0.0 or (
                safety_check_time - self.last_safety_check_time
                >= self.config.safety_check_interval
            )
            should_run_safety_check = (
                periodic_safety_due or neighbor_update_received
            )
            if should_run_safety_check:
                if periodic_safety_due:
                    self.last_safety_check_time = safety_check_time
                safety_check_result = self._check_safety(
                    msg_time=msg_time,
                    active_command=active_command,
                )
                if safety_check_result.status == SafetyStatus.EMERGENCY:
                    self.in_emergency_mode = True
                    emergency_command = self._build_emergency_stop_command(
                        current_state=current_state,
                        commit_time=time.monotonic(),
                    )
                    return PlannerManagerUpdateResult(
                        command=emergency_command,
                        should_publish=True,
                    )
                if safety_check_result.status == SafetyStatus.REPLAN:
                    safety_forced_replan = True
                    swarm_forced_replan = safety_check_result.triggered_by_swarm

        tracking_forced_replan = False
        if self.config.tracking_enabled:
            self._ingest_target_state(target_state)
            tracking_forced_replan = self._update_tracking_goal(
                current_state=current_state,
                msg_time=msg_time,
                target_update_received=target_update_received,
            )

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
        if self.config.tracking_enabled:
            if self.active_goal_position is None:
                return PlannerManagerUpdateResult(
                    command=QuadrotorCommand.hover(
                        start_time=time.monotonic(),
                        message=(
                            "Tracking goal is invalid after formation scale search. "
                            "Switching to hover."
                        ),
                    ),
                    should_publish=True,
                )
            goal_switched = tracking_forced_replan
        else:
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
        self._update_global_progress_from_position(current_state.position)

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
            or tracking_forced_replan
            or active_command is None
            or not active_command.is_track
        ):
            local_start_pva = current_state.pva.copy()
            local_target_pva = self._get_local_target_from_global_progress(
                local_start_pva=local_start_pva,
                current_state=current_state,
            )
            command_for_local_planner = None

        elif safety_forced_replan:
            local_start_pva = active_command.sample_pva(msg_time)
            local_target_pva = self._get_local_target_from_global_progress(
                local_start_pva=local_start_pva,
                current_state=current_state,
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
                local_target_pva = self._get_local_target_from_global_progress(
                    local_start_pva=local_start_pva,
                    current_state=current_state,
                )
                command_for_local_planner = None

            elif near_goal or near_start:
                return PlannerManagerUpdateResult(
                    command=active_command.copy(),
                    should_publish=False,
                )
            else:
                local_start_pva = active_command.sample_pva(msg_time)
                local_target_pva = self._get_local_target_from_global_progress(
                    local_start_pva=local_start_pva,
                    current_state=current_state,
                )
                command_for_local_planner = active_command

        local_plan_res = self.local_planner.plan(
            grid_map=self.grid_map,
            active_command=command_for_local_planner,
            local_start_pva=local_start_pva,
            local_target_pva=local_target_pva,
            msg_time=msg_time,
            neighbor_trajectories=self._get_swarm_avoidance_neighbors(),
            traj_start_time=msg_time,
            tracking_reference=(
                None
                if self.latest_tracking_reference is None
                else self.latest_tracking_reference.copy()
            ),
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
            if safety_forced_replan and swarm_forced_replan:
                self.last_swarm_replan_time = commit_time
            if self.config.tracking_enabled:
                self.last_tracking_replan_time = commit_time
            self.next_traj_id += 1

            return PlannerManagerUpdateResult(
                command=new_command,
                should_publish=True,
            )

        if active_command is not None:
            if active_command.is_track and active_command.is_track_expired(msg_time):
                return PlannerManagerUpdateResult(
                    command=QuadrotorCommand.hover(
                        start_time=time.monotonic(),
                        message=(
                            "Local planning failed and the previous track has expired. "
                            "Switching to hover."
                        ),
                    ),
                    should_publish=True,
                )
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
        delta_t = max(stop_duration / 3.0, 1e-3)

        # Match ego-swarm's emergency stop representation:
        # a cubic Uniform B-spline whose control points are all clamped
        # to the same stop position.
        stop_trajectory = UniformBSpline(
            control_points=np.repeat(
                stop_position.reshape(1, 3),
                repeats=6,
                axis=0,
            ),
            delta_t=delta_t,
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
    ) -> SafetyCheckResult:
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
            return SafetyCheckResult(status=SafetyStatus.SAFE)

        if not active_command.is_track:
            return SafetyCheckResult(status=SafetyStatus.SAFE)

        trajectory = active_command.trajectory
        if not isinstance(trajectory, UniformBSpline):
            raise RuntimeError(
                "Expected TRACK active_command to carry a UniformBSpline trajectory."
            )
        t_cur = active_command.get_elapsed_time(msg_time)
        traj_duration = float(trajectory.duration)
        if traj_duration <= 0.0 or t_cur >= traj_duration:
            return SafetyCheckResult(status=SafetyStatus.SAFE)

        check_end_time = traj_duration
        two_thirds_time = (2.0 * traj_duration) / 3.0
        if t_cur < two_thirds_time:
            check_end_time = two_thirds_time

        sample_dt = float(self.config.safety_sample_dt)
        if sample_dt <= 0.0:
            raise ValueError(f"Expected safety_sample_dt > 0, but got {sample_dt}.")

        swarm_cooldown_active = (
            self.config.swarm_replan_cooldown > 0.0
            and msg_time - self.last_swarm_replan_time < self.config.swarm_replan_cooldown
        )
        cooldown_logged = False

        sample_t = t_cur
        while sample_t <= check_end_time + 1e-9:
            sample_pva = trajectory.evaluate_pva(sample_t)
            sample_pos = sample_pva[:3]
            if self.grid_map.is_occupied_world(sample_pos):
                time_to_collision = max(0.0, sample_t - t_cur)
                print(
                    f"[PlannerManager {self.agent_id}] "
                    f"static obstacle safety trigger at t+{time_to_collision:.3f}s.",
                    flush=True,
                )
                if time_to_collision < self.config.emergency_time:
                    return SafetyCheckResult(status=SafetyStatus.EMERGENCY)
                return SafetyCheckResult(status=SafetyStatus.REPLAN)

            global_sample_time = active_command.start_time + sample_t
            neighbor_collision = self._find_neighbor_collision(
                sample_pos=sample_pos,
                global_sample_time=global_sample_time,
            )
            if neighbor_collision is not None:
                if swarm_cooldown_active:
                    if not cooldown_logged:
                        remaining_cooldown = max(
                            0.0,
                            self.config.swarm_replan_cooldown
                            - (msg_time - self.last_swarm_replan_time),
                        )
                        print(
                            f"[PlannerManager {self.agent_id}] "
                            f"swarm replan cooldown active "
                            f"({remaining_cooldown:.3f}s left), "
                            f"suppressing neighbor trigger from agent {neighbor_collision}.",
                            flush=True,
                        )
                        cooldown_logged = True
                    sample_t += sample_dt
                    continue
                print(
                    f"[PlannerManager {self.agent_id}] "
                    f"neighbor safety trigger with agent {neighbor_collision} "
                    f"at global_time={global_sample_time:.3f}.",
                    flush=True,
                )
                return SafetyCheckResult(
                    status=SafetyStatus.REPLAN,
                    triggered_by_swarm=True,
                )
            sample_t += sample_dt

        return SafetyCheckResult(status=SafetyStatus.SAFE)

    def _find_neighbor_collision(
        self,
        sample_pos: Vector,
        global_sample_time: float,
    ) -> int | None:
        sample_pos = np.asarray(sample_pos, dtype=np.float64).reshape(3)
        clearance = float(self.config.swarm_clearance)
        if clearance <= 0.0:
            raise ValueError(f"Expected swarm_clearance > 0, but got {clearance}.")

        for neighbor_agent_id, neighbor_record in self.neighbor_traj_buffer.items():
            if not self._should_yield_to_neighbor(int(neighbor_agent_id)):
                continue
            neighbor_duration = float(neighbor_record.trajectory.duration)
            if neighbor_duration <= 0.0:
                continue

            neighbor_t = global_sample_time - neighbor_record.start_time
            if neighbor_t < 0.0 or neighbor_t > neighbor_duration:
                continue

            neighbor_pos = np.asarray(
                neighbor_record.trajectory.evaluate_pva(neighbor_t)[:3],
                dtype=np.float64,
            ).reshape(3)
            if np.linalg.norm(sample_pos - neighbor_pos) < clearance:
                return int(neighbor_agent_id)

        return None

    def _get_swarm_avoidance_neighbors(self) -> tuple[NeighborTrajectoryRecord, ...]:
        return tuple(
            neighbor_record
            for neighbor_agent_id, neighbor_record in self.neighbor_traj_buffer.items()
            if self._should_yield_to_neighbor(int(neighbor_agent_id))
        )

    def _should_yield_to_neighbor(self, neighbor_agent_id: int) -> bool:
        if not self.config.swarm_priority_enabled:
            return True
        return self.agent_id > int(neighbor_agent_id)

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

    def _get_global_progress_sampling_dt(self) -> float:
        max_vel = float(self.local_planner.config.max_vel)
        return float(self.config.planning_horizon) / 20.0 / max_vel

    def _update_global_progress_from_position(self, position: Vector) -> None:
        if self.global_trajectory is None:
            return

        current_position = np.asarray(position, dtype=np.float64).reshape(3)
        t_min = float(self.global_progress)
        t_max = float(self.global_trajectory.duration)
        dt = self._get_global_progress_sampling_dt()

        dist_min = np.inf
        dist_min_t = t_min
        t = t_min
        while t <= t_max + 1e-9:
            pva = self.global_trajectory.evaluate_pva(t)
            traj_position = np.asarray(pva[:3], dtype=np.float64).reshape(3)
            distance = np.linalg.norm(traj_position - current_position)
            if distance < dist_min:
                dist_min = distance
                dist_min_t = t
            t += dt

        self.global_progress = dist_min_t

    def _rebuild_global_guidance_from_state(
        self,
        current_state: QuadrotorState,
    ) -> None:
        self.global_trajectory = self._build_global_trajectory(current_state=current_state)
        self.global_progress = 0.0

    def _find_global_reentry_time(
        self,
        start_position: Vector,
    ) -> float | None:
        if self.global_trajectory is None:
            return None

        start_position = np.asarray(start_position, dtype=np.float64).reshape(3)
        t_min = float(self.global_progress)
        t_max = float(self.global_trajectory.duration)
        dt = self._get_global_progress_sampling_dt()
        planning_horizon = float(self.config.planning_horizon)

        first_pva = self.global_trajectory.evaluate_pva(t_min)
        first_position = np.asarray(first_pva[:3], dtype=np.float64).reshape(3)
        first_distance = np.linalg.norm(first_position - start_position)
        if first_distance <= planning_horizon:
            return t_min

        t = t_min + dt
        while t <= t_max + 1e-9:
            pva = self.global_trajectory.evaluate_pva(t)
            position = np.asarray(pva[:3], dtype=np.float64).reshape(3)
            distance = np.linalg.norm(position - start_position)
            if distance <= planning_horizon:
                return t
            t += dt

        return None

    def _get_local_target_from_global_progress(
        self,
        local_start_pva: Vector,
        current_state: QuadrotorState,
    ) -> Vector:
        if self.global_trajectory is None:
            raise RuntimeError(
                "_get_local_target_from_global_progress() requires a valid global trajectory."
            )
        local_target_pva = self.global_trajectory.evaluate_pva(
            float(self.global_trajectory.duration)
        )
        start_position = np.asarray(local_start_pva[:3], dtype=np.float64).reshape(3)
        max_vel = float(self.local_planner.config.max_vel)
        max_acc = float(self.local_planner.config.max_acc)
        dt = self._get_global_progress_sampling_dt()

        for rebuild_attempt in range(2):
            if self.global_trajectory is None:
                raise RuntimeError(
                    "_get_local_target_from_global_progress() requires a valid global trajectory."
                )
            if self.active_goal_position is None:
                raise RuntimeError(
                    "_get_local_target_from_global_progress() requires an active global goal."
                )

            reentry_t = self._find_global_reentry_time(start_position)
            if reentry_t is None:
                if rebuild_attempt == 1:
                    break
                self._rebuild_global_guidance_from_state(current_state=current_state)
                print(
                    f"[PlannerManager {self.agent_id}] rebuilt global guidance "
                    "because the remaining global trajectory stayed outside "
                    "planning_horizon.",
                    flush=True,
                )
                continue

            t_min = float(reentry_t)
            t_max = float(self.global_trajectory.duration)
            t = t_min
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
            break

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
