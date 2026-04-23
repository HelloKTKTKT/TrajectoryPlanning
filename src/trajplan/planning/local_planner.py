from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

import numpy as np

from trajplan.map.grid_map import GridMap
from trajplan.planning.bspline_optimizer import BsplineOptimizer
from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.shared_types import Matrix, PvPair, Vector
from trajplan.trajectory.bspline import UniformBSpline
from trajplan.trajectory.polynomial import MinicostTraj


class _NeighborTrajectoryLike(Protocol):
    agent_id: int
    start_time: float
    trajectory: UniformBSpline


@dataclass(slots=True)
class CollisionSegment:
    start_index: int
    end_index: int


class LocalPlanningStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TARGET_CLOSE = "target_close"


@dataclass(slots=True)
class LocalPlanningResult:
    status: LocalPlanningStatus
    trajectory: UniformBSpline | None
    message: str


@dataclass(slots=True)
class LocalPlannerConfig:
    ctrl_pt_dist: float
    planning_horizon: float
    max_vel: float
    max_acc: float
    collision_weight_increase_factor: float
    max_restart_num: int
    goal_tol: float
    lambda_dist_initial: float


class LocalPlanner:
    """
    Local trajectory planner aligned with the current PlannerManager contract.

    PlannerManager is responsible for:
    - deciding whether planning is needed
    - choosing local_start_pva
    - choosing local_target_pva
    - deciding whether the current active command may be reused

    LocalPlanner is responsible for:
    - building an initial local B-spline
    - trying "extend old" when an active TRACK command is provided
    - running collision-guided optimization
    - refining feasibility if needed
    """

    def __init__(
        self,
        config: LocalPlannerConfig,
        optimizer: BsplineOptimizer,
    ) -> None:
        self.config = config
        self.optimizer = optimizer
        self.ts_check = self.config.ctrl_pt_dist / self.config.max_vel * 1.2

    def plan(
        self,
        grid_map: GridMap,
        active_command: QuadrotorCommand | None,
        local_start_pva: Vector,
        local_target_pva: Vector,
        msg_time: float,
        neighbor_trajectories: Sequence[_NeighborTrajectoryLike] = (),
        traj_start_time: float | None = None,
    ) -> LocalPlanningResult:
        """
        Generate one local B-spline.

        Parameters
        ----------
        grid_map
            Occupancy map used by collision-guided optimization.
        active_command
            Current active TRACK command if the planner is allowed to try
            extending the old local trajectory; otherwise None.
        local_start_pva
            Start p/v/a chosen by PlannerManager.
        local_target_pva
            Local target p/v/a chosen from the maintained global trajectory.
        msg_time
            Timestamp of the state/command snapshot used for this planning tick.
        """
        dist_to_target = np.linalg.norm(local_target_pva[:3] - local_start_pva[:3])
        if dist_to_target < self.config.goal_tol:
            return LocalPlanningResult(
                status=LocalPlanningStatus.TARGET_CLOSE,
                trajectory=None,
                message="Local target is already very close.",
            )

        initial_bspline = self._build_initial_bspline(
            active_command=active_command,
            local_start_pva=local_start_pva,
            local_target_pva=local_target_pva,
            msg_time=msg_time,
        )

        optimized_bspline, success_ctps_opt = (
            self._iterative_collision_guided_optimization(
                grid_map=grid_map,
                initial_bspline=initial_bspline,
                neighbor_trajectories=neighbor_trajectories,
                traj_start_time=msg_time
                if traj_start_time is None
                else traj_start_time,
            )
        )
        if not success_ctps_opt:
            return LocalPlanningResult(
                status=LocalPlanningStatus.FAILURE,
                trajectory=optimized_bspline,
                message="Collision-guided optimization failed.",
            )

        fea_flag, ratio = optimized_bspline.check_feasibility(
            max_velocity=self.config.max_vel,
            max_acceleration=self.config.max_acc,
        )
        if not fea_flag:
            self.optimizer.refine_feasibility(
                optimized_bspline=optimized_bspline,
                ratio=ratio,
                local_start_pva=local_start_pva,
                local_target_pva=local_target_pva,
            )

        return LocalPlanningResult(
            status=LocalPlanningStatus.SUCCESS,
            trajectory=optimized_bspline,
            message="Local trajectory generated successfully.",
        )

    def _build_initial_bspline(
        self,
        active_command: QuadrotorCommand | None,
        local_start_pva: Vector,
        local_target_pva: Vector,
        msg_time: float,
    ) -> UniformBSpline:
        if active_command is None:
            return self._generate_new_bspline(
                local_start_pva=local_start_pva,
                local_target_pva=local_target_pva,
            )

        bspline = self._extend_old_bspline_new_2(
            active_command=active_command,
            local_start_pva=local_start_pva,
            local_target_pva=local_target_pva,
            msg_time=msg_time,
        )

        if bspline is not None:
            return bspline

        return self._generate_new_bspline(
            local_start_pva=local_start_pva,
            local_target_pva=local_target_pva,
        )

    def _generate_new_bspline(
        self,
        local_start_pva: Vector,
        local_target_pva: Vector,
    ) -> UniformBSpline:
        local_start_pva = np.asarray(local_start_pva, dtype=np.float64).reshape(-1)
        local_target_pva = np.asarray(local_target_pva, dtype=np.float64).reshape(-1)

        if local_start_pva.shape != (9,):
            raise ValueError(
                f"Expected local_start_pva shape (9,), but got {local_start_pva.shape}."
            )
        if local_target_pva.shape != (9,):
            raise ValueError(
                f"Expected local_target_pva shape (9,), but got {local_target_pva.shape}."
            )

        distance = float(np.linalg.norm(local_target_pva[:3] - local_start_pva[:3]))
        critical_distance = (self.config.max_vel**2) / self.config.max_acc
        if distance <= critical_distance:
            poly_duration = np.sqrt(distance / self.config.max_acc)
        else:
            poly_duration = (
                distance - critical_distance
            ) / self.config.max_vel + 2.0 * self.config.max_vel / self.config.max_acc

        init_traj = MinicostTraj.from_boundary_pva(
            start_position=local_start_pva[:3],
            end_position=local_target_pva[:3],
            start_velocity=local_start_pva[3:6],
            end_velocity=local_target_pva[3:6],
            start_acceleration=local_start_pva[6:9],
            end_acceleration=local_target_pva[6:9],
            duration=poly_duration,
        )
        point_set, ts_used = self._sample_polynomial_position_points(
            trajectory=init_traj,
        )

        boundary_derivatives = np.vstack(
            (
                local_start_pva[3:6],
                local_target_pva[3:6],
                local_start_pva[6:9],
                local_target_pva[6:9],
            )
        )

        return UniformBSpline.from_point_set(
            point_set=point_set,
            delta_t=ts_used,
            boundary_derivatives=boundary_derivatives,
        )

    def _extend_old_bspline(
        self,
        active_command: QuadrotorCommand,
        local_start_pva: Vector,
        local_target_pva: Vector,
        msg_time: float,
    ) -> UniformBSpline | None:
        old_bspline = active_command.trajectory
        if not isinstance(old_bspline, UniformBSpline):
            raise RuntimeError(
                "Expected TRACK active_command to carry a UniformBSpline trajectory."
            )

        current_progress = min(
            active_command.get_elapsed_time(msg_time),
            old_bspline.duration,
        )

        pseudo_arc_length = [0.0]
        segment_pts: list[np.ndarray] = []
        last_sample_time = current_progress

        for t in np.arange(
            current_progress,
            old_bspline.duration + 1e-3,
            self.ts_check,
            dtype=np.float64,
        ):
            last_sample_time = float(t)
            pt = old_bspline.evaluate_pva(float(t))[:3]
            segment_pts.append(pt)

            if len(segment_pts) >= 2:
                pseudo_arc_length.append(
                    float(np.linalg.norm(segment_pts[-1] - segment_pts[-2]))
                    + pseudo_arc_length[-1]
                )

        local_end = old_bspline.evaluate_pva(last_sample_time)
        poly_time = float(
            np.linalg.norm(local_end[:3] - np.asarray(local_target_pva[:3]))
            / self.config.max_vel
            * 2.0
        )

        if poly_time > self.ts_check:
            one_segment_traj = MinicostTraj.from_boundary_pva(
                start_position=local_end[:3],
                end_position=local_target_pva[:3],
                start_velocity=local_end[3:6],
                end_velocity=local_target_pva[3:6],
                start_acceleration=local_end[6:9],
                end_acceleration=local_target_pva[6:9],
                duration=poly_time,
            )

            for t in np.arange(
                self.ts_check,
                poly_time + 1e-3,
                self.ts_check,
                dtype=np.float64,
            ):
                pt = one_segment_traj.evaluate_position(float(t))
                segment_pts.append(pt)

                if len(segment_pts) >= 2:
                    pseudo_arc_length.append(
                        float(np.linalg.norm(segment_pts[-1] - segment_pts[-2]))
                        + pseudo_arc_length[-1]
                    )

        if len(segment_pts) < 2:
            segment_pts.append(np.asarray(local_target_pva[:3], dtype=np.float64))
            pseudo_arc_length.append(
                float(np.linalg.norm(segment_pts[-1] - segment_pts[-2]))
                + pseudo_arc_length[-1]
            )

        point_set = self._resample_segment_points_to_point_set(
            segment_pts=segment_pts,
            pseudo_arc_length=pseudo_arc_length,
            local_target_pva=local_target_pva,
        )

        if (
            len(point_set)
            > (self.config.planning_horizon / self.config.ctrl_pt_dist) * 3
        ):
            return None

        del point_set[1]
        del point_set[-2]
        point_set = np.vstack(point_set)

        boundary_derivatives = np.vstack(
            (
                np.asarray(local_start_pva[3:6], dtype=np.float64).reshape(3),
                np.asarray(local_target_pva[3:6], dtype=np.float64).reshape(3),
                np.asarray(local_start_pva[6:9], dtype=np.float64).reshape(3),
                np.zeros(3, dtype=np.float64),
            )
        )

        return UniformBSpline.from_point_set(
            point_set=point_set,
            delta_t=self.ts_check,
            boundary_derivatives=boundary_derivatives,
        )

    def _extend_old_bspline_new(
        self,
        active_command: QuadrotorCommand,
        local_start_pva: Vector,
        local_target_pva: Vector,
        msg_time: float,
    ) -> UniformBSpline | None:
        """
        Experimental time-aligned old-trajectory extension.

        Unlike `_extend_old_bspline(...)`, this variant does not resample the
        stitched old-tail/new-segment path by pseudo arc length. It first
        estimates a uniform time step from a nominal 17-control-point target,
        then repeatedly refines that time step if adjacent sampled positions are
        too far apart. Sampling is always performed on a unified time grid.
        """
        old_bspline = active_command.trajectory
        if not isinstance(old_bspline, UniformBSpline):
            raise RuntimeError(
                "Expected TRACK active_command to carry a UniformBSpline trajectory."
            )

        current_progress = min(
            active_command.get_elapsed_time(msg_time),
            old_bspline.duration,
        )
        old_remaining_time = max(0.0, old_bspline.duration - current_progress)

        old_end_pva = old_bspline.evaluate_pva(old_bspline.duration)
        poly_time = float(
            np.linalg.norm(old_end_pva[:3] - np.asarray(local_target_pva[:3]))
            / self.config.max_vel
            * 2.0
        )

        extension_traj: MinicostTraj | None = None
        if poly_time > 1e-9:
            extension_traj = MinicostTraj.from_boundary_pva(
                start_position=old_end_pva[:3],
                end_position=local_target_pva[:3],
                start_velocity=old_end_pva[3:6],
                end_velocity=local_target_pva[3:6],
                start_acceleration=old_end_pva[6:9],
                end_acceleration=local_target_pva[6:9],
                duration=poly_time,
            )

        total_time = old_remaining_time + poly_time
        if total_time <= 1e-9:
            return None

        nominal_control_point_count = 17
        nominal_segment_count = nominal_control_point_count - 3
        ts = total_time / nominal_segment_count

        while True:
            point_set: list[np.ndarray] = []
            max_dist = -1.0
            flag_too_far = False

            sample_times = np.arange(0.0, total_time, ts, dtype=np.float64)
            for tau in sample_times:
                tau_float = float(tau)

                if tau_float <= old_remaining_time + 1e-9:
                    sample_pva = old_bspline.evaluate_pva(current_progress + tau_float)
                else:
                    if extension_traj is None:
                        sample_pva = np.asarray(local_target_pva, dtype=np.float64)
                    else:
                        sample_pva = extension_traj.evaluate_pva(
                            tau_float - old_remaining_time
                        )

                point_set.append(np.asarray(sample_pva[:3], dtype=np.float64))

                if len(point_set) >= 2:
                    dist_cur = np.linalg.norm(point_set[-1] - point_set[-2])
                    max_dist = max(max_dist, dist_cur)

                if max_dist >= self.config.ctrl_pt_dist * 1.5:
                    flag_too_far = True
                    break

            final_position = np.asarray(local_target_pva[:3], dtype=np.float64)
            if len(point_set) == 0 or not np.allclose(point_set[-1], final_position):
                point_set.append(final_position)

            if not flag_too_far:
                break

            ts /= 1.5

        if len(point_set) < 4:
            return None

        del point_set[1]
        del point_set[-2]
        point_set_matrix = np.vstack(point_set)

        start_pva = old_bspline.evaluate_pva(current_progress)
        boundary_derivatives = np.vstack(
            (
                np.asarray(start_pva[3:6], dtype=np.float64).reshape(3),
                np.asarray(local_target_pva[3:6], dtype=np.float64).reshape(3),
                np.asarray(start_pva[6:9], dtype=np.float64).reshape(3),
                np.asarray(local_target_pva[6:9], dtype=np.float64).reshape(3),
            )
        )

        return UniformBSpline.from_point_set(
            point_set=point_set_matrix,
            delta_t=ts,
            boundary_derivatives=boundary_derivatives,
        )

    def _extend_old_bspline_new_2(
        self,
        active_command: QuadrotorCommand,
        local_start_pva: Vector,
        local_target_pva: Vector,
        msg_time: float,
    ) -> UniformBSpline | None:
        """
        Geometry-preserving old-trajectory extension with re-estimated timing.

        This variant keeps the stitched old-tail/new-segment geometry, but it
        does not inherit the old remaining execution time. Instead, it:
        1. samples the stitched spatial path,
        2. resamples that path by pseudo arc length,
        3. estimates a fresh total duration from the stitched path length using
           the same max-vel/max-acc logic as new trajectory generation,
        4. derives a new uniform spline interval from the final point-set size.
        """
        old_bspline = active_command.trajectory
        if not isinstance(old_bspline, UniformBSpline):
            raise RuntimeError(
                "Expected TRACK active_command to carry a UniformBSpline trajectory."
            )

        current_progress = min(
            active_command.get_elapsed_time(msg_time),
            old_bspline.duration,
        )

        pseudo_arc_length = [0.0]
        segment_pts: list[np.ndarray] = []
        last_sample_time = current_progress

        for t in np.arange(
            current_progress,
            old_bspline.duration + 1e-3,
            self.ts_check,
            dtype=np.float64,
        ):
            last_sample_time = float(t)
            pt = np.asarray(old_bspline.evaluate_pva(float(t))[:3], dtype=np.float64)
            segment_pts.append(pt)

            if len(segment_pts) >= 2:
                pseudo_arc_length.append(
                    float(np.linalg.norm(segment_pts[-1] - segment_pts[-2]))
                    + pseudo_arc_length[-1]
                )

        local_end = old_bspline.evaluate_pva(last_sample_time)
        poly_time = float(
            np.linalg.norm(local_end[:3] - np.asarray(local_target_pva[:3]))
            / self.config.max_vel
            * 2.0
        )

        if poly_time > self.ts_check:
            extension_traj = MinicostTraj.from_boundary_pva(
                start_position=local_end[:3],
                end_position=local_target_pva[:3],
                start_velocity=local_end[3:6],
                end_velocity=local_target_pva[3:6],
                start_acceleration=local_end[6:9],
                end_acceleration=local_target_pva[6:9],
                duration=poly_time,
            )

            for t in np.arange(
                self.ts_check,
                poly_time + 1e-3,
                self.ts_check,
                dtype=np.float64,
            ):
                pt = np.asarray(
                    extension_traj.evaluate_position(float(t)),
                    dtype=np.float64,
                )
                segment_pts.append(pt)

                if len(segment_pts) >= 2:
                    pseudo_arc_length.append(
                        float(np.linalg.norm(segment_pts[-1] - segment_pts[-2]))
                        + pseudo_arc_length[-1]
                    )

        final_position = np.asarray(local_target_pva[:3], dtype=np.float64)
        if len(segment_pts) == 0:
            segment_pts.append(final_position.copy())
            pseudo_arc_length = [0.0]
        elif not np.allclose(segment_pts[-1], final_position):
            segment_pts.append(final_position.copy())
            pseudo_arc_length.append(
                float(np.linalg.norm(segment_pts[-1] - segment_pts[-2]))
                + pseudo_arc_length[-1]
            )

        if len(segment_pts) < 2 or pseudo_arc_length[-1] <= 1e-9:
            return None

        point_set = self._resample_segment_points_to_point_set(
            segment_pts=segment_pts,
            pseudo_arc_length=pseudo_arc_length,
            local_target_pva=local_target_pva,
        )

        if (
            len(point_set)
            > (self.config.planning_horizon / self.config.ctrl_pt_dist) * 3
        ):
            return None

        total_length = float(pseudo_arc_length[-1])
        total_time = self._estimate_travel_time_from_distance(total_length)
        if total_time <= 1e-9:
            return None

        del point_set[1]
        del point_set[-2]
        point_set_matrix = np.vstack(point_set)

        segment_count = point_set_matrix.shape[0] + 1
        if segment_count <= 0:
            return None
        ts = total_time / float(segment_count)
        ts = max(ts, 1e-3)

        start_pva = old_bspline.evaluate_pva(current_progress)
        boundary_derivatives = np.vstack(
            (
                np.asarray(start_pva[3:6], dtype=np.float64).reshape(3),
                np.asarray(local_target_pva[3:6], dtype=np.float64).reshape(3),
                np.asarray(start_pva[6:9], dtype=np.float64).reshape(3),
                np.asarray(local_target_pva[6:9], dtype=np.float64).reshape(3),
            )
        )

        return UniformBSpline.from_point_set(
            point_set=point_set_matrix,
            delta_t=ts,
            boundary_derivatives=boundary_derivatives,
        )

    def _resample_segment_points_to_point_set(
        self,
        segment_pts: list[np.ndarray],
        pseudo_arc_length: list[float],
        local_target_pva: Vector,
    ) -> list[np.ndarray]:
        sample_dist = self.config.ctrl_pt_dist

        while True:
            point_set: list[np.ndarray] = []
            sample_length = 0.0
            idx = 0

            while (idx <= len(segment_pts) - 2) and (
                sample_length <= pseudo_arc_length[-1]
            ):
                if pseudo_arc_length[idx] <= sample_length < pseudo_arc_length[idx + 1]:
                    dist_to_id = sample_length - pseudo_arc_length[idx]
                    dist_to_id_next = pseudo_arc_length[idx + 1] - sample_length
                    dist_total = pseudo_arc_length[idx + 1] - pseudo_arc_length[idx]

                    if dist_total < 1e-6:
                        pt_selected = segment_pts[idx]
                    else:
                        pt_selected = (
                            dist_to_id_next / dist_total * segment_pts[idx]
                            + dist_to_id / dist_total * segment_pts[idx + 1]
                        )
                    point_set.append(pt_selected)
                    sample_length += sample_dist
                else:
                    idx += 1

            point_set.append(np.asarray(local_target_pva[:3], dtype=np.float64))
            if len(point_set) >= 7:
                break
            sample_dist /= 1.5

        return point_set

    def _sample_polynomial_position_points(
        self,
        trajectory: MinicostTraj,
    ) -> tuple[Matrix, float]:
        ts_check = self.ts_check
        duration = trajectory.duration

        while True:
            point_set: list[np.ndarray] = []
            max_dist = -1.0
            flag_too_far = False

            sample_times = np.arange(0.0, duration, ts_check, dtype=np.float64)
            for t in sample_times:
                pt = trajectory.evaluate_position(float(t))
                point_set.append(pt)

                if len(point_set) >= 2:
                    dist_cur = np.linalg.norm(point_set[-1] - point_set[-2])
                    max_dist = max(max_dist, dist_cur)

                if max_dist >= self.config.ctrl_pt_dist * 1.5:
                    flag_too_far = True
                    break

            if len(point_set) == 0 or not np.allclose(
                point_set[-1], trajectory.evaluate_position(duration)
            ):
                point_set.append(trajectory.evaluate_position(duration))

            if not flag_too_far and len(point_set) >= 7:
                break
            ts_check /= 1.5

        del point_set[1]
        del point_set[-2]
        return np.vstack(point_set), ts_check

    def _estimate_travel_time_from_distance(self, distance: float) -> float:
        distance = float(max(distance, 0.0))
        critical_distance = (self.config.max_vel**2) / self.config.max_acc
        if distance <= critical_distance:
            return float(np.sqrt(distance / self.config.max_acc))
        return float(
            (distance - critical_distance) / self.config.max_vel
            + 2.0 * self.config.max_vel / self.config.max_acc
        )

    def _iterative_collision_guided_optimization(
        self,
        grid_map: GridMap,
        initial_bspline: UniformBSpline,
        neighbor_trajectories: Sequence[_NeighborTrajectoryLike],
        traj_start_time: float,
    ) -> tuple[UniformBSpline, bool]:
        bspline = initial_bspline.copy()
        lambda_avoidance = self.config.lambda_dist_initial
        pv_pair_info: list[list[PvPair]] = [
            [] for _ in range(bspline.control_points.shape[0])
        ]
        iter_cur = 0
        success_ctps_opt = False

        while iter_cur < self.config.max_restart_num:
            collision_segments = self._get_collision_segments(
                grid_map=grid_map,
                bspline=bspline,
            )
            has_neighbor_collision = self._check_bspline_neighbor_collision(
                bspline=bspline,
                neighbor_trajectories=neighbor_trajectories,
                traj_start_time=traj_start_time,
            )

            if len(collision_segments) == 0 and not has_neighbor_collision:
                success_ctps_opt = True
                break

            if len(collision_segments) > 0:
                collision_ctps_index_list, astar_list = (
                    self._get_astar_paths_and_collision_ctps(
                        grid_map=grid_map,
                        bspline=bspline,
                        collision_segments=collision_segments,
                    )
                )

                pv_pair_info = self._update_pv_pair_info(
                    collision_ctps_index_list=collision_ctps_index_list,
                    astar_list=astar_list,
                    pv_pair_info=pv_pair_info,
                    control_points=np.asarray(bspline.control_points, dtype=np.float64),
                    grid_map=grid_map,
                )

            bspline = self.optimizer.optimize(
                initial_bspline=bspline,
                pv_pair_info=pv_pair_info,
                lambda_dist_apply=lambda_avoidance,
                lambda_swarm_apply=lambda_avoidance,
                traj_start_time=traj_start_time,
                neighbor_trajectories=neighbor_trajectories,
            )

            occ_flag = self._check_bspline_collision(
                grid_map=grid_map,
                bspline=bspline,
            )
            neighbor_occ_flag = self._check_bspline_neighbor_collision(
                bspline=bspline,
                neighbor_trajectories=neighbor_trajectories,
                traj_start_time=traj_start_time,
            )
            if occ_flag or neighbor_occ_flag:
                lambda_avoidance *= self.config.collision_weight_increase_factor
                iter_cur += 1
            else:
                success_ctps_opt = True
                break

        return bspline, success_ctps_opt

    def _check_bspline_neighbor_collision(
        self,
        bspline: UniformBSpline,
        neighbor_trajectories: Sequence[_NeighborTrajectoryLike],
        traj_start_time: float,
        current_progress: float = 0.0,
    ) -> bool:
        if len(neighbor_trajectories) == 0:
            return False

        sample_divisor = max(int(self.optimizer.config.swarm_sample_divisor), 1)
        t_step = bspline.delta_t / float(sample_divisor)
        clearance = float(self.optimizer.config.swarm_clearance)
        ellipsoid_a = 2.0
        ellipsoid_b = 1.0
        inv_a2 = 1.0 / (ellipsoid_a * ellipsoid_a)
        inv_b2 = 1.0 / (ellipsoid_b * ellipsoid_b)

        for t in np.arange(
            current_progress,
            2.0 / 3.0 * bspline.duration + 1e-3,
            t_step,
            dtype=np.float64,
        ):
            check_pt = np.asarray(bspline.evaluate_pva(float(t))[:3], dtype=np.float64)
            global_sample_time = float(traj_start_time) + float(t)

            for neighbor in neighbor_trajectories:
                neighbor_t = global_sample_time - neighbor.start_time
                neighbor_duration = float(neighbor.trajectory.duration)
                if neighbor_t < 0.0 or neighbor_t > neighbor_duration:
                    continue

                neighbor_pt = np.asarray(
                    neighbor.trajectory.evaluate_pva(neighbor_t)[:3],
                    dtype=np.float64,
                )
                diff = np.asarray(check_pt - neighbor_pt, dtype=np.float64).reshape(3)
                ellip_dist = np.sqrt(
                    diff[2] * diff[2] * inv_a2
                    + (diff[0] * diff[0] + diff[1] * diff[1]) * inv_b2
                )
                if ellip_dist < clearance:
                    return True

        return False

    def _get_collision_segments(
        self,
        grid_map: GridMap,
        bspline: UniformBSpline,
    ) -> list[CollisionSegment]:
        control_points = np.asarray(bspline.control_points, dtype=np.float64)
        if control_points.ndim != 2 or control_points.shape[1] != 3:
            raise ValueError(
                "Expected bspline.control_points to have shape (N, 3), "
                f"but got {control_points.shape}."
            )

        collision_segments: list[CollisionSegment] = []
        step_size = grid_map.resolution / 2.0
        occ_flag = False
        last_occ_flag = False
        in_index = 0
        out_index = 0
        in_flag = False
        out_flag = False
        order = 3

        if control_points.shape[0] <= 2 * order:
            return []

        idx_end = control_points.shape[0] - order
        for i in range(order, idx_end):
            dist = np.linalg.norm(control_points[i + 1] - control_points[i])

            for step in np.arange(
                0.0, dist + step_size - 1e-3, step_size, dtype=np.float64
            ):
                if dist < 1e-3:
                    check_pt = control_points[i]
                else:
                    percentage = min(step / dist, 1.0)
                    check_pt = (
                        control_points[i] * (1.0 - percentage)
                        + control_points[i + 1] * percentage
                    )

                occ_flag = grid_map.check_occupancy(check_pt)
                if occ_flag and not last_occ_flag:
                    in_index = i
                    in_flag = True
                elif last_occ_flag and not occ_flag:
                    out_index = i + 1
                    out_flag = True

                last_occ_flag = occ_flag

                if in_flag and out_flag:
                    collision_segments.append(
                        CollisionSegment(
                            start_index=int(in_index),
                            end_index=int(out_index),
                        )
                    )
                    in_flag = False
                    out_flag = False

        i = 0
        while i < len(collision_segments) - 1:
            if collision_segments[i + 1].start_index < collision_segments[i].end_index:
                collision_segments[i] = CollisionSegment(
                    start_index=collision_segments[i].start_index,
                    end_index=collision_segments[i + 1].end_index,
                )
                del collision_segments[i + 1]
            else:
                i += 1

        return collision_segments

    def _get_astar_paths_and_collision_ctps(
        self,
        grid_map: GridMap,
        bspline: UniformBSpline,
        collision_segments: list[CollisionSegment],
    ) -> tuple[list[list[int | float]], list[Matrix]]:
        control_points = np.asarray(bspline.control_points, dtype=np.float64)
        if control_points.ndim != 2 or control_points.shape[1] != 3:
            raise ValueError(
                "Expected bspline.control_points to have shape (N, 3), "
                f"but got {control_points.shape}."
            )

        collision_ctps_index_list: list[list[int | float]] = []
        astar_list: list[Matrix] = []

        for seg_id, seg in enumerate(collision_segments):
            start_ctp_index = int(seg.start_index)
            end_ctp_index = int(seg.end_index)

            if seg_id == 0 and start_ctp_index == 3:
                if grid_map.check_occupancy(control_points[start_ctp_index]):
                    found_free = False
                    for candidate in (2, 1, 0):
                        if not grid_map.check_occupancy(control_points[candidate]):
                            start_ctp_index = candidate
                            found_free = True
                            break
                    if not found_free:
                        raise RuntimeError(
                            "Failed to find a free A* start control point for the first collision segment."
                        )

            start_p = control_points[start_ctp_index]
            target_p = control_points[end_ctp_index]
            start_index = grid_map.world_to_grid(start_p)
            target_index = grid_map.world_to_grid(target_p)

            grid_map.search_reset()
            grid_map.update_start_target_index(start_index, target_index)
            success = grid_map.astar_search()
            if not success:
                raise RuntimeError(
                    f"A* failed for collision segment ({start_ctp_index}, {end_ctp_index})."
                )

            optimal_path = grid_map.retrieve_optimal_path()
            if len(optimal_path) == 0:
                raise RuntimeError(
                    f"A* returned an empty path for collision segment ({start_ctp_index}, {end_ctp_index})."
                )

            path_i: list[np.ndarray] = [start_p]
            for node_index in optimal_path[1:-1]:
                path_i.append(grid_map.grid_to_world(node_index))
            path_i.append(target_p)
            astar_list.append(np.vstack(path_i))

            collision_ctps_index: list[int | float] = []
            if end_ctp_index - start_ctp_index == 1:
                collision_ctps_index.append((start_ctp_index + end_ctp_index) / 2)
            else:
                for i in range(int(start_ctp_index + 1), int(end_ctp_index)):
                    collision_ctps_index.append(i)
            collision_ctps_index_list.append(collision_ctps_index)

        return collision_ctps_index_list, astar_list

    def _update_pv_pair_info(
        self,
        collision_ctps_index_list: list[list[int | float]],
        astar_list: list[Matrix],
        pv_pair_info: list[list[PvPair]],
        control_points: Matrix,
        grid_map: GridMap,
    ) -> list[list[PvPair]]:
        for ctps_index_seg_i, astar_seg_i in zip(collision_ctps_index_list, astar_list):
            if len(ctps_index_seg_i) == 0:
                continue

            for ctp_index in ctps_index_seg_i:
                ctp_index_float = float(ctp_index)
                is_int_index = ctp_index_float.is_integer()
                idx = int(ctp_index_float)

                if is_int_index:
                    control_point = control_points[idx]
                else:
                    control_point = (control_points[idx] + control_points[idx + 1]) / 2

                if idx == 0:
                    law = control_points[1] - control_points[0]
                elif idx == control_points.shape[0] - 1:
                    law = control_points[-1] - control_points[-2]
                else:
                    if is_int_index:
                        law = control_points[idx + 1] - control_points[idx - 1]
                    else:
                        law = control_points[idx + 1] - control_points[idx]

                intersection_point = self._find_intersection_point(
                    astar_i=astar_seg_i,
                    ctp_i=control_point,
                    law_i=law,
                )
                point_diff = intersection_point - control_point
                unit_dir = point_diff / (np.linalg.norm(point_diff) + 1e-6)
                anchor_point = self._get_anchor_point_obs_surface(
                    intersect_pt=intersection_point,
                    control_pt=control_point,
                    grid_map=grid_map,
                )

                pv_pair_info[idx].append(
                    PvPair(
                        surface_anchor=anchor_point,
                        unit_dir=unit_dir,
                    )
                )

            first_ctp_index = float(ctps_index_seg_i[0])
            if first_ctp_index.is_integer():
                left_source_index = int(first_ctp_index)
                left_endpoint_index = left_source_index - 1
            else:
                left_source_index = int(first_ctp_index)
                left_endpoint_index = left_source_index

            if (
                0 <= left_endpoint_index < len(pv_pair_info)
                and left_endpoint_index != left_source_index
                and len(pv_pair_info[left_source_index]) > 0
            ):
                left_source_pair = pv_pair_info[left_source_index][-1]
                pv_pair_info[left_endpoint_index].append(
                    PvPair(
                        surface_anchor=left_source_pair.surface_anchor.copy(),
                        unit_dir=left_source_pair.unit_dir.copy(),
                    )
                )

            last_ctp_index = float(ctps_index_seg_i[-1])
            if last_ctp_index.is_integer():
                right_source_index = int(last_ctp_index)
                right_endpoint_index = right_source_index + 1
            else:
                right_source_index = int(last_ctp_index)
                right_endpoint_index = right_source_index + 1

            if (
                0 <= right_endpoint_index < len(pv_pair_info)
                and right_endpoint_index != right_source_index
                and len(pv_pair_info[right_source_index]) > 0
            ):
                right_source_pair = pv_pair_info[right_source_index][-1]
                pv_pair_info[right_endpoint_index].append(
                    PvPair(
                        surface_anchor=right_source_pair.surface_anchor.copy(),
                        unit_dir=right_source_pair.unit_dir.copy(),
                    )
                )
        return pv_pair_info

    def _check_bspline_collision(
        self,
        grid_map: GridMap,
        bspline: UniformBSpline,
        current_progress: float = 0.0,
    ) -> bool:
        bspline_start_p = bspline.evaluate_pva(0.0)[:3]
        bspline_end_p = bspline.evaluate_pva(bspline.duration)[:3]
        straight_line_dist = np.linalg.norm(bspline_start_p - bspline_end_p)
        if straight_line_dist < 1e-6:
            return grid_map.check_occupancy(bspline_start_p)

        wp_check_num = max(straight_line_dist / grid_map.resolution, 1.0)
        t_step = bspline.duration / wp_check_num
        for t in np.arange(
            current_progress,
            2.0 / 3.0 * bspline.duration + 1e-3,
            t_step,
            dtype=np.float64,
        ):
            check_pt = bspline.evaluate_pva(float(t))[:3]
            if grid_map.check_occupancy(check_pt):
                return True
        return False

    @staticmethod
    def _find_intersection_point(
        astar_i: Matrix,
        ctp_i: Vector,
        law_i: Vector,
    ) -> Vector:
        astar_id = astar_i.shape[0] // 2
        last_astar_id = astar_id
        last_val = (astar_i[astar_id] - ctp_i) @ law_i
        while True:
            if last_val >= 0:
                astar_id -= 1
            else:
                astar_id += 1

            if astar_id < 0:
                return astar_i[0]
            if astar_id > astar_i.shape[0] - 1:
                return astar_i[-1]

            val = (astar_i[astar_id] - ctp_i) @ law_i
            if val * last_val <= 0 and not (abs(val) == 0 and abs(last_val) == 0):
                p_astar_c = astar_i[astar_id]
                p_astar_last = astar_i[last_astar_id]
                direction = p_astar_last - p_astar_c
                percentage = ((ctp_i - p_astar_c) @ law_i) / (
                    (p_astar_last - p_astar_c) @ law_i
                )
                return p_astar_c + direction * percentage

            last_val = val
            last_astar_id = astar_id

    @staticmethod
    def _get_anchor_point_obs_surface(
        intersect_pt: Vector,
        control_pt: Vector,
        grid_map: GridMap,
    ) -> Vector:
        length = np.linalg.norm(intersect_pt - control_pt) + 1e-6
        anchor_pt_obs_surface = intersect_pt
        for a in np.arange(
            grid_map.resolution,
            length + grid_map.resolution - 1e-6,
            grid_map.resolution,
            dtype=np.float64,
        ):
            check_pt = (1.0 - a / length) * intersect_pt + (a / length) * control_pt
            if grid_map.check_occupancy(check_pt):
                percentage = (a - grid_map.resolution) / length
                anchor_pt_obs_surface = (
                    1.0 - percentage
                ) * intersect_pt + percentage * control_pt
                break
        return anchor_pt_obs_surface
