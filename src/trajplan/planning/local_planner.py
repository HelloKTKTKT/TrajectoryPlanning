from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from trajplan.map.grid_map import GridMap
from trajplan.planning.bspline_optimizer import BsplineOptimizer
from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.shared_types import Matrix, PvPair, Vector
from trajplan.trajectory.bspline import UniformBSpline
from trajplan.trajectory.polynomial import OneSegmentMinimumJerkTrajectory


@dataclass(slots=True)
class CollisionSegment:
    start_index: int
    end_index: int


# Result of local planning
@dataclass(slots=True)
class LocalPlanningResult:
    status: str
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
    Full local planner skeleton.

    Planned full pipeline
    ---------------------
    Step 1
        Build initial B-spline.
        - generate new
        - or extend old, then fallback to new if needed

    Step 2 + Step 3
        Run an iterative collision-guided optimization loop.
        - detect collision segments
        - build A* guidance and pv pairs
        - optimize
        - re-check in the next loop iteration

    Step 4
        Refine feasibility if needed.
    """

    def __init__(
        self,
        config: LocalPlannerConfig,
        optimizer: BsplineOptimizer,
    ) -> None:
        self.config = config
        self.ts_check = self.config.ctrl_pt_dist / self.config.max_vel * 1.2
        self.optimizer = optimizer

    def plan(
        self,
        grid_map: GridMap,
        active_command: QuadrotorCommand | None,
        local_start_pva: Vector,
        local_target_pva: Vector,
        current_time: float,
    ) -> LocalPlanningResult:
        # --------------------------------------------------------------
        # Step 0: check if local target is already reached, if so, no replan
        # --------------------------------------------------------------
        dist_to_target = np.linalg.norm(local_target_pva[:3] - local_start_pva[:3])
        if dist_to_target < self.config.goal_tol:
            return LocalPlanningResult(
                status="target_close",
                trajectory=None,
                message="Local target is already very close.",
            )

        # --------------------------------------------------------------
        # Step 1: build initial B-spline
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Step 2 + Step 3: iterative collision-guided optimization loop
        # --------------------------------------------------------------
        optimized_bspline, success_ctps_opt = (
            self._iterative_collision_guided_optimization(
                grid_map=grid_map,
                initial_bspline=initial_bspline,
            )
        )

        if not success_ctps_opt:
            return LocalPlanningResult(
                status="failure",
                trajectory=optimized_bspline,
                message="Iterative collision-guided optimization failed.",
            )

        # --------------------------------------------------------------
        # Step 4: feasibility refine
        # --------------------------------------------------------------
        fea_flag, ratio = optimized_bspline.check_feasibility(
            max_velocity=self.config.max_vel, max_acceleration=self.config.max_acc
        )
        if not fea_flag:
            self.optimizer.refine_feasibility(
                optimized_bspline=optimized_bspline,
                ratio=ratio,
            )
        # reparam_bspline
        point_set = []
        for t in np.arange(
            0.0, optimized_bspline.duration + 1e-3, optimized_bspline.delta_t
        ):
            pt = optimized_bspline.evaluate_pva(float(t))[:3]
            point_set.append(pt)
        del point_set[1]
        del point_set[-2]
        point_set = np.vstack(point_set)
        start_pva = optimized_bspline.evaluate_pva(0.0)
        end_pva = optimized_bspline.evaluate_pva(optimized_bspline.duration)
        tip_derivatives = np.vstack(
            (
                start_pva[3:6],
                end_pva[3:6],
                start_pva[6:9],
                end_pva[6:9],
            )
        )
        final_bspline = UniformBSpline.from_point_set(
            point_set=point_set,
            delta_t=optimized_bspline.delta_t,
            boundary_derivatives=tip_derivatives,
        )

        return LocalPlanningResult(
            status="success",
            trajectory=final_bspline,
            message="Local trajectory generated successfully.",
        )

    # ------------------------------------------------------------------
    # Step 1: initial bspline generation
    # ------------------------------------------------------------------

    def _build_initial_bspline(
        self,
        active_command: QuadrotorCommand | None,
        local_start_pva: Vector,
        local_target_pva: Vector,
        current_time: float,
    ) -> UniformBSpline | None:
        if active_command is None:
            return self._generate_new_bspline(
                local_start_pva=local_start_pva,
                local_target_pva=local_target_pva,
            )

        bspline = self._extend_old_bspline(
            active_command=active_command,
            local_target_pva=local_target_pva,
            current_time=current_time,
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
    ) -> UniformBSpline | None:
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

        distance = np.linalg.norm(local_target_pva[:3] - local_start_pva[:3])
        critical_distance = (self.config.max_vel**2) / self.config.max_acc
        if distance <= critical_distance:
            poly_duration = np.sqrt(distance / self.config.max_acc)
        else:
            poly_duration = (
                distance - critical_distance
            ) / self.config.max_vel + 2 * self.config.max_vel / self.config.max_acc

        try:
            init_traj = self._build_minimum_jerk_trajectory(
                local_start_pva=local_start_pva,
                local_target_pva=local_target_pva,
                duration=poly_duration,
            )
        except Exception:
            return None

        try:
            point_set, ts_used = self._sample_polynomial_position_points(
                trajectory=init_traj,
            )
        except Exception:
            return None

        boundary_derivatives = np.vstack(
            (
                local_start_pva[3:6],
                local_target_pva[3:6],
                local_start_pva[6:9],
                local_target_pva[6:9],
            )
        )

        try:
            return UniformBSpline.from_point_set(
                point_set=point_set,
                delta_t=ts_used,
                boundary_derivatives=boundary_derivatives,
            )
        except Exception:
            return None

    def _extend_old_bspline(
        self,
        active_command: QuadrotorCommand,
        local_target_pva: Vector,
        current_time: float,
    ) -> UniformBSpline | None:
        if not active_command.is_track:
            return None
        if active_command.trajectory is None:
            return None

        old_bspline = active_command.trajectory
        current_progress = min(
            active_command.get_elapsed_time(current_time),
            old_bspline.duration,
        )

        pseudo_arc_length = [0.0]
        segment_pts: list[np.ndarray] = []

        # 1. Sample old local trajectory from current progress
        for t in np.arange(
            current_progress,
            old_bspline.duration + 1e-3,
            self.ts_check,
            dtype=np.float64,
        ):
            pt = old_bspline.evaluate_pva(float(t))[:3]
            segment_pts.append(pt)

            if len(segment_pts) >= 2:
                pseudo_arc_length.append(
                    np.linalg.norm(segment_pts[-1] - segment_pts[-2])
                    + pseudo_arc_length[-1]
                )

        # 2. Check if extension is needed
        local_end = old_bspline.evaluate_pva(old_bspline.duration)
        poly_time = np.linalg.norm(
            local_end[:3] - np.asarray(local_target_pva[:3], dtype=np.float64)
        ) / (self.config.max_vel**2)

        if poly_time > self.ts_check:
            try:
                onesegment_traj = self._build_minimum_jerk_trajectory(
                    local_start_pva=local_end,
                    local_target_pva=local_target_pva,
                    duration=poly_time,
                )
            except Exception:
                return None

            for t in np.arange(
                self.ts_check,
                poly_time + 1e-3,
                self.ts_check,
                dtype=np.float64,
            ):
                pt = onesegment_traj.evaluate_position(float(t))
                segment_pts.append(pt)

                if len(segment_pts) >= 2:
                    pseudo_arc_length.append(
                        np.linalg.norm(segment_pts[-1] - segment_pts[-2])
                        + pseudo_arc_length[-1]
                    )

        # 3. Make sure we have at least 2 points
        if len(segment_pts) < 2:
            segment_pts.append(np.asarray(local_target_pva[:3], dtype=np.float64))
            pseudo_arc_length.append(
                np.linalg.norm(segment_pts[-1] - segment_pts[-2])
                + pseudo_arc_length[-1]
            )

        # 4. Resample point_set using pseudo arc length
        point_set = self._resample_segment_points_to_point_set(
            segment_pts=segment_pts,
            pseudo_arc_length=pseudo_arc_length,
            local_target_pva=local_target_pva,
        )

        # 5. Strict fallback rule from simulator_v0.1
        if (
            len(point_set)
            > (self.config.planning_horizon / self.config.ctrl_pt_dist) * 3
        ):
            return None

        del point_set[1]
        del point_set[-2]

        point_set_selected = np.vstack(point_set)

        x_local_cur = old_bspline.evaluate_pva(current_progress)
        tip_derivatives_selected = np.vstack(
            (
                x_local_cur[3:6].flatten(),
                np.asarray(local_target_pva[3:6], dtype=np.float64).flatten(),
                x_local_cur[6:9].flatten(),
                np.zeros(3, dtype=np.float64),
            )
        )

        try:
            return UniformBSpline.from_point_set(
                point_set=point_set_selected,
                delta_t=self.ts_check,
                boundary_derivatives=tip_derivatives_selected,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Step 1 helpers
    # ------------------------------------------------------------------
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

    def _build_minimum_jerk_trajectory(
        self,
        local_start_pva: Vector,
        local_target_pva: Vector,
        duration: float,
    ) -> OneSegmentMinimumJerkTrajectory:
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

        return OneSegmentMinimumJerkTrajectory(
            start_position=local_start_pva[:3],
            end_position=local_target_pva[:3],
            start_velocity=local_start_pva[3:6],
            end_velocity=local_target_pva[3:6],
            start_acceleration=local_start_pva[6:9],
            end_acceleration=local_target_pva[6:9],
            duration=duration,
        )

    def _sample_polynomial_position_points(
        self,
        trajectory: OneSegmentMinimumJerkTrajectory,
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
            else:
                ts_check /= 1.5

        del point_set[1]
        del point_set[-2]

        point_set_selected = np.vstack(point_set)
        return point_set_selected, ts_check

    # ------------------------------------------------------------------
    # Step 2 + Step 3: iterative collision-guided optimization loop
    # ------------------------------------------------------------------

    def _iterative_collision_guided_optimization(
        self,
        grid_map: GridMap,
        initial_bspline: UniformBSpline,
    ) -> tuple[UniformBSpline, bool]:
        bspline = initial_bspline.copy()
        lambda_dist = self.config.lambda_dist_initial
        pv_pair_info = [[] for _ in range(bspline.control_points.shape[0])]
        iter_cur = 0
        success_ctps_opt = False

        while iter_cur < self.config.max_restart_num:
            collision_segments = self._get_collision_segments(
                grid_map=grid_map,
                bspline=bspline,
            )

            if len(collision_segments) == 0:
                success_ctps_opt = True
                break

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

            # grid_map.plot_3d(trajectory=bspline)

            bspline = self.optimizer.optimize(
                initial_bspline=bspline,
                pv_pair_info=pv_pair_info,
                lambda_dist_apply=lambda_dist,
            )

            # grid_map.plot_3d(trajectory=bspline)

            occ_flag = self._check_bspline_collision(
                grid_map=grid_map,
                bspline=bspline,
            )

            if occ_flag:
                lambda_dist = lambda_dist * 2.0
                iter_cur += 1
            else:
                success_ctps_opt = True
                break

        return bspline, success_ctps_opt

    def _get_collision_segments(
        self,
        grid_map: GridMap,
        bspline: UniformBSpline,
    ) -> list[CollisionSegment]:
        ctps = np.asarray(bspline.control_points, dtype=np.float64)

        if ctps.ndim != 2 or ctps.shape[1] != 3:
            raise ValueError(
                "Expected bspline.control_points to have shape (N, 3), "
                f"but got {ctps.shape}."
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

        if ctps.shape[0] <= 2 * order:
            return []

        idx_end = ctps.shape[0] - order

        for i in range(order, idx_end):
            dist = np.linalg.norm(ctps[i + 1] - ctps[i])

            for step in np.arange(
                0.0, dist + step_size - 1e-3, step_size, dtype=np.float64
            ):
                if dist < 1e-3:
                    check_pt = ctps[i]
                else:
                    percentage = min(step / dist, 1.0)
                    check_pt = ctps[i] * (1.0 - percentage) + ctps[i + 1] * percentage

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
        ctps = np.asarray(bspline.control_points, dtype=np.float64)

        if ctps.ndim != 2 or ctps.shape[1] != 3:
            raise ValueError(
                "Expected bspline.control_points to have shape (N, 3), "
                f"but got {ctps.shape}."
            )

        collision_ctps_index_list: list[list[int | float]] = []
        astar_list: list[Matrix] = []

        for seg_id, seg in enumerate(collision_segments):
            start_ctp_index = int(seg.start_index)
            end_ctp_index = int(seg.end_index)

            # Special handling for the first collision segment.
            # If its start index is 3 but ctp[3] is already inside obstacle,
            # search backward to find the nearest free control point.
            if seg_id == 0 and start_ctp_index == 3:
                if grid_map.check_occupancy(ctps[start_ctp_index]):
                    found_free = False
                    for candidate in (2, 1, 0):
                        if not grid_map.check_occupancy(ctps[candidate]):
                            start_ctp_index = candidate
                            found_free = True
                            break
                    if not found_free:
                        raise RuntimeError(
                            "Failed to find a free A* start control point for the first collision segment."
                        )

            start_p = ctps[start_ctp_index]
            target_p = ctps[end_ctp_index]

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

            path_i: list[np.ndarray] = []
            path_i.append(start_p)

            # In the current Python implementation, retrieve_optimal_path already
            # returns points ordered from start to target, so no reverse is needed.
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
    ) -> None:  # here we might modify the input pv_pair_info in-place for efficiency
        for ctps_index_seg_i, astar_seg_i in zip(collision_ctps_index_list, astar_list):
            for ctp_index in ctps_index_seg_i:
                # Convert to integer index and handle fractional indices
                ctp_index_float = float(ctp_index)
                is_int_index = ctp_index_float.is_integer()
                idx = int(ctp_index_float)

                # Calculate control point position
                if is_int_index:
                    ctp = control_points[idx]
                else:
                    ctp = (control_points[idx] + control_points[idx + 1]) / 2

                # Calculate directional law vector more efficiently
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
                    ctp_i=ctp,
                    law_i=law,
                )
                point_diff = intersection_point - ctp
                unit_dir = point_diff / (np.linalg.norm(point_diff) + 1e-6)

                # Get anchor point and store result
                anchor_point = self._get_anchor_point_obs_surface(
                    intersect_pt=intersection_point, control_pt=ctp, grid_map=grid_map
                )

                pv_pair_temp = PvPair(
                    surface_anchor=anchor_point,
                    unit_dir=unit_dir,
                )
                pv_pair_info[idx].append(pv_pair_temp)
        return pv_pair_info

    def _check_bspline_collision(
        self,
        grid_map: GridMap,
        bspline: UniformBSpline,
        current_progress: float = 0.0,
    ) -> bool:
        bspline_start_p = bspline.evaluate_pva(0.0)[:3]
        bspline_end_p = bspline.evaluate_pva(bspline.duration)[:3]
        wp_check_num = (
            np.linalg.norm(bspline_start_p - bspline_end_p) / grid_map.resolution
        )
        t_step = bspline.duration / wp_check_num
        occ_flag = False
        for t in np.arange(current_progress, 2 / 3 * bspline.duration + 1e-3, t_step):
            check_pt = bspline.evaluate_pva(t)[:3]
            occ_flag = grid_map.check_occupancy(check_pt)
            if occ_flag:
                break
        return occ_flag

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
                intersect_point = astar_i[0]
                break
            if astar_id > astar_i.shape[0] - 1:
                intersect_point = astar_i[-1]
                break
            val = (astar_i[astar_id] - ctp_i) @ law_i
            if val * last_val <= 0 and not (abs(val) == 0 and abs(last_val) == 0):
                p_astar_c = astar_i[astar_id]
                p_astar_last = astar_i[last_astar_id]
                direction = p_astar_last - p_astar_c
                percentage = ((ctp_i - p_astar_c) @ law_i) / (
                    (p_astar_last - p_astar_c) @ law_i
                )
                intersect_point = p_astar_c + direction * percentage
                break
            last_val = val
            last_astar_id = astar_id
        return intersect_point

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
        ):
            check_pt = (1 - a / length) * intersect_pt + (a / length) * control_pt
            occ_flag = grid_map.check_occupancy(check_pt)
            if occ_flag:
                percentage = (a - grid_map.resolution) / length
                anchor_pt_obs_surface = (
                    1 - percentage
                ) * intersect_pt + percentage * control_pt
                break
        return anchor_pt_obs_surface
