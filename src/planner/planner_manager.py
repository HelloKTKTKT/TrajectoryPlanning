import itertools
import copy
import time
import numpy as np
import src.trajectory.parameters as traj_para
import src.map_search.parameters as map_para
from src.trajectory.trajectory_manager import LinearMpcTraj, UniformBspline, MinisnapTraj
from scipy.optimize import minimize


class PlannerManager:
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.global_trajectory = LinearMpcTraj()
        self.local_trajectory = UniformBspline()
        self.local_search_index_list = self.get_local_search_index_list()

    def get_local_search_index_list(self):
        xmax, ymax, zmax = self.grid_map.map_size.flatten()
        temp = [np.array([x, y, z], dtype=int) for x, y, z in itertools.product(range(-xmax + 1, xmax),
                                                                                range(-ymax + 1, ymax),
                                                                                range(-zmax + 1, zmax))]
        return sorted(temp, key=lambda t: abs(t[0]) + abs(t[1]) + abs(t[2]))

    def generate_global_trajectory(self, start_state, target_state, start_time):
        """
        :param start_state: should be collision free
        :param target_state: should be collision free
        :param start_time: time stamp
        Generate global trajectory using LMPC, with traj.const.MAX_DURATION (default 10s). Then cut when stabilizing
        around the xtar, then update the actural Xpred and duration, reset current_progress.
        """
        self.global_trajectory.upadte_traj_info(start_state, target_state, start_time)

    def generate_local_trajectory(self, type_start, flag_new_traj, xc_pva, start_time):
        """

        :param type_start: indicating whether the local trajectory is generated from "ODOM" or "LOCAL".
        :param flag_new_traj: T or F, indicating if the local trajectory is generated using previous bspline.
        :param xc_pva:
        :param start_time:
        :return:
        """
        # Step 1: find the start and target pva, update global progress
        if type_start == "ODOM":
            x_start_pva = xc_pva
        else:  # type_start = "LOCAL"
            x_start_pva = self.local_trajectory.evaluate_pva(self.local_trajectory.current_progress)
        x_target_pva = self.get_local_target_update_global_progress(x_start_pva)

        # Step 2: get initial local bspline
        dist_to_target = np.linalg.norm(x_target_pva[:3] - x_start_pva[:3])
        if dist_to_target < 0.2:
            return True
        else:
            if flag_new_traj:
                bspline = generate_new_bspline(x_start_pva, x_target_pva, start_time)
            else:
                bspline = self.extend_old_bspline(x_start_pva, x_target_pva, start_time)

        # Step 3: control points optimize
        lambda_dist = traj_para.LAMBDA_DIST
        pv_pair_info = [[] for _ in range(bspline.control_points.shape[0])]
        iter_cur = 0
        success_ctps_opt = False
        while iter_cur < traj_para.MAX_RESART_NUMS_SET:
            """
            Find the collision segments. For each segment, get the list of collision ctp index, and the astar.
            Note: when finding the collision ctps, we get rid of the first 3 and last 3 points, and for the middle part,
            we only consider the first 2/3 points.
            """
            collision_ctps_index_list, astar_list = self.get_collision_ctps_astar_pathes(bspline.control_points)

            # Find the pv_pair for each collision ctps, and append it to the corresponding pv_pair_info.
            self.update_pv_pair_info(collision_ctps_index_list, astar_list, pv_pair_info, bspline.control_points)

            # LBFGS optimize and bspline info update
            if len(collision_ctps_index_list) == 0:
                success_ctps_opt = True
                break
            else:
                start_optimize(bspline, pv_pair_info, lambda_dist)
                occ_flag = self.check_bspline_collision(bspline)
                if occ_flag:
                    lambda_dist = lambda_dist * 2
                    iter_cur += 1
                else:
                    success_ctps_opt = True
                    break

        # Step 4:  refine the trajectory
        if success_ctps_opt:
            fea_flag, ratio = bspline.check_feasibility()
            if not fea_flag:
                start_refine(bspline, ratio)
            self.local_trajectory = reparam_bspline(bspline)

        return success_ctps_opt

    def get_local_target_update_global_progress(self, x_start_pva):
        t_step = traj_para.PLANNING_HORIZON / (traj_para.MAX_V * 20)
        dist_min = np.inf
        dist_min_t = self.global_trajectory.current_progress
        global_xtar = self.global_trajectory.evaluate_pva(t_cur=self.global_trajectory.duration)
        local_target = global_xtar  # Default to global target if no conditions are met
        for t in np.arange(self.global_trajectory.current_progress, self.global_trajectory.duration + 1e-3, t_step):
            global_xref_t = self.global_trajectory.evaluate_pva(t_cur=t)
            dist = np.linalg.norm(global_xref_t[:3] - x_start_pva[:3])

            # Update minimum distance and progress
            if dist < dist_min:
                dist_min = dist
                dist_min_t = t

            if dist >= traj_para.PLANNING_HORIZON:
                local_target = global_xref_t
                break
        self.global_trajectory.current_progress = dist_min_t
        dist_to_end = np.linalg.norm(local_target[:3] - global_xtar[:3])
        threshold = (traj_para.MAX_V ** 2) / (2 * traj_para.MAX_A)
        if dist_to_end >= threshold:
            local_target_v = local_target[3:6]
        else:
            local_target_v = np.zeros((3, 1))
        local_target = np.vstack((local_target[:3], local_target_v, np.zeros((3, 1))))
        return local_target

    def extend_old_bspline(self, x_start_pva, x_target_pva, start_time):
        ts_check = traj_para.CTRL_PT_DIST / traj_para.MAX_V * 1.2
        pseudo_arc_length = [0.0]
        segment_pts = []
        for t in np.arange(self.local_trajectory.current_progress, self.local_trajectory.duration + 1e-3, ts_check):
            pt = self.local_trajectory.evaluate_pva(t_cur=t)[:3]
            segment_pts.append(pt)
            if len(segment_pts) >= 2:
                pseudo_arc_length.append(np.linalg.norm(segment_pts[-1] - segment_pts[-2]) + pseudo_arc_length[-1])

        # Check if we need to extend an additional segment of trajectory and sample
        local_end = self.local_trajectory.evaluate_pva(t_cur=self.local_trajectory.duration)
        poly_time = (np.linalg.norm(local_end[:3] - x_target_pva[:3])) / (traj_para.MAX_V ** 2)
        if poly_time > ts_check:
            onesegment_traj = MinisnapTraj(waypoints=np.hstack((local_end[:3], x_target_pva[:3])).T,
                                           ho_start=np.hstack((local_end[3:6], local_end[6:9])).T,
                                           ho_end=np.hstack((x_target_pva[3:6], np.zeros((3, 1)))).T,
                                           s=3,
                                           time_idx=np.array([0, poly_time]))
            for t in np.arange(ts_check, poly_time + 1e-3, ts_check):
                pt = onesegment_traj.get_pos(t).reshape(-1, 1)
                segment_pts.append(pt)
                if len(segment_pts) >= 2:
                    pseudo_arc_length.append(np.linalg.norm(segment_pts[-1] - segment_pts[-2]) + pseudo_arc_length[-1])

        # Make sure we have at least 2 points inside segment_pts []
        if len(segment_pts) < 2:
            segment_pts.append(x_target_pva[:3])
            pseudo_arc_length.append(np.linalg.norm(segment_pts[-1] - segment_pts[-2]) + pseudo_arc_length[-1])

        # sample point set
        sample_dist = traj_para.CTRL_PT_DIST
        while True:
            point_set = []
            sample_length = 0.0
            idx = 0
            while (idx <= len(segment_pts) - 2) and (sample_length <= pseudo_arc_length[-1]):
                if pseudo_arc_length[idx] <= sample_length < pseudo_arc_length[idx + 1]:
                    dist_to_id = sample_length - pseudo_arc_length[idx]
                    dist_to_id_next = pseudo_arc_length[idx + 1] - sample_length
                    dist_total = pseudo_arc_length[idx + 1] - pseudo_arc_length[idx]
                    pt_selected = (dist_to_id / dist_total * segment_pts[idx] +
                                   dist_to_id_next / dist_total * segment_pts[idx + 1])
                    point_set.append(pt_selected)
                    sample_length += sample_dist
                else:
                    idx += 1
            point_set.append(x_target_pva[:3])
            if len(point_set) >= 7:
                break
            sample_dist /= 1.5

        if len(point_set) > (traj_para.PLANNING_HORIZON / traj_para.CTRL_PT_DIST) * 3:
            new_bspline = generate_new_bspline(x_start_pva, x_target_pva, start_time)
        else:
            del point_set[1]
            del point_set[-2]
            point_set_selected = np.hstack(point_set).T
            x_local_cur = self.local_trajectory.evaluate_pva(self.local_trajectory.current_progress)
            tip_derivatives_selected = np.vstack((x_local_cur[3:6].flatten(), x_target_pva[3:6].flatten(),
                                                  x_local_cur[6:9].flatten(), np.zeros(3)))
            new_bspline = UniformBspline()
            new_bspline.generate_unifrom_bspline_trajectory(
                start_time=start_time, ts=ts_check, point_set=point_set_selected,
                tip_derivatives=tip_derivatives_selected)

        return new_bspline

    def get_collision_ctps_astar_pathes(self, ctps):
        collision_segments = []
        step_size = map_para.GRID_SIZE / 2
        occ_flag, last_occ_flag = False, False
        in_index, out_index = 0, 0
        in_flag, out_flag = False, False
        order = 3
        idx_end = ctps.shape[0] - order
        # idx_end = int(ctps.shape[0] - order - ((ctps.shape[0] - 2 * order) // 3)) + 1
        for i in range(order, idx_end):
            dist = np.linalg.norm(ctps[i + 1] - ctps[i])
            for step in np.arange(0, dist + step_size - 1e-3, step_size):
                if dist < 1e-3:
                    check_pt = ctps[i]
                else:
                    percentage = min(step / dist, 1.0)
                    check_pt = ctps[i] * (1 - percentage) + ctps[i + 1] * percentage
                occ_flag = self.grid_map.check_occupancy(check_pt)
                if occ_flag and not last_occ_flag:  # steps into an obstacle
                    in_index = i
                    in_flag = True
                elif last_occ_flag and not occ_flag:  # steps out an obstacle
                    out_index = i + 1
                    out_flag = True
                last_occ_flag = occ_flag
                if in_flag and out_flag:
                    collision_segments.append([int(in_index), int(out_index)])
                    in_flag, out_flag = False, False
        # segments merge
        i = 0
        while True:
            if i + 1 > len(collision_segments) - 1:
                break
            if collision_segments[i + 1][0] < collision_segments[i][1]:
                collision_segments[i][1] = collision_segments[i + 1][1]
                del collision_segments[i + 1]
            else:
                i += 1

        # get astar pathes and collision control point index
        collision_ctps_index_list = []
        astar_list = []
        for seg in collision_segments:
            t_start = time.time()

            path_i = []

            self.grid_map.search_reset()  # this take around 65ms out of 3700ms

            start_p = ctps[seg[0]]
            target_p = ctps[seg[1]]
            start_index = (start_p / map_para.GRID_SIZE).astype(int)
            target_index = (target_p / map_para.GRID_SIZE).astype(int)

            self.grid_map.update_start_target_index(start_index, target_index)  # this take 3645ms out of 3700ms
            self.grid_map.astar_search()
            # self.grid_map.retrieve_optimal_path()
            # path_i.append(start_p)
            # for node in reversed(self.grid_map.optimal_path[1:-1]):
            #     path_i.append(node.index.flatten() * map_para.GRID_SIZE + np.ones(3) * map_para.GRID_SIZE / 2)
            optimal_path = self.grid_map.retrieve_optimal_path()
            path_i.append(start_p)
            for node_index in reversed(optimal_path[1:-1]):
                path_i.append(node_index.flatten() * map_para.GRID_SIZE + np.ones(3) * map_para.GRID_SIZE / 2)

            path_i.append(target_p)
            path_i = np.vstack(path_i)
            astar_list.append(path_i)

            collision_ctps_index = []
            if seg[1] - seg[0] == 1:
                collision_ctps_index.append((seg[0] + seg[1]) / 2)
            else:
                for i in range(int(seg[0] + 1), int(seg[1])):
                    collision_ctps_index.append(i)
            collision_ctps_index_list.append(collision_ctps_index)

            print(f'Astar search time: {((time.time() - t_start) * 1_000):.3f} ms')

        return collision_ctps_index_list, astar_list

    def update_pv_pair_info(self, collision_ctps_index_list, astar_list, pv_pair_info, ctps_opt):
        
        def find_intersection_point(astar_i, ctp_i, law_i):
            astar_id = astar_i.shape[0] // 2
            last_astar_id = astar_id
            last_val = (astar_i[astar_id] - ctp_i) @ law_i
            while True:
                if last_val >= 0:  # angle < pi/2, will reach to that point later
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
                    percentage = ((ctp_i - p_astar_c) @ law_i) / ((p_astar_last - p_astar_c) @ law_i)
                    intersect_point = p_astar_c + direction * percentage
                    break
                last_val = val
                last_astar_id = astar_id
            return intersect_point

        def get_anchor_point_obs_surface(intersect_pt, control_pt):
            length = np.linalg.norm(intersect_pt - control_pt) + 1e-5
            anchor_pt_obs_surface = intersect_pt
            for a in np.arange(map_para.GRID_SIZE, length + map_para.GRID_SIZE - 1e-10, map_para.GRID_SIZE):
                check_pt = (1 - a / length) * intersect_pt + a / length * control_pt
                occ_flag = self.grid_map.check_occupancy(check_pt)
                if occ_flag:
                    percentage = (a - map_para.GRID_SIZE) / length
                    anchor_pt_obs_surface = (1 - percentage) * intersect_pt + percentage * control_pt
                    break
            return anchor_pt_obs_surface

        for ctps_index_seg_i, astar_seg_i in zip(collision_ctps_index_list, astar_list):
            for ctp_index in ctps_index_seg_i:
                # Convert to integer index and handle fractional indices
                is_int_index = isinstance(ctp_index, int)
                idx = int(ctp_index)

                # Calculate control point position
                if is_int_index:
                    ctp = ctps_opt[idx]
                else:
                    ctp = (ctps_opt[idx] + ctps_opt[idx + 1]) / 2

                # Calculate directional law vector more efficiently
                if idx == 0:
                    law = ctps_opt[1] - ctps_opt[0]
                elif idx == ctps_opt.shape[0] - 1:
                    law = ctps_opt[-1] - ctps_opt[-2]
                else:
                    if is_int_index:
                        law = ctps_opt[idx + 1] - ctps_opt[idx - 1]
                    else:
                        law = ctps_opt[idx + 1] - ctps_opt[idx]

                # Process intersection and direction
                intersection_point = find_intersection_point(astar_i=astar_seg_i, ctp_i=ctp, law_i=law)
                point_diff = intersection_point - ctp
                unit_dir = point_diff / (np.linalg.norm(point_diff) + 1e-5)

                # Get anchor point and store result
                anchor_point_obs_surface = get_anchor_point_obs_surface(
                    intersect_pt=intersection_point,
                    control_pt=ctp
                )

                # Store in dictionary using the integer index
                pv_pair_info[idx].append(np.vstack((anchor_point_obs_surface, unit_dir)))

    def check_bspline_collision(self, bspline):
        bspline_start_p = bspline.evaluate_pva(0.0)[:3]
        bspline_end_p = bspline.evaluate_pva(bspline.duration)[:3]
        wp_check_num = np.linalg.norm(bspline_start_p - bspline_end_p) / map_para.GRID_SIZE
        t_step = bspline.duration / wp_check_num
        occ_flag = False
        for t in np.arange(bspline.current_progress, 2 / 3 * bspline.duration + 1e-3, t_step):
            check_pt = bspline.evaluate_pva(t)[:3]
            occ_flag = self.grid_map.check_occupancy(check_pt)
            if occ_flag:
                break
        return occ_flag


def generate_new_bspline(x_start_pva, x_target_pva, start_time):
    ts_check = traj_para.CTRL_PT_DIST / traj_para.MAX_V * 1.2
    dist_to_target = np.linalg.norm(x_target_pva[:3] - x_start_pva[:3])

    # Calculate duration_max_a (max_a enough / not enough)
    critical_distance = (traj_para.MAX_V ** 2) / traj_para.MAX_A
    if dist_to_target <= critical_distance:
        duration_max_a = np.sqrt(dist_to_target / traj_para.MAX_A)
    else:
        duration_max_a = ((dist_to_target - critical_distance) / traj_para.MAX_V +
                          2 * traj_para.MAX_V / traj_para.MAX_A)

    # Generate one-segment trajectory and sample a point set.
    onesegment_traj = MinisnapTraj(waypoints=np.hstack((x_start_pva[:3], x_target_pva[:3])).T,
                                   ho_start=np.hstack((x_start_pva[3:6], x_start_pva[6:9])).T,
                                   ho_end=np.hstack((x_target_pva[3:6], x_target_pva[6:9])).T,
                                   s=3,
                                   time_idx=np.array([0, duration_max_a]))
    while True:
        point_set = []
        max_dist = -1.0
        flag_too_far = False
        for t in np.arange(0, duration_max_a, ts_check):
            pt = onesegment_traj.get_pos(t).reshape(-1, 1)
            point_set.append(pt)
            if len(point_set) >= 2:
                dist_cur = np.linalg.norm(point_set[-1] - point_set[-2])
                max_dist = max(max_dist, dist_cur)
            if max_dist >= traj_para.CTRL_PT_DIST * 1.5:
                flag_too_far = True
                break
        if not flag_too_far and len(point_set) >= 7:
            break
        else:
            ts_check /= 1.5
    del point_set[1]
    del point_set[-2]
    point_set_selected = np.hstack(point_set).T
    tip_derivatives_selected = np.vstack((x_start_pva[3:6].flatten(), x_target_pva[3:6].flatten(),
                                          x_start_pva[6:9].flatten(), x_target_pva[6:9].flatten()))
    new_bspline = UniformBspline()
    new_bspline.generate_unifrom_bspline_trajectory(start_time=start_time, ts=ts_check,
                                                    point_set=point_set_selected, tip_derivatives=tip_derivatives_selected)

    return new_bspline


def start_optimize(bspline, pv_pair_info, lambda_dist):
    param = {
        "ctps_opt": bspline.control_points.copy(),
        "pv_pair_info": pv_pair_info,
        "lambda_dist": lambda_dist,
        "t_ubs": bspline.ts
    }
    start_index = 3
    end_index = param["ctps_opt"].shape[0] - 3
    x0 = param["ctps_opt"][start_index:end_index].flatten()
    result = minimize(objective_gradient_rebound, x0, method='L-BFGS-B', jac=True, args=(param,), tol=traj_para.TOL)
    bspline.control_points[start_index:end_index] = result.x.reshape(-1, 3)

def start_refine(bspline, ratio):
    bspline_opt = copy.deepcopy(bspline)
    point_set_ref = []
    for t in np.arange(0, bspline_opt.duration + 1e-3, bspline_opt.ts):
        pt = bspline_opt.evaluate_pva(t)[:3]
        point_set_ref.append(pt)
    point_set_ref = np.hstack(point_set_ref).T

    bspline.duration *= ratio
    bspline.ts *= ratio
    param = {
        "ctps_opt": bspline.control_points.copy(),
        "t_ubs": bspline.ts,
        "ref_point": point_set_ref
    }

    start_index = 3
    end_index = param["ctps_opt"].shape[0] - 3
    x0 = param["ctps_opt"][start_index:end_index].flatten()
    result = minimize(objective_gradient_refine, x0, method='L-BFGS-B', jac=True, args=(param,), tol=traj_para.TOL)
    bspline.control_points[start_index:end_index] = result.x.reshape(-1, 3)

def objective_gradient_rebound(x, param):
    x.shape = (-1, 3)
    ctps_opt = param["ctps_opt"]
    pv_pair_info = param["pv_pair_info"]
    lambda_dist = param["lambda_dist"]
    x = np.vstack((ctps_opt[:3, :], x, ctps_opt[-3:, :]))
    qnum, qdim = x.shape
    delta_t = param["t_ubs"]

    # 1. Smoothness
    inv_dt3 = (1 / delta_t) ** 3
    inv_dt2 = (1 / delta_t) ** 2
    jerk_array = np.array([-1, 3, -3, 1])
    grad_smoothness = np.zeros_like(x)
    cost_smoothness = 0.0
    for i in range(qnum - 3):
        jerk_i = inv_dt3 * jerk_array @ x[i:i + 4, :]
        cost_smoothness += jerk_i @ jerk_i
        grad_smoothness[i:i + 4, :] += 2 * inv_dt3 * np.outer(jerk_array, jerk_i)

    # 2. Feasibility
    grad_feasibility = np.zeros_like(x)
    cost_feasibility = 0.0
    # 2.1: velocity feasibility
    for i in range(qnum - 1):
        vi = (x[i + 1] - x[i]) / delta_t
        for j in range(3):
            if vi[j] > traj_para.MAX_V:
                # multiply inv_dt2 to make vel & acc same scale
                cost_feasibility += (vi[j] - traj_para.MAX_V) ** 2 * inv_dt2
                grad_contrib = 2 * (vi[j] - traj_para.MAX_V) / delta_t * inv_dt2
            elif vi[j] < -traj_para.MAX_V:
                cost_feasibility += (vi[j] + traj_para.MAX_V) ** 2 * inv_dt2
                grad_contrib = 2 * (vi[j] + traj_para.MAX_V) / delta_t * inv_dt2
            else:
                grad_contrib = 0.0
            grad_feasibility[i + 1, j] += grad_contrib
            grad_feasibility[i, j] -= grad_contrib
    # 2.2: accel feasibility
    for i in range(qnum - 2):
        ai = (x[i] - 2 * x[i + 1] + x[i + 2]) * inv_dt2
        for j in range(3):
            if ai[j] > traj_para.MAX_A:
                cost_feasibility += (ai[j] - traj_para.MAX_A) ** 2
                grad_contrib = 2 * (ai[j] - traj_para.MAX_A) * inv_dt2
            elif ai[j] < - traj_para.MAX_A:
                cost_feasibility += (ai[j] + traj_para.MAX_A) ** 2
                grad_contrib = 2 * (ai[j] + traj_para.MAX_A) * inv_dt2
            else:
                grad_contrib = 0.0
            grad_feasibility[i, j] += grad_contrib
            grad_feasibility[i + 1, j] -= 2 * grad_contrib
            grad_feasibility[i + 2, j] += grad_contrib

    # 3. Distance
    grad_dist = np.zeros_like(x)
    cost_dist = 0.0
    d_safe = traj_para.SAFE_DIST
    for i in range(qnum):
        pv_pair_list_i = pv_pair_info[i]
        for pv_pair in pv_pair_list_i:
            anchor = pv_pair[0]
            unit_dir = pv_pair[1]
            dist = (x[i] - anchor) @ unit_dir
            c_dist = d_safe - dist
            if c_dist <= 0:
                # nothing happened
                continue
            elif 0 < c_dist <= d_safe:
                cost_dist += c_dist ** 3
                grad_dist[i] += -3 * c_dist ** 2 * unit_dir
            else:
                cost_dist += 3 * d_safe * c_dist ** 2 - 3 * d_safe ** 2 * c_dist + d_safe ** 3
                grad_dist[i] += -(6 * d_safe * c_dist - 3 * d_safe ** 2) * unit_dir

    cost = traj_para.LAMBDA_SMOOTH * cost_smoothness + traj_para.LAMBDA_FEASIBILITY * cost_feasibility + \
           lambda_dist * cost_dist
    grad = traj_para.LAMBDA_SMOOTH * grad_smoothness + traj_para.LAMBDA_FEASIBILITY * grad_feasibility + \
           lambda_dist * grad_dist
    grad = grad[3:-3, :]
    # print(f'cost_smooth: {traj_para.LAMBDA_SMOOTH * cost_smoothness} | '
    #       f'cost_feasibility: {traj_para.LAMBDA_FEASIBILITY * cost_feasibility} | '
    #       f'cost_dist: {lambda_dist * cost_dist} | '
    #       f'cost_total: {cost}')
    return cost, grad.flatten()

def objective_gradient_refine(x, param):
    x.shape = (-1, 3)
    ctps_opt = param["ctps_opt"]
    x = np.vstack((ctps_opt[:3, :], x, ctps_opt[-3:, :]))
    qnum, qdim = x.shape
    delta_t = param["t_ubs"]
    ref_point = param["ref_point"]

    # 1. Smoothness
    inv_dt3 = (1 / delta_t) ** 3
    inv_dt2 = (1 / delta_t) ** 2
    jerk_array = np.array([-1, 3, -3, 1])
    grad_smoothness = np.zeros_like(x)
    cost_smoothness = 0.0
    for i in range(qnum - 3):
        jerk_i = inv_dt3 * jerk_array @ x[i:i + 4, :]
        cost_smoothness += jerk_i @ jerk_i
        grad_smoothness[i:i + 4, :] += 2 * inv_dt3 * np.outer(jerk_array, jerk_i)

    # 2. Feasibility
    grad_feasibility = np.zeros_like(x)
    cost_feasibility = 0.0
    # 2.1: velocity feasibility
    for i in range(qnum - 1):
        vi = (x[i + 1] - x[i]) / delta_t
        for j in range(3):
            if vi[j] > traj_para.MAX_V:
                # multiply inv_dt2 to make vel & acc same scale
                cost_feasibility += (vi[j] - traj_para.MAX_V) ** 2 * inv_dt2
                grad_contrib = 2 * (vi[j] - traj_para.MAX_V) / delta_t * inv_dt2
            elif vi[j] < -traj_para.MAX_V:
                cost_feasibility += (vi[j] + traj_para.MAX_V) ** 2 * inv_dt2
                grad_contrib = 2 * (vi[j] + traj_para.MAX_V) / delta_t * inv_dt2
            else:
                grad_contrib = 0.0
            grad_feasibility[i + 1, j] += grad_contrib
            grad_feasibility[i, j] -= grad_contrib
    # 2.2: accel feasibility
    for i in range(qnum - 2):
        ai = (x[i] - 2 * x[i + 1] + x[i + 2]) * inv_dt2
        for j in range(3):
            if ai[j] > traj_para.MAX_A:
                cost_feasibility += (ai[j] - traj_para.MAX_A) ** 2
                grad_contrib = 2 * (ai[j] - traj_para.MAX_A) * inv_dt2
            elif ai[j] < - traj_para.MAX_A:
                cost_feasibility += (ai[j] + traj_para.MAX_A) ** 2
                grad_contrib = 2 * (ai[j] + traj_para.MAX_A) * inv_dt2
            else:
                grad_contrib = 0.0
            grad_feasibility[i, j] += grad_contrib
            grad_feasibility[i + 1, j] -= 2 * grad_contrib
            grad_feasibility[i + 2, j] += grad_contrib

    # 3. fitness
    grad_fit = np.zeros_like(x)
    cost_fit = 0.0
    para_a = 5.0
    para_b = 1.0
    for i in range(2, qnum - 2):
        xpt = 1/6 * (x[i - 1] + 4 * x[i] + x[i + 1]) - ref_point[i - 1]
        v = ref_point[i] - ref_point[i - 2]
        v_dir = v / (np.linalg.norm(v) + 1e-5)
        x_dot_v = xpt @ v_dir
        x_cross_v = np.cross(xpt, v_dir)
        cost_fit += (x_dot_v / para_a) ** 2 + (np.linalg.norm(x_cross_v) / para_b) ** 2

        # gradient w.r.t. xi
        #   df/dxi = 2*(x·v)/a2 * v  +  2/b2 * (v × (xi × v))
        df_dxi = (2 * x_dot_v / para_a**2) * v_dir \
                 + (2.0 / para_b**2) * np.cross(v_dir, x_cross_v)

        # chain‐rule back into the three control‐points
        grad_fit[i - 1] += df_dxi * (1.0 / 6.0)
        grad_fit[i] += df_dxi * (4.0 / 6.0)
        grad_fit[i + 1] += df_dxi * (1.0 / 6.0)

    cost = traj_para.LAMBDA_SMOOTH * cost_smoothness + traj_para.LAMBDA_FEASIBILITY * cost_feasibility + \
           traj_para.LAMBDA_FITNESS * cost_fit
    grad = traj_para.LAMBDA_SMOOTH * grad_smoothness + traj_para.LAMBDA_FEASIBILITY * grad_feasibility + \
           traj_para.LAMBDA_FITNESS * grad_fit
    grad = grad[3:-3, :]

    # print(f'cost_smooth: {traj_para.LAMBDA_SMOOTH * cost_smoothness} | '
    #       f'cost_feasibility: {traj_para.LAMBDA_FEASIBILITY * cost_feasibility} | '
    #       f'cost_fitness: {traj_para.LAMBDA_FITNESS * cost_fit} | '
    #       f'cost_total: {cost}')

    return cost, grad.flatten()

def reparam_bspline(bspline):
    point_set = []
    for t in np.arange(0, bspline.duration + 1e-3, bspline.ts):
        pt = bspline.evaluate_pva(t)[:3]
        point_set.append(pt)
    del point_set[1]
    del point_set[-2]
    point_set = np.hstack(point_set).T

    traj_new = UniformBspline()
    traj_new.generate_unifrom_bspline_trajectory(
        start_time=bspline.start_time,
        ts=bspline.ts,
        point_set=point_set,
        tip_derivatives=bspline.tip_derivatives)
    return traj_new
