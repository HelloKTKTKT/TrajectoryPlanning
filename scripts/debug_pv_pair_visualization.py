from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from trajplan.map.grid_map import GridMap
from trajplan.planning.local_planner import LocalPlanner, PvPair
from trajplan.trajectory.bspline import UniformBSpline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def make_planner() -> LocalPlanner:
    return LocalPlanner(
        ctrl_pt_dist=0.4,
        planning_horizon=5.0,
        max_v=4.0,
        max_a=5.0,
    )


def make_grid_map() -> GridMap:
    return GridMap(
        map_size=np.array([50, 50, 30], dtype=np.int64),
        resolution=0.2,
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )


def make_bspline(control_points: np.ndarray) -> UniformBSpline:
    return UniformBSpline(
        control_points=np.asarray(control_points, dtype=np.float64),
        delta_t=0.5,
    )


def segment_tuples(segments) -> list[tuple[int, int]]:
    return [(seg.start_index, seg.end_index) for seg in segments]


def build_cases() -> list[
    tuple[str, np.ndarray, list[tuple[tuple[int, int, int], tuple[int, int, int]]]]
]:
    """
    Build several moderate pv-pair visualization cases.

    Notes
    -----
    - Grid-map resolution and map size stay fixed.
    - Control points are arranged like short natural trajectory segments.
    - Neighboring control points are typically spaced around 0.2 to 0.5 m.
    """
    cases: list[
        tuple[
            str,
            np.ndarray,
            list[tuple[tuple[int, int, int], tuple[int, int, int]]],
        ]
    ] = []

    # ------------------------------------------------------------------
    # Case 1: straight passage along y with one box obstacle.
    # ------------------------------------------------------------------
    control_points_1 = np.array(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.4, 1.0],
            [1.0, 0.8, 1.0],
            [1.05, 1.2, 1.0],
            [1.1, 1.6, 1.0],
            [1.15, 2.0, 1.0],
            [1.18, 2.4, 1.0],
            [1.22, 2.8, 1.0],
            [1.29, 3.2, 1.0],
            [1.4, 3.6, 1.0],
            [1.4, 4.0, 1.0],
            [1.6, 4.4, 1.0],
            [1.7, 4.8, 1.0],
            [1.75, 5.2, 1.0],
            [1.75, 5.6, 1.0],
            [1.75, 6.0, 1.0],
            [1.75, 6.4, 1.0],
            [1.75, 6.8, 1.0],
            [1.75, 7.2, 1.0],
            [1.75, 7.6, 1.0],
            [1.75, 8.0, 1.0],
        ],
        dtype=np.float64,
    )
    obstacle_blocks_1 = [((0, 15, 0), (10, 10, 10))]
    cases.append(("straight_single_box", control_points_1, obstacle_blocks_1))

    return cases


def print_debug_info(
    control_points: np.ndarray,
    collision_segments,
    collision_ctps_index_list,
    astar_list,
    pv_pair_info,
) -> None:
    print("=" * 100)
    print("PV Pair Debug")
    print("control_points:")
    print(control_points)
    print(f"collision_segments       : {segment_tuples(collision_segments)}")
    print(f"collision_ctps_index_list: {collision_ctps_index_list}")

    for k, path in enumerate(astar_list):
        print(f"\nastar_list[{k}] shape    : {path.shape}")
        print(f"astar_list[{k}] start    : {path[0]}")
        print(f"astar_list[{k}] end      : {path[-1]}")
        print(f"astar_list[{k}] path     :\n{path}")

    print("\npv_pair_info:")
    nonempty_count = 0
    for idx, pair_list in enumerate(pv_pair_info):
        if len(pair_list) == 0:
            continue
        nonempty_count += 1
        print(f"  control point index {idx}: {len(pair_list)} pair(s)")
        for j, pair in enumerate(pair_list):
            if idx == 0:
                law = control_points[1] - control_points[0]
            elif idx == control_points.shape[0] - 1:
                law = control_points[-1] - control_points[-2]
            else:
                law = control_points[idx + 1] - control_points[idx - 1]

            law_unit = law / (np.linalg.norm(law) + 1e-9)
            dot_val = float(np.dot(law_unit, pair.unit_dir))

            print(
                f"    pair[{j}] surface_anchor = {pair.surface_anchor}, "
                f"unit_dir = {pair.unit_dir}"
            )
            print(
                f"             law = {law}, "
                f"law_unit = {law_unit}, "
                f"dot(law_unit, unit_dir) = {dot_val}"
            )
    if nonempty_count == 0:
        print("  all empty")
    print("=" * 100)


def collect_obstacle_voxel_centers(
    grid_map: GridMap,
    map_size: np.ndarray,
) -> np.ndarray:
    pts = []
    for ix in range(int(map_size[0])):
        for iy in range(int(map_size[1])):
            for iz in range(int(map_size[2])):
                world = grid_map.grid_to_world(np.array([ix, iy, iz], dtype=np.int64))
                if grid_map.check_occupancy(world):
                    pts.append(world)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def visualize_case(
    grid_map: GridMap,
    map_size: np.ndarray,
    control_points: np.ndarray,
    collision_segments,
    collision_ctps_index_list,
    astar_list,
    pv_pair_info,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --------------------------------------------------------------
    # 1. obstacles
    # --------------------------------------------------------------
    obstacle_pts = collect_obstacle_voxel_centers(grid_map=grid_map, map_size=map_size)
    if obstacle_pts.shape[0] > 0:
        ax.scatter(
            obstacle_pts[:, 0],
            obstacle_pts[:, 1],
            obstacle_pts[:, 2],
            marker="s",
            s=60,
            alpha=0.25,
            label="occupied voxel centers",
        )

    # --------------------------------------------------------------
    # 2. control points
    # --------------------------------------------------------------
    ax.plot(
        control_points[:, 0],
        control_points[:, 1],
        control_points[:, 2],
        "-o",
        linewidth=2,
        markersize=5,
        label="control points",
    )

    for i, p in enumerate(control_points):
        ax.text(p[0], p[1], p[2], f"{i}", fontsize=8)

    # --------------------------------------------------------------
    # 3. collision segments
    # --------------------------------------------------------------
    for seg_id, seg in enumerate(collision_segments):
        seg_pts = control_points[seg.start_index : seg.end_index + 1]
        ax.plot(
            seg_pts[:, 0],
            seg_pts[:, 1],
            seg_pts[:, 2],
            linewidth=4,
            label=f"collision segment {seg_id}" if seg_id == 0 else None,
        )

    # --------------------------------------------------------------
    # 4. A* paths
    # --------------------------------------------------------------
    for k, path in enumerate(astar_list):
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            "--",
            linewidth=2,
            label=f"A* path {k}",
        )

    # --------------------------------------------------------------
    # 5. collision ctp points
    # --------------------------------------------------------------
    plotted_collision_ctp_label = False
    for ctps_index_seg_i in collision_ctps_index_list:
        for ctp_index in ctps_index_seg_i:
            ctp_index_float = float(ctp_index)
            idx = int(ctp_index_float)
            if ctp_index_float.is_integer():
                ctp = control_points[idx]
            else:
                ctp = 0.5 * (control_points[idx] + control_points[idx + 1])

            ax.scatter(
                [ctp[0]],
                [ctp[1]],
                [ctp[2]],
                s=80,
                marker="x",
                label="collision ctp" if not plotted_collision_ctp_label else None,
            )
            plotted_collision_ctp_label = True

    # --------------------------------------------------------------
    # 6. pv pairs
    # --------------------------------------------------------------
    plotted_anchor_label = False
    plotted_dir_label = False

    for idx, pair_list in enumerate(pv_pair_info):
        if len(pair_list) == 0:
            continue

        ctp = control_points[idx]

        for pair in pair_list:
            anchor = pair.surface_anchor
            unit_dir = pair.unit_dir

            # anchor point
            ax.scatter(
                [anchor[0]],
                [anchor[1]],
                [anchor[2]],
                s=60,
                marker="^",
                label="surface anchor" if not plotted_anchor_label else None,
            )
            plotted_anchor_label = True

            # line from control point to anchor
            line_pts = np.vstack((ctp, anchor))
            ax.plot(
                line_pts[:, 0],
                line_pts[:, 1],
                line_pts[:, 2],
                ":",
                linewidth=2,
            )

            # unit direction arrow starting from anchor
            scale = 1.0
            ax.quiver(
                anchor[0],
                anchor[1],
                anchor[2],
                unit_dir[0],
                unit_dir[1],
                unit_dir[2],
                length=scale,
                normalize=True,
                label="unit dir" if not plotted_dir_label else None,
            )
            plotted_dir_label = True

    ax.set_title("PV Pair Debug Visualization")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    planner = make_planner()
    # Keep visualization bounds consistent with make_grid_map().
    map_size = np.array([50, 50, 30], dtype=np.int64)

    for case_name, control_points, obstacle_blocks in build_cases():
        print("\n" + "#" * 120)
        print(f"Case: {case_name}")
        print("#" * 120)

        grid_map = make_grid_map()

        for obs_start, obs_size in obstacle_blocks:
            grid_map.add_obstacle(
                obs_start_index=np.array(obs_start, dtype=np.int64),
                obs_size=np.array(obs_size, dtype=np.int64),
            )

        bspline = make_bspline(control_points)

        collision_segments = planner._get_collision_segments(
            grid_map=grid_map,
            bspline=bspline,
        )

        collision_ctps_index_list, astar_list = (
            planner._get_astar_paths_and_collision_ctps(
                grid_map=grid_map,
                bspline=bspline,
                collision_segments=collision_segments,
            )
        )

        pv_pair_info: list[list[PvPair]] = [[] for _ in range(control_points.shape[0])]
        planner._update_pv_pair_info(
            collision_ctps_index_list=collision_ctps_index_list,
            astar_list=astar_list,
            pv_pair_info=pv_pair_info,
            control_points=control_points,
            grid_map=grid_map,
        )

        print_debug_info(
            control_points=control_points,
            collision_segments=collision_segments,
            collision_ctps_index_list=collision_ctps_index_list,
            astar_list=astar_list,
            pv_pair_info=pv_pair_info,
        )

        visualize_case(
            grid_map=grid_map,
            map_size=map_size,
            control_points=control_points,
            collision_segments=collision_segments,
            collision_ctps_index_list=collision_ctps_index_list,
            astar_list=astar_list,
            pv_pair_info=pv_pair_info,
        )


if __name__ == "__main__":
    main()
