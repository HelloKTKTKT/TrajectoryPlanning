from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from trajplan.map.grid_map import GridMap
from trajplan.planning.local_planner import LocalPlanner
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
        map_size=np.array([40, 20, 5], dtype=np.int64),
        resolution=1.0,
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )


def make_bspline(control_points: np.ndarray) -> UniformBSpline:
    return UniformBSpline(
        control_points=np.asarray(control_points, dtype=np.float64),
        delta_t=0.5,
    )


def segments_to_tuples(segments) -> list[tuple[int, int]]:
    return [(seg.start_index, seg.end_index) for seg in segments]


def print_astar_info(astar_list: list[np.ndarray]) -> None:
    if len(astar_list) == 0:
        print("astar_list               : []")
        return

    for k, path in enumerate(astar_list):
        print(f"astar_list[{k}] shape    : {path.shape}")
        print(f"astar_list[{k}] start    : {path[0]}")
        print(f"astar_list[{k}] end      : {path[-1]}")
        if path.shape[0] >= 2:
            seg_lens = np.linalg.norm(path[1:] - path[:-1], axis=1)
            print(f"astar_list[{k}] length   : {np.sum(seg_lens):.4f}")
        else:
            print(f"astar_list[{k}] length   : 0.0000")
        print(f"astar_list[{k}] path     :\n{path}")


def run_case(
    case_name: str,
    control_points: np.ndarray,
    obstacle_blocks: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> None:
    planner = make_planner()
    grid_map = make_grid_map()

    for obs_start, obs_size in obstacle_blocks:
        grid_map.add_obstacle(
            obs_start_index=np.array(obs_start, dtype=np.int64),
            obs_size=np.array(obs_size, dtype=np.int64),
        )

    bspline = make_bspline(control_points)

    print("=" * 100)
    print(f"Case: {case_name}")
    print(f"control point count      : {control_points.shape[0]}")
    print(f"obstacle blocks          : {obstacle_blocks}")
    print("control points:")
    print(control_points)

    try:
        collision_segments = planner._get_collision_segments(
            grid_map=grid_map,
            bspline=bspline,
        )
        collision_segment_tuples = segments_to_tuples(collision_segments)

        print(f"collision_segments       : {collision_segment_tuples}")

        collision_ctps_index_list, astar_list = (
            planner._get_astar_paths_and_collision_ctps(
                grid_map=grid_map,
                bspline=bspline,
                collision_segments=collision_segments,
            )
        )

        print(f"collision_ctps_index_list: {collision_ctps_index_list}")
        print_astar_info(astar_list)

    except Exception as e:
        print(f"EXCEPTION                : {type(e).__name__}: {e}")


def main() -> None:
    # y = 6.5, z = 0.5 for all cases unless stated otherwise.
    # order = 3, so only links i = 3 ... N-4 are checked.

    # ------------------------------------------------------------------
    # Case 1
    # No obstacle, so:
    # - collision_segments should be empty
    # - astar_list should be empty
    # ------------------------------------------------------------------
    ctps_no_collision = np.array(
        [
            [0.5, 6.5, 0.5],
            [1.5, 6.5, 0.5],
            [2.5, 6.5, 0.5],
            [3.5, 6.5, 0.5],
            [4.5, 6.5, 0.5],
            [5.5, 6.5, 0.5],
            [6.5, 6.5, 0.5],
            [7.5, 6.5, 0.5],
            [8.5, 6.5, 0.5],
            [9.5, 6.5, 0.5],
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Case 2
    # One simple crossing segment.
    # cp[4] -> cp[5] crosses obstacle at x=5.
    # Expected:
    # - one collision segment
    # - one A* path
    # - internal collision_ctps_index_list should contain the inner indices
    # ------------------------------------------------------------------
    ctps_one_segment = np.array(
        [
            [0.5, 6.5, 0.5],
            [1.5, 6.5, 0.5],
            [2.5, 6.5, 0.5],
            [3.5, 6.5, 0.5],
            [4.5, 6.5, 0.5],
            [6.5, 6.5, 0.5],
            [7.5, 6.5, 0.5],
            [8.5, 6.5, 0.5],
            [9.5, 6.5, 0.5],
            [10.5, 6.5, 0.5],
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Case 3
    # Two separated collision segments.
    # This checks:
    # - multiple A* calls
    # - multiple astar paths
    # - multiple collision_ctps_index sublists
    # This layout is chosen so that links 4->5 and 7->8 each cross one obstacle, and there are enough trailing control points so both segments can produce enter and exit events without being truncated by the checked range.
    # ------------------------------------------------------------------
    ctps_two_segments = np.array(
        [
            [0.5, 6.5, 0.5],
            [1.5, 6.5, 0.5],
            [2.5, 6.5, 0.5],
            [3.5, 6.5, 0.5],
            [4.5, 6.5, 0.5],
            [6.5, 6.5, 0.5],
            [7.5, 6.5, 0.5],
            [8.5, 6.5, 0.5],
            [10.5, 6.5, 0.5],
            [11.5, 6.5, 0.5],
            [12.5, 6.5, 0.5],
            [13.5, 6.5, 0.5],
            [14.5, 6.5, 0.5],
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Case 4
    # This layout is chosen so that all points lie on the line x=1, z=1, while y increases.
    # The obstacle occupies x in [0, 2), y in [3, 5), z in [0, 2).
    # The long jump from y=2.4 to y=7.0 should make exactly one checked link cross
    # the obstacle, and the next link should immediately exit, producing a segment
    # with end-start == 1 and triggering the half-index logic.
    # ------------------------------------------------------------------
    ctps_half_index = np.array(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.4, 1.0],
            [1.0, 0.8, 1.0],
            [1.0, 1.2, 1.0],
            [1.0, 1.6, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 2.4, 1.0],
            [1.0, 7.0, 1.0],
            [1.0, 7.5, 1.0],
            [1.0, 8.0, 1.0],
            [1.0, 8.5, 1.0],
            [1.0, 9.0, 1.0],
            [1.0, 9.5, 1.0],
            [1.0, 10.0, 1.0],
            [1.0, 10.5, 1.0],
            [1.0, 11.0, 1.0],
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Case 5
    # First collision segment starts at index 3, and ctp[3] is inside obstacle.
    # This is designed to trigger your special handling:
    # search backward 2 -> 1 -> 0 for a free A* start point.
    # ------------------------------------------------------------------
    ctps_first_seg_start_inside = np.array(
        [
            [0.5, 6.5, 0.5],  # free
            [1.5, 6.5, 0.5],  # free
            [2.5, 6.5, 0.5],  # free
            [3.5, 6.5, 0.5],  # occupied
            [4.5, 6.5, 0.5],  # occupied
            [6.5, 6.5, 0.5],  # free
            [7.5, 6.5, 0.5],
            [8.5, 6.5, 0.5],
            [9.5, 6.5, 0.5],
            [10.5, 6.5, 0.5],
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Case 6
    # Fail case for the special handling:
    # ctp[3], ctp[2], ctp[1], ctp[0] all inside obstacle.
    # Expected: RuntimeError in _get_astar_paths_and_collision_ctps
    # ------------------------------------------------------------------
    ctps_first_seg_all_backward_occupied = np.array(
        [
            [0.5, 6.5, 0.5],  # occupied
            [1.5, 6.5, 0.5],  # occupied
            [2.5, 6.5, 0.5],  # occupied
            [3.5, 6.5, 0.5],  # occupied
            [4.5, 6.5, 0.5],  # occupied
            [6.5, 6.5, 0.5],  # free
            [7.5, 6.5, 0.5],
            [8.5, 6.5, 0.5],
            [9.5, 6.5, 0.5],
            [10.5, 6.5, 0.5],
        ],
        dtype=np.float64,
    )

    cases = [
        (
            "no_collision",
            ctps_no_collision,
            [],
        ),
        (
            "one_collision_segment",
            ctps_one_segment,
            [((5, 6, 0), (1, 1, 1))],
        ),
        (
            "two_separated_collision_segments",
            ctps_two_segments,
            [
                ((5, 6, 0), (1, 1, 1)),
                ((9, 6, 0), (1, 1, 1)),
            ],
        ),
        (
            "half_index_case",
            ctps_half_index,
            [((0, 3, 0), (2, 2, 2))],
        ),
        (
            "first_segment_start_index_3_inside_obstacle",
            ctps_first_seg_start_inside,
            [((3, 6, 0), (2, 1, 1))],
        ),
        (
            "first_segment_backward_search_fails",
            ctps_first_seg_all_backward_occupied,
            [((0, 6, 0), (5, 1, 1))],
        ),
    ]

    for case_name, ctps, obstacle_blocks in cases:
        run_case(
            case_name=case_name,
            control_points=ctps,
            obstacle_blocks=obstacle_blocks,
        )


if __name__ == "__main__":
    main()
