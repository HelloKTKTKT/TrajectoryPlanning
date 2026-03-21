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
        map_size=np.array([30, 12, 5], dtype=np.int64),
        resolution=1.0,
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )


def make_bspline(control_points: np.ndarray) -> UniformBSpline:
    return UniformBSpline(
        control_points=np.asarray(control_points, dtype=np.float64),
        delta_t=0.5,
    )


def segment_pairs(segments) -> list[tuple[int, int]]:
    return [(seg.start_index, seg.end_index) for seg in segments]


def run_case(
    case_name: str,
    control_points: np.ndarray,
    obstacle_blocks: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    expected: list[tuple[int, int]] | None = None,
) -> None:
    planner = make_planner()
    grid_map = make_grid_map()

    for obs_start, obs_size in obstacle_blocks:
        grid_map.add_obstacle(
            obs_start_index=np.array(obs_start, dtype=np.int64),
            obs_size=np.array(obs_size, dtype=np.int64),
        )

    bspline = make_bspline(control_points)
    segments = planner._get_collision_segments(grid_map=grid_map, bspline=bspline)
    actual = segment_pairs(segments)

    print("=" * 100)
    print(f"Case: {case_name}")
    print(f"Control point count : {control_points.shape[0]}")
    print(f"Obstacle blocks     : {obstacle_blocks}")
    print(f"Collision segments  : {actual}")

    if expected is not None:
        print(f"Expected            : {expected}")
        print("PASS" if actual == expected else "FAIL")


def main() -> None:
    # All cases use y=4.5, z=0.5 so they pass through grid row (., 4, 0).
    # Reminder:
    # _get_collision_segments only checks links i = 3 ... N-4
    # so the cases are built to place collisions in the middle range.

    # ------------------------------------------------------------------
    # Case 1: no collision
    # Expected: []
    # ------------------------------------------------------------------
    ctps_no_collision = np.array(
        [
            [0.5, 4.5, 0.5],
            [1.5, 4.5, 0.5],
            [2.5, 4.5, 0.5],
            [3.5, 4.5, 0.5],
            [4.5, 4.5, 0.5],
            [5.5, 4.5, 0.5],
            [6.5, 4.5, 0.5],
            [7.5, 4.5, 0.5],
            [8.5, 4.5, 0.5],
            [9.5, 4.5, 0.5],
        ],
        dtype=np.float64,
    )
    run_case(
        case_name="no_collision",
        control_points=ctps_no_collision,
        obstacle_blocks=[],
        expected=[],
    )

    # ------------------------------------------------------------------
    # Case 2: one control point is inside obstacle
    # Obstacle cell is x in [4,5), y in [4,5), z in [0,1)
    # cp[4] = [4.5, 4.5, 0.5] lies inside
    # Expected: one collision segment
    # ------------------------------------------------------------------
    ctps_cp_inside = np.array(
        [
            [0.5, 4.5, 0.5],
            [1.5, 4.5, 0.5],
            [2.5, 4.5, 0.5],
            [3.5, 4.5, 0.5],
            [4.5, 4.5, 0.5],  # inside obstacle
            [5.5, 4.5, 0.5],
            [6.5, 4.5, 0.5],
            [7.5, 4.5, 0.5],
            [8.5, 4.5, 0.5],
            [9.5, 4.5, 0.5],
        ],
        dtype=np.float64,
    )
    run_case(
        case_name="control_point_inside_obstacle",
        control_points=ctps_cp_inside,
        obstacle_blocks=[((4, 4, 0), (1, 1, 1))],
        expected=[(3, 6)],
    )

    # ------------------------------------------------------------------
    # Case 3: both endpoints are free, but the segment between them crosses obstacle
    # cp[4] = 3.5 and cp[5] = 5.5 are both free, but the link crosses voxel x=4
    # Expected: one collision segment
    # ------------------------------------------------------------------
    ctps_cross_only = np.array(
        [
            [0.5, 4.5, 0.5],
            [1.5, 4.5, 0.5],
            [2.5, 4.5, 0.5],
            [3.0, 4.5, 0.5],
            [3.5, 4.5, 0.5],  # free
            [5.5, 4.5, 0.5],  # free
            [6.5, 4.5, 0.5],
            [7.5, 4.5, 0.5],
            [8.5, 4.5, 0.5],
            [9.5, 4.5, 0.5],
        ],
        dtype=np.float64,
    )
    run_case(
        case_name="segment_crosses_obstacle_but_endpoints_free",
        control_points=ctps_cross_only,
        obstacle_blocks=[((4, 4, 0), (1, 1, 1))],
        expected=[(4, 6)],
    )

    # ------------------------------------------------------------------
    # Case 4: two separated obstacle crossings
    # Expected: two separate collision segments, no merge
    # ------------------------------------------------------------------
    ctps_two_separated = np.array(
        [
            [0.5, 4.5, 0.5],
            [1.5, 4.5, 0.5],
            [2.5, 4.5, 0.5],
            [3.0, 4.5, 0.5],
            [3.5, 4.5, 0.5],  # free
            [5.5, 4.5, 0.5],  # free, crosses obstacle x=4
            [6.5, 4.5, 0.5],  # free
            [8.5, 4.5, 0.5],  # free, crosses obstacle x=7
            [9.5, 4.5, 0.5],
            [10.5, 4.5, 0.5],
            [11.5, 4.5, 0.5],
        ],
        dtype=np.float64,
    )
    run_case(
        case_name="two_separated_collision_segments",
        control_points=ctps_two_separated,
        obstacle_blocks=[
            ((4, 4, 0), (1, 1, 1)),
            ((7, 4, 0), (1, 1, 1)),
        ],
        expected=[(4, 6), (6, 8)],
    )

    # ------------------------------------------------------------------
    # Case 5: one longer continuous collision region over multiple links
    # Obstacle spans x=4 and x=5, so consecutive links remain occupied
    # Expected: one merged / continuous collision segment
    # ------------------------------------------------------------------
    ctps_continuous = np.array(
        [
            [0.5, 4.5, 0.5],
            [1.5, 4.5, 0.5],
            [2.5, 4.5, 0.5],
            [3.0, 4.5, 0.5],
            [3.5, 4.5, 0.5],
            [4.5, 4.5, 0.5],
            [6.5, 4.5, 0.5],
            [7.5, 4.5, 0.5],
            [8.5, 4.5, 0.5],
            [9.5, 4.5, 0.5],
        ],
        dtype=np.float64,
    )
    run_case(
        case_name="continuous_collision_region",
        control_points=ctps_continuous,
        obstacle_blocks=[((4, 4, 0), (2, 1, 1))],
        expected=[(4, 7)],
    )


if __name__ == "__main__":
    main()
