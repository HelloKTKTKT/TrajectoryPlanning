from __future__ import annotations
import time
import numpy as np
from trajplan.map.grid_map import GridMap  # noqa: E402


RNG_SEED = 20260317
NUM_RANDOM_SCENARIOS = 100
MAP_SIZE = np.array([40, 40, 10], dtype=np.int64)
RESOLUTION = 0.2
ORIGIN = np.zeros(3, dtype=np.float64)


def sample_free_index(grid_map: GridMap, rng: np.random.Generator) -> np.ndarray:
    """Sample one free grid index uniformly at random."""
    while True:
        idx = np.array(
            [
                rng.integers(0, grid_map.nx),
                rng.integers(0, grid_map.ny),
                rng.integers(0, grid_map.nz),
            ],
            dtype=np.int64,
        )
        if not grid_map.is_occupied_index(idx):
            return idx


def add_random_obstacles(
    grid_map: GridMap, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Add randomly generated axis-aligned block obstacles."""
    obstacle_specs: list[tuple[np.ndarray, np.ndarray]] = []

    obstacle_count = int(rng.integers(8, 26))

    for _ in range(obstacle_count):
        size = np.array(
            [
                rng.integers(1, 7),
                rng.integers(1, 7),
                rng.integers(1, 4),
            ],
            dtype=np.int64,
        )

        start = np.array(
            [
                rng.integers(0, grid_map.nx),
                rng.integers(0, grid_map.ny),
                rng.integers(0, grid_map.nz),
            ],
            dtype=np.int64,
        )

        grid_map.add_obstacle(start, size)
        obstacle_specs.append((start, size))

    return obstacle_specs


def run_random_scenarios() -> None:
    rng = np.random.default_rng(RNG_SEED)

    success_count = 0
    failure_count = 0
    explored_ratios: list[float] = []
    run_times_ms: list[float] = []

    print("=" * 100)
    print("Random GridMap A* debug test")
    print(f"seed = {RNG_SEED}")
    print(f"num_scenarios = {NUM_RANDOM_SCENARIOS}")
    print(f"map_size = {MAP_SIZE.tolist()}, total_voxels = {int(np.prod(MAP_SIZE))}")
    print(f"resolution = {RESOLUTION}")
    print("=" * 100)

    for scenario_id in range(1, NUM_RANDOM_SCENARIOS + 1):
        grid_map = GridMap(map_size=MAP_SIZE, resolution=RESOLUTION, origin=ORIGIN)
        obstacle_specs = add_random_obstacles(grid_map, rng)

        start_idx = sample_free_index(grid_map, rng)
        target_idx = sample_free_index(grid_map, rng)
        while np.array_equal(start_idx, target_idx):
            target_idx = sample_free_index(grid_map, rng)

        grid_map.update_start_target_index(start_idx, target_idx)

        t0 = time.perf_counter()
        success = grid_map.astar_search()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        explored = int(np.count_nonzero(grid_map.visited))
        total = int(grid_map.N)
        ratio = explored / total

        run_times_ms.append(elapsed_ms)
        explored_ratios.append(ratio)

        if success:
            success_count += 1
            path = grid_map.retrieve_optimal_path()
            path_len = len(path)
            status = "SUCCESS"
        else:
            failure_count += 1
            path_len = 0
            status = "FAIL"

        print(
            f"[random {scenario_id:03d}] {status} | "
            f"obstacles={len(obstacle_specs):02d} | "
            f"start={start_idx.tolist()} | target={target_idx.tolist()} | "
            f"explored={explored}/{total} ({ratio:.4f}) | "
            f"time={elapsed_ms:.3f} ms | path_len={path_len}"
        )

    print("-" * 100)
    print("Random scenario summary")
    print(f"success_count = {success_count}")
    print(f"failure_count = {failure_count}")
    print(f"success_rate = {success_count / NUM_RANDOM_SCENARIOS:.4f}")
    print(
        "explored_ratio "
        f"min={min(explored_ratios):.4f}, "
        f"mean={float(np.mean(explored_ratios)):.4f}, "
        f"max={max(explored_ratios):.4f}"
    )
    print(
        "time_ms "
        f"min={min(run_times_ms):.3f}, "
        f"mean={float(np.mean(run_times_ms)):.3f}, "
        f"max={max(run_times_ms):.3f}"
    )
    print(
        "Note: random scenarios only guarantee start/target are inside the map and free. "
        "They do not guarantee that a valid path exists."
    )
    print("=" * 100)


def run_invalid_start_target_scenarios() -> None:
    print("Special invalid start/target scenarios")
    print("=" * 100)

    scenarios = [
        {
            "name": "start in obstacle, target free",
            "start": np.array([3, 3, 1], dtype=np.int64),
            "target": np.array([10, 10, 2], dtype=np.int64),
        },
        {
            "name": "start free, target in obstacle",
            "start": np.array([1, 1, 1], dtype=np.int64),
            "target": np.array([4, 4, 1], dtype=np.int64),
        },
        {
            "name": "start and target both in obstacle",
            "start": np.array([2, 2, 1], dtype=np.int64),
            "target": np.array([5, 5, 1], dtype=np.int64),
        },
    ]

    for case_id, case in enumerate(scenarios, start=1):
        grid_map = GridMap(
            map_size=np.array([20, 20, 6], dtype=np.int64),
            resolution=RESOLUTION,
            origin=ORIGIN,
        )

        # Add one block obstacle that contains all obstacle-designated indices above.
        grid_map.add_obstacle(
            obs_start_index=np.array([2, 2, 1], dtype=np.int64),
            obs_size=np.array([4, 4, 2], dtype=np.int64),
        )

        start_idx = case["start"]
        target_idx = case["target"]

        print(f"[invalid {case_id}] {case['name']}")
        print(
            f"  start  = {start_idx.tolist()}, occupied = {grid_map.is_occupied_index(start_idx)}"
        )
        print(
            f"  target = {target_idx.tolist()}, occupied = {grid_map.is_occupied_index(target_idx)}"
        )

        try:
            grid_map.update_start_target_index(start_idx, target_idx)
            t0 = time.perf_counter()
            success = grid_map.astar_search()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            explored = int(np.count_nonzero(grid_map.visited))
            print(
                f"  astar_search returned {success} | explored={explored}/{grid_map.N} "
                f"| time={elapsed_ms:.3f} ms"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  raised {type(exc).__name__}: {exc}")

        print("-" * 100)


if __name__ == "__main__":
    run_random_scenarios()
    run_invalid_start_target_scenarios()
