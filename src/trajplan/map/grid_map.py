from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from trajplan.shared_types import BoolArray, FloatArray, IntVector, Vector
from trajplan.utils import as_vector3


class GridMap:
    """
    3D occupancy grid map.

    Parameters
    ----------
    map_size
        Number of grid cells in each dimension, shape (3,), ordered as (nx, ny, nz).
    resolution
        Edge length of one grid cell in world units.
    origin
        World-coordinate origin of the map minimum corner, shape (3,).
    """

    def __init__(
        self,
        map_size: IntVector | tuple[int, int, int] | list[int],
        resolution: float,
        origin: Vector | None = None,
    ) -> None:
        map_size_array = np.asarray(map_size, dtype=np.int64).reshape(-1)
        resolution = float(resolution)

        if map_size_array.shape != (3,):
            raise ValueError(
                f"Expected map_size to have shape (3,), but got {map_size_array.shape}."
            )
        if np.any(map_size_array <= 0):
            raise ValueError(
                f"Expected positive map_size entries, but got {map_size_array}."
            )
        if resolution <= 0.0:
            raise ValueError(f"Expected resolution > 0, but got {resolution}.")

        if origin is None:
            origin_array = np.zeros(3, dtype=np.float64)
        else:
            origin_array = as_vector3(origin)

        self.map_size: IntVector = map_size_array
        self.nx, self.ny, self.nz = map_size_array.tolist()
        self.N = int(self.nx * self.ny * self.nz)

        self.resolution = resolution
        self.origin: Vector = origin_array

        # Flattened coordinate array for heuristic computation.
        self.idx_arr: FloatArray = np.vstack(
            np.unravel_index(
                np.arange(self.N),
                (self.nx, self.ny, self.nz),
            )
        ).T.astype(np.float64)

        # Per-node data in contiguous arrays.
        self.is_obstacle: BoolArray = np.zeros(self.N, dtype=np.bool_)
        self.visited: BoolArray = np.zeros(self.N, dtype=np.bool_)
        self.heuristic: FloatArray = np.full(self.N, -1.0, dtype=np.float64)
        self.g_score: FloatArray = np.full(self.N, np.inf, dtype=np.float64)
        self.parent: IntVector = np.full(self.N, -1, dtype=np.int64)
        self.open_heap: list[tuple[float, int]] = []

        # Precompute 26-neighbor steps in 3D index space.
        self.steps: IntVector = np.array(
            list(itertools.product((-1, 0, 1), repeat=3)),
            dtype=np.int64,
        )
        self.steps = self.steps[np.any(self.steps != 0, axis=1)]
        self.step_lengths: FloatArray = np.linalg.norm(self.steps, axis=1)

        # Constants for the same heuristic form used in the draft code.
        sqrt2 = math.sqrt(2.0)
        sqrt3 = math.sqrt(3.0)
        self.c1 = sqrt3 - 3.0
        self.c2 = sqrt2 - 2.0

        self.start_idx: int | None = None
        self.target_idx: int | None = None

    def to_flat(self, idx3: IntVector | tuple[int, int, int]) -> int:
        """
        Convert a 3D grid index to a flattened index.

        Parameters
        ----------
        idx3
            Grid index [ix, iy, iz].

        Returns
        -------
        int
            Flattened index.
        """
        idx3_array = np.asarray(idx3, dtype=np.int64).reshape(-1)

        if idx3_array.shape != (3,):
            raise ValueError(
                f"Expected idx3 to have shape (3,), but got {idx3_array.shape}."
            )

        ix, iy, iz = idx3_array.tolist()
        return int(ix * (self.ny * self.nz) + iy * self.nz + iz)

    def to_index(self, u: int) -> IntVector:
        """
        Convert a flattened index back to a 3D grid index.

        Parameters
        ----------
        u
            Flattened index.

        Returns
        -------
        IntVector
            Grid index [ix, iy, iz] with shape (3,).
        """
        u = int(u)

        ix = u // (self.ny * self.nz)
        rem = u % (self.ny * self.nz)
        iy, iz = divmod(rem, self.nz)

        return np.array([ix, iy, iz], dtype=np.int64)

    def is_inside_index(self, idx3: IntVector | tuple[int, int, int]) -> bool:
        """
        Check whether a 3D grid index lies inside the map bounds.
        """
        idx3_array = np.asarray(idx3, dtype=np.int64).reshape(-1)

        if idx3_array.shape != (3,):
            return False

        return bool(
            0 <= idx3_array[0] < self.nx
            and 0 <= idx3_array[1] < self.ny
            and 0 <= idx3_array[2] < self.nz
        )

    def world_to_grid(self, pt_in: Vector) -> IntVector:
        """
        Convert a world-coordinate point to a grid index.

        Parameters
        ----------
        pt_in
            World-coordinate point with shape (3,).

        Returns
        -------
        IntVector
            Grid index [ix, iy, iz].
        """
        pt = as_vector3(pt_in)
        idx = np.floor((pt - self.origin) / self.resolution).astype(np.int64)
        return idx

    def grid_to_world(self, idx3: IntVector | tuple[int, int, int]) -> FloatArray:
        """
        Convert a grid index to the world-coordinate center of that voxel.

        Parameters
        ----------
        idx3
            Grid index [ix, iy, iz].

        Returns
        -------
        FloatArray
            World-coordinate voxel center with shape (3,).
        """
        idx3_array = np.asarray(idx3, dtype=np.float64).reshape(-1)

        if idx3_array.shape != (3,):
            raise ValueError(
                f"Expected idx3 to have shape (3,), but got {idx3_array.shape}."
            )

        return self.origin + (idx3_array + 0.5) * self.resolution

    def update_start_target_index(
        self,
        agent_start_index: IntVector,
        agent_target_index: IntVector,
    ) -> None:
        """
        Set start and target indices, then update the heuristic.
        """
        start_idx3 = np.asarray(agent_start_index, dtype=np.int64).reshape(-1)
        target_idx3 = np.asarray(agent_target_index, dtype=np.int64).reshape(-1)

        if start_idx3.shape != (3,):
            raise ValueError(
                f"Expected agent_start_index to have shape (3,), but got {start_idx3.shape}."
            )
        if target_idx3.shape != (3,):
            raise ValueError(
                f"Expected agent_target_index to have shape (3,), but got {target_idx3.shape}."
            )
        if not self.is_inside_index(start_idx3):
            raise ValueError(f"Start index {start_idx3} is outside the map.")
        if not self.is_inside_index(target_idx3):
            raise ValueError(f"Target index {target_idx3} is outside the map.")

        self.start_idx = self.to_flat(start_idx3)
        self.target_idx = self.to_flat(target_idx3)

        self.update_heuristic()

    def update_heuristic(self) -> None:
        """
        Vectorized update of the heuristic for all nodes.
        """
        if self.target_idx is None:
            raise ValueError("Target index is not set.")

        target = self.to_index(self.target_idx).astype(np.float64)
        d = np.abs(self.idx_arr - target)

        sumv = d.sum(axis=1)
        mins = d.min(axis=1)
        maxs = d.max(axis=1)
        mids = sumv - mins - maxs

        # Same heuristic form as the draft code.
        self.heuristic = sumv + self.c1 * mins + self.c2 * (mids - mins)

    def search_reset(self) -> None:
        """
        Reset search-related arrays and clear the open heap.
        """
        self.visited[:] = False
        self.g_score[:] = np.inf
        self.parent[:] = -1
        self.open_heap = []

    def clear_obstacles(self) -> None:
        """
        Remove all obstacles from the map.
        """
        self.is_obstacle[:] = False

    def set_occupied_index(self, idx3: IntVector | tuple[int, int, int]) -> None:
        """
        Mark one voxel as occupied.
        """
        idx3_array = np.asarray(idx3, dtype=np.int64).reshape(-1)

        if not self.is_inside_index(idx3_array):
            return

        self.is_obstacle[self.to_flat(idx3_array)] = True

    def is_occupied_index(self, idx3: IntVector | tuple[int, int, int]) -> bool:
        """
        Check whether one voxel is occupied.
        Outside the map is treated as occupied.
        """
        idx3_array = np.asarray(idx3, dtype=np.int64).reshape(-1)

        if not self.is_inside_index(idx3_array):
            return True

        return bool(self.is_obstacle[self.to_flat(idx3_array)])

    def add_obstacle(
        self,
        obs_start_index: IntVector,
        obs_size: IntVector,
    ) -> None:
        """
        Mark a rectangular block of obstacles.

        Parameters
        ----------
        obs_start_index
            3D integer grid index of the obstacle block minimum corner.
        obs_size
            3D integer block size in number of voxels.
        """
        obs_start = np.asarray(obs_start_index, dtype=np.int64).reshape(-1)
        obs_size = np.asarray(obs_size, dtype=np.int64).reshape(-1)

        if obs_start.shape != (3,):
            raise ValueError(
                f"Expected obs_start_index to have shape (3,), but got {obs_start.shape}."
            )
        if obs_size.shape != (3,):
            raise ValueError(
                f"Expected obs_size to have shape (3,), but got {obs_size.shape}."
            )
        if np.any(obs_size <= 0):
            raise ValueError(f"Expected positive obs_size entries, but got {obs_size}.")

        end = obs_start + obs_size

        x0 = max(int(obs_start[0]), 0)
        y0 = max(int(obs_start[1]), 0)
        z0 = max(int(obs_start[2]), 0)

        x1 = min(int(end[0]), self.nx)
        y1 = min(int(end[1]), self.ny)
        z1 = min(int(end[2]), self.nz)

        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            return

        for x, y, z in itertools.product(
            range(x0, x1),
            range(y0, y1),
            range(z0, z1),
        ):
            self.is_obstacle[self.to_flat((x, y, z))] = True

    def is_occupied_world(self, pt_in: Vector) -> bool:
        """
        Check whether a continuous-space point lies in an obstacle or outside the map.

        Parameters
        ----------
        pt_in
            World-coordinate point with shape (3,).

        Returns
        -------
        bool
            True if occupied or outside the map, False otherwise.
        """
        idx = self.world_to_grid(pt_in)
        return self.is_occupied_index(idx)

    def check_occupancy(self, pt_in: Vector) -> bool:
        """
        Backward-compatible wrapper of is_occupied_world.
        """
        return self.is_occupied_world(pt_in)

    def astar_search(self) -> bool:
        """
        Run A* search.

        Returns
        -------
        bool
            True if a path to target is found, False otherwise.
        """
        if self.start_idx is None or self.target_idx is None:
            raise ValueError("Start/target indices are not set.")

        if self.is_obstacle[self.start_idx]:
            raise ValueError("Start voxel is occupied.")
        if self.is_obstacle[self.target_idx]:
            raise ValueError("Target voxel is occupied.")

        self.search_reset()
        self.g_score[self.start_idx] = 0.0
        heapq.heappush(
            self.open_heap,
            (self.heuristic[self.start_idx], self.start_idx),
        )

        while self.open_heap:
            _, u = heapq.heappop(self.open_heap)

            if self.visited[u]:
                continue

            self.visited[u] = True

            if u == self.target_idx:
                return True

            current_idx3 = self.to_index(u)

            for step, step_length in zip(self.steps, self.step_lengths):
                neighbor_idx3 = current_idx3 + step

                if not self.is_inside_index(neighbor_idx3):
                    continue

                v = self.to_flat(neighbor_idx3)

                if self.visited[v] or self.is_obstacle[v]:
                    continue

                tentative = self.g_score[u] + step_length

                if tentative < self.g_score[v]:
                    self.g_score[v] = tentative
                    self.parent[v] = u
                    heapq.heappush(
                        self.open_heap,
                        (tentative + self.heuristic[v], v),
                    )

        return False

    def retrieve_optimal_path(self) -> list[IntVector]:
        """
        Reconstruct the path from start to target.

        Returns
        -------
        list[IntVector]
            Path as a list of 3D grid indices ordered from start to target.
            Returns an empty list if no valid path exists.
        """
        if self.target_idx is None:
            return []

        if not self.visited[self.target_idx]:
            return []

        path: list[IntVector] = []
        u = self.target_idx

        while u >= 0:
            path.append(self.to_index(u))
            u = int(self.parent[u])

        path.reverse()
        return path
