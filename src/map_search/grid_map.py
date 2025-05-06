import numpy as np
import itertools
import heapq
import math
import src.map_search.parameters as map_para


class GridMap:
    def __init__(self, map_size):
        """
        :param map_size: 1D numpy array (int), number of grids in each dimension (nx, ny, nz)
        """
        # Grid dimensions
        self.map_size = map_size.astype(int)
        self.nx, self.ny, self.nz = self.map_size
        self.N = self.nx * self.ny * self.nz

        # Flattened coordinate array for heuristic computation
        self.idx_arr = np.vstack(
            np.unravel_index(
                np.arange(self.N),
                (self.nx, self.ny, self.nz)
            )
        ).T  # shape: (N, 3)

        # Per-node data in contiguous arrays
        self.is_obstacle = np.zeros(self.N, dtype=bool)
        self.visited     = np.zeros(self.N, dtype=bool)
        self.heuristic   = np.full(self.N, -1.0, dtype=float)
        self.g_score     = np.full(self.N, np.inf, dtype=float)
        self.parent      = np.full(self.N, -1, dtype=int)
        self.open_heap = []

        # Precompute neighbor offsets (flat index) and their Euclidean lengths
        steps = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=int)
        steps = steps[np.any(steps != 0, axis=1)]  # remove the zero offset
        self.step_offsets = (
            steps[:, 0] * (self.ny * self.nz)
          + steps[:, 1] * self.nz
          + steps[:, 2]
        )  # shape: (~26,)
        self.step_lengths = np.linalg.norm(steps, axis=1)

        # Constants for Octile/Euclidean heuristic
        sqrt2 = math.sqrt(2.0)
        sqrt3 = math.sqrt(3.0)
        self.c1 = sqrt3 - 3.0
        self.c2 = sqrt2 - 2.0

        # Start and target flat indices
        self.start_idx = None
        self.target_idx = None

    def to_flat(self, idx3):
        """
        Convert a 3D index to flattened index.
        :param idx3: array-like [ix, iy, iz]
        :return: int flat index
        """
        ix, iy, iz = idx3
        return int(ix * (self.ny * self.nz) + iy * self.nz + iz)

    def to_index(self, u):
        """
        Convert a flat index back to 3D.
        :param u: int flat index
        :return: numpy array [ix, iy, iz]
        """
        ix = u // (self.ny * self.nz)
        rem = u % (self.ny * self.nz)
        iy, iz = divmod(rem, self.nz)
        return np.array([ix, iy, iz], dtype=int)

    def update_start_target_index(self, agent_start_index, agent_target_index):
        """
        Set start/target and recompute heuristic + reset search structures.
        """
        self.start_idx  = self.to_flat(agent_start_index.astype(int))
        self.target_idx = self.to_flat(agent_target_index.astype(int))
        self.update_heuristic()
        self.search_reset()

    def update_heuristic(self):
        """
        Vectorized update of the Octile/Euler heuristic for all nodes.
        """
        target = self.to_index(self.target_idx).astype(float)
        d = np.abs(self.idx_arr - target)
        sumv = d.sum(axis=1)
        mins = d.min(axis=1)
        maxs = d.max(axis=1)
        mids = sumv - mins - maxs

        # h = sumv + c1*mins + c2*(mids-mins)
        self.heuristic = sumv + self.c1 * mins + self.c2 * (mids - mins)

    def search_reset(self):
        """
        Reset visited, g_score, parent arrays and clear the open heap.
        """
        self.visited[:] = False
        self.g_score[:] = np.inf
        self.parent[:]  = -1
        self.open_heap  = []  # will hold tuples (f_score, flat_index)

    def add_obstacle(self, obs_start_index, obs_size):
        """
        Mark a rectangular block of obstacles.
        :param obs_start_index: 3-array int, start of obstacle block
        :param obs_size: 3-array int, size of block in each dimension
        """
        obs_start = obs_start_index.astype(int)
        obs_size  = obs_size.astype(int)
        end = obs_start + obs_size
        for x, y, z in itertools.product(
            range(obs_start[0], end[0]),
            range(obs_start[1], end[1]),
            range(obs_start[2], end[2])
        ):
            u = self.to_flat((x, y, z))
            self.is_obstacle[u] = True

    def check_occupancy(self, pt_in):
        """
        Check if a point in continuous space lies in an obstacle or outside the map.
        """
        pt = pt_in.flatten()
        idx = (pt / map_para.GRID_SIZE).astype(int)
        if (0 <= idx[0] < self.nx and
            0 <= idx[1] < self.ny and
            0 <= idx[2] < self.nz):
            return self.is_obstacle[self.to_flat(idx)]
        return True

    def astar_search(self):
        """
        Run A* search. Returns True if a path to target is found.
        """
        if self.start_idx is None or self.target_idx is None:
            raise ValueError("Start/target indices not set.")

        # Initialize start
        self.g_score[self.start_idx] = 0.0
        heapq.heappush(
            self.open_heap,
            (self.heuristic[self.start_idx], self.start_idx)
        )

        # Main search loop
        while self.open_heap:
            f_cur, u = heapq.heappop(self.open_heap)
            if self.visited[u]:
                continue
            self.visited[u] = True

            if u == self.target_idx:
                return True

            # Expand neighbors
            for ds, dl in zip(self.step_offsets, self.step_lengths):
                v = u + ds
                if not (0 <= v < self.N):
                    continue
                if self.visited[v] or self.is_obstacle[v]:
                    continue

                tentative = self.g_score[u] + dl
                if tentative < self.g_score[v]:
                    self.g_score[v] = tentative
                    self.parent[v]  = u
                    heapq.heappush(
                        self.open_heap,
                        (tentative + self.heuristic[v], v)
                    )
        return False

    def retrieve_optimal_path(self):
        """
        Reconstruct the path from start to target as a list of 3D tuples.
        """
        if self.target_idx is None:
            return []

        path = []
        u = self.target_idx
        while u >= 0:
            idx3 = self.to_index(u)
            path.append(idx3)
            # path.append((int(idx3[0]), int(idx3[1]), int(idx3[2])))
            u = self.parent[u]
        return path
