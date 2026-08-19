from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from trajplan.map.grid_map import GridMapConfig
from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.shared_types import Vector
from trajplan.visualization.messages import (
    AgentVisualizationSnapshot,
)


@dataclass(slots=True)
class _AxisBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass(slots=True)
class _AgentArtists:
    point: object
    local_traj_line: object
    agent_history_line: object


class LiveVisualizer:
    _AGENT_COLOR_CYCLE = (
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    )

    def __init__(
        self,
        grid_map_config: GridMapConfig,
        title: str = "Trajectory Visualizer",
        local_traj_sample_num: int = 80,
        initial_position: Vector | None = None,
        goal_positions: Sequence[Vector] | None = None,
        agent_initial_positions: Mapping[int, Vector] | None = None,
        agent_goal_positions: Mapping[int, Sequence[Vector]] | None = None,
        fig: plt.Figure | None = None,
    ) -> None:
        self.grid_map_config = grid_map_config
        self.title = str(title)
        self.local_traj_sample_num = int(local_traj_sample_num)

        if self.local_traj_sample_num < 2:
            raise ValueError(
                "Expected local_traj_sample_num >= 2, "
                f"but got {self.local_traj_sample_num}."
            )

        if agent_initial_positions is None and agent_goal_positions is None:
            single_initial_positions: dict[int, np.ndarray] = {}
            single_goal_positions: dict[int, list[np.ndarray]] = {}
            if initial_position is not None:
                single_initial_positions[0] = np.asarray(
                    initial_position, dtype=np.float64
                ).reshape(3)
            if goal_positions is not None:
                single_goal_positions[0] = [
                    np.asarray(goal_position, dtype=np.float64).reshape(3)
                    for goal_position in goal_positions
                ]
            self._agent_initial_positions = single_initial_positions
            self._agent_goal_positions = single_goal_positions
        else:
            self._agent_initial_positions = self._normalize_agent_initial_positions(
                agent_initial_positions
            )
            self._agent_goal_positions = self._normalize_agent_goal_positions(
                agent_goal_positions
            )

        self._latest_agent_snapshots: dict[int, AgentVisualizationSnapshot] = {}
        self._agent_history_positions: dict[int, list[np.ndarray]] = {}
        self._last_history_timestamp: dict[int, float] = {}
        self._agent_artists: dict[int, _AgentArtists] = {}
        self._obstacle_artists: list[Poly3DCollection] = []

        self._owns_fig = fig is None
        if fig is None:
            self._interactive_enabled = "agg" not in plt.get_backend().lower()
            plt.ion()
            self.fig = plt.figure(figsize=(10, 8))
        else:
            self._interactive_enabled = False
            self.fig = fig
        self.ax = self.fig.add_subplot(111, projection="3d")

        self._bounds = self._compute_axis_bounds(grid_map_config)
        self._configure_axes()
        self._draw_obstacles()
        self._draw_static_reference_points()

        for agent_id in self._known_agent_ids():
            self._ensure_agent_artists(agent_id)
        self._refresh_legend()

    def update(
        self,
        agent_snapshot: AgentVisualizationSnapshot | None = None,
        agent_snapshots: Mapping[int, AgentVisualizationSnapshot | None] | None = None,
    ) -> None:
        self._ingest_agent_snapshot(agent_snapshot)

        if agent_snapshots is not None:
            for snapshot in agent_snapshots.values():
                self._ingest_agent_snapshot(snapshot)

        if not plt.fignum_exists(self.fig.number):
            return

        for agent_id in self._known_agent_ids():
            self._ensure_agent_artists(agent_id)
            self._update_agent_artists(agent_id)

        self.fig.canvas.draw_idle()
        if self._interactive_enabled and self._owns_fig:
            plt.pause(0.001)

    def close(self) -> None:
        if not self._owns_fig:
            return
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

    def wait_until_closed(self) -> None:
        if not self._owns_fig:
            return
        if not self._interactive_enabled:
            return
        if not plt.fignum_exists(self.fig.number):
            return

        plt.ioff()
        plt.show(block=True)

    def add_obstacle_world(self, center: Vector, size: Vector) -> None:
        center_arr = np.asarray(center, dtype=np.float64).reshape(3)
        size_arr = np.asarray(size, dtype=np.float64).reshape(3)
        min_corner = center_arr - size_arr / 2.0
        faces = self._build_cuboid_faces(min_corner=min_corner, size=size_arr)
        collection = Poly3DCollection(
            faces,
            facecolors="tab:red",
            edgecolors="tab:red",
            linewidths=0.5,
            alpha=0.25,
        )
        self.ax.add_collection3d(collection)
        self._obstacle_artists.append(collection)
        self.fig.canvas.draw_idle()

    def clear_obstacle_artists(self) -> None:
        for collection in self._obstacle_artists:
            try:
                collection.remove()
            except (ValueError, NotImplementedError):
                pass
        self._obstacle_artists.clear()
        self.fig.canvas.draw_idle()

    def _ingest_agent_snapshot(
        self,
        snapshot: AgentVisualizationSnapshot | None,
    ) -> None:
        if snapshot is None:
            return
        agent_id = int(snapshot.agent_id)
        self._latest_agent_snapshots[agent_id] = snapshot
        self._record_agent_history(snapshot)

    def _known_agent_ids(self) -> list[int]:
        agent_ids = set(self._agent_initial_positions.keys())
        agent_ids.update(self._agent_goal_positions.keys())
        agent_ids.update(self._latest_agent_snapshots.keys())
        agent_ids.update(self._agent_artists.keys())
        return sorted(agent_ids)

    def _configure_axes(self) -> None:
        self.ax.set_title(self.title)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_xlim(self._bounds.x_min, self._bounds.x_max)
        self.ax.set_ylim(self._bounds.y_min, self._bounds.y_max)
        self.ax.set_zlim(self._bounds.z_min, self._bounds.z_max)
        self.ax.set_box_aspect(
            (
                self._bounds.x_max - self._bounds.x_min,
                self._bounds.y_max - self._bounds.y_min,
                self._bounds.z_max - self._bounds.z_min,
            )
        )

    def _draw_obstacles(self) -> None:
        resolution = float(self.grid_map_config.resolution)
        origin = np.asarray(self.grid_map_config.origin, dtype=np.float64).reshape(3)

        for obstacle in self.grid_map_config.obstacles:
            min_corner = origin + resolution * obstacle.obs_start_index.astype(np.float64)
            size = resolution * obstacle.obs_size.astype(np.float64)
            faces = self._build_cuboid_faces(min_corner=min_corner, size=size)
            collection = Poly3DCollection(
                faces,
                facecolors="tab:red",
                edgecolors="tab:red",
                linewidths=0.5,
                alpha=0.25,
            )
            self.ax.add_collection3d(collection)
            self._obstacle_artists.append(collection)

    def _draw_static_reference_points(self) -> None:
        for agent_id in self._known_agent_ids():
            color = self._get_agent_color(agent_id)

            initial_position = self._agent_initial_positions.get(agent_id)
            if initial_position is not None:
                self.ax.plot(
                    [initial_position[0]],
                    [initial_position[1]],
                    [initial_position[2]],
                    marker="x",
                    linestyle="None",
                    markersize=9,
                    markeredgewidth=2.0,
                    color=color,
                    alpha=0.95,
                )
                self.ax.text(
                    float(initial_position[0]),
                    float(initial_position[1]),
                    float(initial_position[2]),
                    f"{agent_id}:S",
                    color=color,
                    fontsize=9,
                )

            for goal_idx, goal_position in enumerate(
                self._agent_goal_positions.get(agent_id, []),
                start=1,
            ):
                self.ax.plot(
                    [goal_position[0]],
                    [goal_position[1]],
                    [goal_position[2]],
                    marker="*",
                    linestyle="None",
                    markersize=11,
                    color=color,
                    alpha=0.9,
                )
                self.ax.text(
                    float(goal_position[0]),
                    float(goal_position[1]),
                    float(goal_position[2]),
                    f"{agent_id}.{goal_idx}",
                    color=color,
                    fontsize=9,
                )

    def _ensure_agent_artists(self, agent_id: int) -> None:
        if agent_id in self._agent_artists:
            return

        color = self._get_agent_color(agent_id)
        (point,) = self.ax.plot(
            [],
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=8,
            color=color,
            label=f"agent {agent_id}",
        )
        (local_traj_line,) = self.ax.plot(
            [],
            [],
            [],
            linestyle="--",
            linewidth=2.0,
            color=color,
            alpha=0.9,
            label="_nolegend_",
        )
        (agent_history_line,) = self.ax.plot(
            [],
            [],
            [],
            linestyle="-",
            linewidth=1.5,
            color=color,
            alpha=0.55,
            label="_nolegend_",
        )
        self._agent_artists[agent_id] = _AgentArtists(
            point=point,
            local_traj_line=local_traj_line,
            agent_history_line=agent_history_line,
        )
        self._agent_history_positions.setdefault(agent_id, [])
        self._refresh_legend()

    def _refresh_legend(self) -> None:
        handles = [
            self._agent_artists[agent_id].point for agent_id in sorted(self._agent_artists)
        ]
        labels = [f"agent {agent_id}" for agent_id in sorted(self._agent_artists)]
        if handles:
            self.ax.legend(handles, labels, loc="upper right")

    def _update_agent_artists(self, agent_id: int) -> None:
        artists = self._agent_artists[agent_id]
        snapshot = self._latest_agent_snapshots.get(agent_id)

        if snapshot is None:
            artists.point.set_data_3d([], [], [])
            artists.local_traj_line.set_data_3d([], [], [])
            artists.agent_history_line.set_data_3d([], [], [])
            return

        base_color = self._get_agent_color(agent_id)
        position = np.asarray(snapshot.state.position, dtype=np.float64).reshape(3)
        artists.point.set_data_3d(
            [position[0]],
            [position[1]],
            [position[2]],
        )
        artists.point.set_color(base_color)

        marker = "o"
        local_traj_points: np.ndarray | None = None
        command = snapshot.active_command

        if command is None:
            marker = "o"
        elif command.is_hover:
            marker = "^"
        elif command.is_track:
            marker = "o"
            local_traj_points = self._sample_future_local_trajectory(
                command=command,
                timestamp=snapshot.timestamp,
            )
        elif command.is_emergency_stop:
            marker = "X"
            local_traj_points = self._sample_future_local_trajectory(
                command=command,
                timestamp=snapshot.timestamp,
            )
        elif command.is_finish:
            marker = "s"

        artists.point.set_marker(marker)
        artists.local_traj_line.set_color(base_color)
        artists.agent_history_line.set_color(base_color)

        if local_traj_points is None or local_traj_points.size == 0:
            artists.local_traj_line.set_data_3d([], [], [])
        else:
            artists.local_traj_line.set_data_3d(
                local_traj_points[:, 0],
                local_traj_points[:, 1],
                local_traj_points[:, 2],
            )

        history_positions = self._agent_history_positions.get(agent_id, [])
        if len(history_positions) == 0:
            artists.agent_history_line.set_data_3d([], [], [])
        else:
            history_points = np.vstack(history_positions)
            artists.agent_history_line.set_data_3d(
                history_points[:, 0],
                history_points[:, 1],
                history_points[:, 2],
            )

    def _sample_future_local_trajectory(
        self,
        command: QuadrotorCommand,
        timestamp: float,
    ) -> np.ndarray | None:
        trajectory = command.trajectory
        if trajectory is None:
            return None

        t_end = float(trajectory.duration)
        t_cur = min(command.get_elapsed_time(timestamp), t_end)
        if t_end <= t_cur:
            position = trajectory.evaluate_pva(t_end)[:3]
            return np.asarray(position, dtype=np.float64).reshape(1, 3)

        sample_times = np.linspace(
            t_cur,
            t_end,
            self.local_traj_sample_num,
            dtype=np.float64,
        )
        return np.vstack(
            [
                np.asarray(
                    trajectory.evaluate_pva(float(sample_t))[:3],
                    dtype=np.float64,
                )
                for sample_t in sample_times
            ]
        )

    def _record_agent_history(
        self,
        agent_snapshot: AgentVisualizationSnapshot,
    ) -> None:
        agent_id = int(agent_snapshot.agent_id)
        timestamp = float(agent_snapshot.timestamp)
        last_timestamp = self._last_history_timestamp.get(agent_id)
        if last_timestamp is not None and timestamp <= last_timestamp:
            return

        position = np.asarray(agent_snapshot.state.position, dtype=np.float64).reshape(3)
        self._agent_history_positions.setdefault(agent_id, []).append(position.copy())
        self._last_history_timestamp[agent_id] = timestamp

    def _get_agent_color(self, agent_id: int) -> str:
        color_index = int(agent_id) % len(self._AGENT_COLOR_CYCLE)
        return self._AGENT_COLOR_CYCLE[color_index]

    @staticmethod
    def _normalize_agent_initial_positions(
        agent_initial_positions: Mapping[int, Vector] | None,
    ) -> dict[int, np.ndarray]:
        if agent_initial_positions is None:
            return {}
        normalized: dict[int, np.ndarray] = {}
        for agent_id, position in agent_initial_positions.items():
            normalized[int(agent_id)] = np.asarray(
                position,
                dtype=np.float64,
            ).reshape(3)
        return normalized

    @staticmethod
    def _normalize_agent_goal_positions(
        agent_goal_positions: Mapping[int, Sequence[Vector]] | None,
    ) -> dict[int, list[np.ndarray]]:
        if agent_goal_positions is None:
            return {}
        normalized: dict[int, list[np.ndarray]] = {}
        for agent_id, goal_positions in agent_goal_positions.items():
            normalized[int(agent_id)] = [
                np.asarray(goal_position, dtype=np.float64).reshape(3)
                for goal_position in goal_positions
            ]
        return normalized

    @staticmethod
    def _compute_axis_bounds(grid_map_config: GridMapConfig) -> _AxisBounds:
        origin = np.asarray(grid_map_config.origin, dtype=np.float64).reshape(3)
        map_size = np.asarray(grid_map_config.map_size, dtype=np.int64).reshape(3)
        resolution = float(grid_map_config.resolution)
        extent = map_size.astype(np.float64) * resolution
        max_corner = origin + extent
        return _AxisBounds(
            x_min=float(origin[0]),
            x_max=float(max_corner[0]),
            y_min=float(origin[1]),
            y_max=float(max_corner[1]),
            z_min=float(origin[2]),
            z_max=float(max_corner[2]),
        )

    @staticmethod
    def _build_cuboid_faces(
        min_corner: np.ndarray,
        size: np.ndarray,
    ) -> list[list[np.ndarray]]:
        x0, y0, z0 = min_corner.tolist()
        dx, dy, dz = size.tolist()
        x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz

        v000 = np.array([x0, y0, z0], dtype=np.float64)
        v001 = np.array([x0, y0, z1], dtype=np.float64)
        v010 = np.array([x0, y1, z0], dtype=np.float64)
        v011 = np.array([x0, y1, z1], dtype=np.float64)
        v100 = np.array([x1, y0, z0], dtype=np.float64)
        v101 = np.array([x1, y0, z1], dtype=np.float64)
        v110 = np.array([x1, y1, z0], dtype=np.float64)
        v111 = np.array([x1, y1, z1], dtype=np.float64)

        return [
            [v000, v100, v110, v010],
            [v001, v101, v111, v011],
            [v000, v100, v101, v001],
            [v010, v110, v111, v011],
            [v000, v010, v011, v001],
            [v100, v110, v111, v101],
        ]
