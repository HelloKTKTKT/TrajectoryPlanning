from __future__ import annotations

from collections.abc import Mapping, Sequence
from multiprocessing import Queue
from multiprocessing.synchronize import Event as ProcessEvent
import sys

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from trajplan.map.grid_map import GridMapConfig
from trajplan.runtime.channels import ControlQueues
from trajplan.runtime.ipc import drain_latest, put_latest
from trajplan.shared_types import Vector
from trajplan.visualization.live_visualizer import LiveVisualizer
from trajplan.visualization.messages import (
    AgentVisualizationSnapshot,
    GoalUpdateMessage,
    ObstacleUpdateMessage,
)


class InteractiveVisualizerWindow(QMainWindow):
    def __init__(
        self,
        grid_map_config: GridMapConfig,
        agent_visualization_queues: Mapping[int, Queue],
        planner_visualization_queues: Mapping[int, "Queue | None"],
        control_queues: Mapping[int, ControlQueues],
        agent_initial_positions: Mapping[int, Vector],
        agent_goal_positions: Mapping[int, Sequence[Vector]],
        swarm: bool = False,
        stop_event: ProcessEvent | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Crazyflie Interactive Planner")

        self._grid_map_config = grid_map_config
        self._agent_visualization_queues = dict(agent_visualization_queues)
        self._planner_visualization_queues = dict(planner_visualization_queues)
        self._control_queues = dict(control_queues)
        self._swarm = bool(swarm)
        self._stop_event = stop_event

        self._agent_ids = sorted(self._control_queues.keys())
        if len(self._agent_ids) == 0:
            raise ValueError("Expected at least one control queue.")

        self._latest_snapshots: dict[int, AgentVisualizationSnapshot | None] = {
            aid: None for aid in self._agent_ids
        }
        self._current_goals: dict[int, list[np.ndarray]] = {
            int(aid): [np.asarray(g, dtype=np.float64).reshape(3) for g in goals]
            for aid, goals in agent_goal_positions.items()
        }
        self._runtime_obstacles_2d: list[tuple[np.ndarray, np.ndarray]] = []

        self._fig_3d = Figure(figsize=(9, 7))
        self._canvas_3d = FigureCanvasQTAgg(self._fig_3d)
        self._live_viz = LiveVisualizer(
            grid_map_config=grid_map_config,
            title="Crazyflie Live View",
            agent_initial_positions=agent_initial_positions,
            agent_goal_positions=self._current_goals,
            fig=self._fig_3d,
        )

        self._fig_2d = Figure(figsize=(5, 5))
        self._canvas_2d = FigureCanvasQTAgg(self._fig_2d)
        self._ax_2d = self._fig_2d.add_subplot(111)
        self._configure_2d_axes()
        self._canvas_2d.mpl_connect("button_press_event", self._on_2d_click)

        central = QWidget()
        outer = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._canvas_3d)
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._canvas_2d, stretch=1)
        right_layout.addWidget(self._build_controls())
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        outer.addWidget(splitter)
        self.setCentralWidget(central)
        self.resize(1400, 850)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(33)

    def _configure_2d_axes(self) -> None:
        bounds = self._live_viz._bounds
        self._ax_2d.set_title("Top-down (click: set goal / obstacle)")
        self._ax_2d.set_xlabel("x [m]")
        self._ax_2d.set_ylabel("y [m]")
        self._ax_2d.set_xlim(bounds.x_min, bounds.x_max)
        self._ax_2d.set_ylim(bounds.y_min, bounds.y_max)
        self._ax_2d.set_aspect("equal", adjustable="box")
        self._ax_2d.grid(True, alpha=0.3)

        resolution = float(self._grid_map_config.resolution)
        origin = np.asarray(self._grid_map_config.origin, dtype=np.float64).reshape(3)
        for obstacle in self._grid_map_config.obstacles:
            min_corner = origin + resolution * obstacle.obs_start_index.astype(np.float64)
            size = resolution * obstacle.obs_size.astype(np.float64)
            patch = Rectangle(
                (float(min_corner[0]), float(min_corner[1])),
                float(size[0]),
                float(size[1]),
                facecolor="tab:red",
                edgecolor="tab:red",
                alpha=0.25,
            )
            self._ax_2d.add_patch(patch)

    def _build_controls(self) -> QWidget:
        box = QGroupBox("Controls")
        layout = QVBoxLayout(box)

        mode_row = QHBoxLayout()
        self._mode_goal_radio = QRadioButton("Goal")
        self._mode_obs_radio = QRadioButton("Obstacle")
        self._mode_goal_radio.setChecked(True)
        mode_row.addWidget(self._mode_goal_radio)
        mode_row.addWidget(self._mode_obs_radio)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self._goal_z_spin = QDoubleSpinBox()
        self._goal_z_spin.setRange(0.0, 3.0)
        self._goal_z_spin.setSingleStep(0.1)
        self._goal_z_spin.setValue(0.8)
        self._goal_z_spin.setSuffix(" m")
        layout.addLayout(self._label_row("Goal Z:", self._goal_z_spin))

        self._obs_size_spin = QDoubleSpinBox()
        self._obs_size_spin.setRange(0.1, 2.0)
        self._obs_size_spin.setSingleStep(0.1)
        self._obs_size_spin.setValue(0.4)
        self._obs_size_spin.setSuffix(" m")
        layout.addLayout(self._label_row("Obstacle XY size:", self._obs_size_spin))

        random_row = QHBoxLayout()
        self._random_count_spin = QSpinBox()
        self._random_count_spin.setRange(1, 10)
        self._random_count_spin.setValue(5)
        self._random_button = QPushButton("Add Random Obstacles")
        self._random_button.clicked.connect(self._on_random_obstacles)
        random_row.addWidget(self._random_button)
        random_row.addWidget(self._random_count_spin)
        layout.addLayout(random_row)

        self._clear_obs_button = QPushButton("Clear Obstacles")
        self._clear_obs_button.clicked.connect(self._on_clear_obstacles)
        layout.addWidget(self._clear_obs_button)

        if self._swarm and len(self._agent_ids) > 1:
            self._agent_combo: QComboBox | None = QComboBox()
            for aid in self._agent_ids:
                self._agent_combo.addItem(str(aid), userData=int(aid))
            self._apply_all_check: QCheckBox | None = QCheckBox("Apply to all")
            layout.addLayout(self._label_row("Agent:", self._agent_combo))
            layout.addWidget(self._apply_all_check)
        else:
            self._agent_combo = None
            self._apply_all_check = None

        self._quit_button = QPushButton("Quit")
        self._quit_button.clicked.connect(self.close)
        layout.addWidget(self._quit_button)

        layout.addStretch(1)
        return box

    @staticmethod
    def _label_row(label: str, widget: QWidget) -> QVBoxLayout:
        row = QVBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label))
        row.addWidget(widget)
        return row

    def _target_agent_ids(self) -> list[int]:
        if (
            self._swarm
            and self._apply_all_check is not None
            and self._apply_all_check.isChecked()
        ):
            return list(self._agent_ids)
        if self._agent_combo is not None:
            return [int(self._agent_combo.currentData())]
        return [self._agent_ids[0]]

    def _on_2d_click(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if getattr(event, "button", None) not in (1, "left"):
            return

        x = float(event.xdata)
        y = float(event.ydata)
        z = float(self._goal_z_spin.value())

        if self._mode_goal_radio.isChecked():
            goal = np.array([x, y, z], dtype=np.float64)
            for aid in self._target_agent_ids():
                msg = GoalUpdateMessage(agent_id=int(aid), goal_position=goal.copy())
                put_latest(self._control_queues[int(aid)].gui_to_planner, msg)
                self._current_goals[int(aid)] = [goal.copy()]
            print(
                f"[GUI] goal sent: {goal.tolist()} "
                f"agents={[int(a) for a in self._target_agent_ids()]}",
                flush=True,
            )
        else:
            s = float(self._obs_size_spin.value())
            center = np.array([x, y, z], dtype=np.float64)
            size = np.array([s, s, 1.5], dtype=np.float64)
            self._send_obstacle(center=center, size=size)

    def _send_obstacle(self, center: np.ndarray, size: np.ndarray) -> None:
        msg = ObstacleUpdateMessage(action="add", center=center, size=size)
        for aid in self._agent_ids:
            put_latest(self._control_queues[int(aid)].gui_to_planner, msg)
        self._live_viz.add_obstacle_world(center, size)
        self._runtime_obstacles_2d.append((center.copy(), size.copy()))

    def _on_random_obstacles(self) -> None:
        count = int(self._random_count_spin.value())
        bounds = self._live_viz._bounds
        rng = np.random.default_rng()
        s = float(self._obs_size_spin.value())
        z = float(self._goal_z_spin.value())

        for _ in range(count):
            xy = None
            for _attempt in range(30):
                cand_x = float(rng.uniform(bounds.x_min + 0.1, bounds.x_max - 0.1))
                cand_y = float(rng.uniform(bounds.y_min + 0.1, bounds.y_max - 0.1))
                too_close = False
                for snap in self._latest_snapshots.values():
                    if snap is None:
                        continue
                    p = np.asarray(snap.state.position, dtype=np.float64).reshape(3)
                    if float(np.hypot(p[0] - cand_x, p[1] - cand_y)) < 0.3:
                        too_close = True
                        break
                if not too_close:
                    xy = (cand_x, cand_y)
                    break
            if xy is None:
                continue
            center = np.array([xy[0], xy[1], z], dtype=np.float64)
            size = np.array([s, s, 1.5], dtype=np.float64)
            self._send_obstacle(center=center, size=size)

    def _on_clear_obstacles(self) -> None:
        msg = ObstacleUpdateMessage(action="clear", center=None, size=None)
        for aid in self._agent_ids:
            put_latest(self._control_queues[int(aid)].gui_to_planner, msg)
        self._live_viz.clear_obstacle_artists()
        self._runtime_obstacles_2d.clear()

    def _on_tick(self) -> None:
        for aid, q in self._agent_visualization_queues.items():
            snapshot = drain_latest(q)
            if snapshot is not None:
                self._latest_snapshots[int(aid)] = snapshot

        for q in self._planner_visualization_queues.values():
            if q is None:
                continue
            drain_latest(q)

        self._live_viz.update(agent_snapshots=self._latest_snapshots)
        self._update_2d_overlay()
        self._canvas_3d.draw_idle()
        self._canvas_2d.draw_idle()

    def _update_2d_overlay(self) -> None:
        for artist in list(self._ax_2d.lines):
            artist.remove()
        for artist in list(self._ax_2d.patches):
            if artist.get_label() == "__dynamic__":
                artist.remove()

        for center, size in self._runtime_obstacles_2d:
            half_x = float(size[0]) / 2.0
            half_y = float(size[1]) / 2.0
            patch = Rectangle(
                (float(center[0]) - half_x, float(center[1]) - half_y),
                float(size[0]),
                float(size[1]),
                facecolor="tab:red",
                edgecolor="tab:red",
                alpha=0.25,
                label="__dynamic__",
            )
            self._ax_2d.add_patch(patch)

        for aid, snap in self._latest_snapshots.items():
            if snap is None:
                continue
            color = self._live_viz._get_agent_color(int(aid))
            p = np.asarray(snap.state.position, dtype=np.float64).reshape(3)
            self._ax_2d.plot(
                [float(p[0])],
                [float(p[1])],
                marker="o",
                linestyle="None",
                color=color,
                markersize=10,
                label="__dynamic__",
            )

        for aid, goals in self._current_goals.items():
            if not goals:
                continue
            color = self._live_viz._get_agent_color(int(aid))
            goal = goals[0]
            self._ax_2d.plot(
                [float(goal[0])],
                [float(goal[1])],
                marker="*",
                linestyle="None",
                color=color,
                markersize=14,
                markeredgecolor="black",
                label="__dynamic__",
            )

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        if self._stop_event is not None:
            self._stop_event.set()
        super().closeEvent(event)


def run_interactive_app(
    grid_map_config: GridMapConfig,
    agent_visualization_queues: Mapping[int, Queue],
    planner_visualization_queues: Mapping[int, "Queue | None"],
    control_queues: Mapping[int, ControlQueues],
    agent_initial_positions: Mapping[int, Vector],
    agent_goal_positions: Mapping[int, Sequence[Vector]],
    swarm: bool = False,
    stop_event: ProcessEvent | None = None,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = InteractiveVisualizerWindow(
        grid_map_config=grid_map_config,
        agent_visualization_queues=agent_visualization_queues,
        planner_visualization_queues=planner_visualization_queues,
        control_queues=control_queues,
        agent_initial_positions=agent_initial_positions,
        agent_goal_positions=agent_goal_positions,
        swarm=swarm,
        stop_event=stop_event,
    )
    window.show()
    return app.exec()
