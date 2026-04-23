import time
from typing import Any, Sequence
import numpy as np
from multiprocessing import Process
from multiprocessing.synchronize import (
    Event as ProcessEvent,
)  # note this is only for annotation, when creating event, use from multiprocessing import Event
from trajplan.planning.messages import (
    NeighborTrajectoryMessage,
    PlannerTickInput,
    PlannerTickOutput,
    is_newer_neighbor_message,
)
from trajplan.runtime.channels import PlannerManagerAgentQueues, PlannerSwarmQueues
from trajplan.planning.planner_manager import PlannerManager
from trajplan.runtime.ipc import drain_all, drain_latest, put_latest

from trajplan.map.grid_map import GridMap
from trajplan.config import (
    build_grid_map_config,
    build_bspline_optimizer_config,
    build_local_planner_config,
    build_planner_manager_config,
)
from trajplan.planning.bspline_optimizer import BsplineOptimizer
from trajplan.planning.local_planner import LocalPlanner
from trajplan.shared_types import Vector
from trajplan.visualization.messages import PlannerVisualizationSnapshot


class PlannerManagerProcess(Process):
    def __init__(
        self,
        agent_id: int,
        planner_manager_agent_queues: PlannerManagerAgentQueues,
        quadrotor_cfg: dict[str, Any],
        planning_cfg: dict[str, Any],
        goal_positions: Sequence[Vector],
        stop_event: ProcessEvent,
        planning_interval: float,
        idle_sleep_time: float,
        swarm_queues: PlannerSwarmQueues | None = None,
        visualization_queue: Any | None = None,
    ) -> None:
        super().__init__()
        self.agent_id = int(agent_id)
        self.planner_manager_agent_queues = planner_manager_agent_queues
        self.stop_event = stop_event
        self.planning_interval = planning_interval
        self.idle_sleep_time = idle_sleep_time
        self.swarm_queues = swarm_queues
        self.planning_cfg = planning_cfg
        self.quadrotor_cfg = quadrotor_cfg
        self.goal_positions = [
            np.asarray(goal_position, dtype=np.float64).reshape(-1)
            for goal_position in goal_positions
        ]
        self.visualization_queue = visualization_queue

        # Placeholder only. The real object will be constructed in run().
        self.planner_manager: PlannerManager | None = None
        self._pending_neighbor_update: bool = False

    def run(self) -> None:
        grid_map_config = build_grid_map_config(self.planning_cfg)
        bspline_optimizer_config = build_bspline_optimizer_config(
            self.quadrotor_cfg, self.planning_cfg
        )
        local_planner_config = build_local_planner_config(
            self.quadrotor_cfg, self.planning_cfg
        )
        planner_manager_config = build_planner_manager_config(self.planning_cfg)

        grid_map = GridMap.from_config(grid_map_config)
        bspline_optimizer = BsplineOptimizer(config=bspline_optimizer_config)
        local_planner = LocalPlanner(
            optimizer=bspline_optimizer,
            config=local_planner_config,
        )
        self.planner_manager = PlannerManager(
            agent_id=self.agent_id,
            local_planner=local_planner,
            config=planner_manager_config,
            grid_map=grid_map,
        )
        self.planner_manager.add_goals(self.goal_positions)

        while not self.stop_event.is_set():
            self._pending_neighbor_update = (
                self._drain_swarm_messages() or self._pending_neighbor_update
            )

            tick_input = drain_latest(self.planner_manager_agent_queues.agent_to_pm)
            if not isinstance(tick_input, PlannerTickInput):
                time.sleep(self.idle_sleep_time)
                continue

            result = self.planner_manager.update_and_get_command(
                current_state=tick_input.state,
                msg_time=tick_input.timestamp,
                active_command=tick_input.active_command,
                neighbor_update_received=self._pending_neighbor_update,
            )
            global_duration = None
            if self.planner_manager.global_trajectory is not None:
                global_duration = float(self.planner_manager.global_trajectory.duration)
            progress_suffix = (
                "global_progress=None"
                if global_duration is None
                else "global_progress="
                f"{self.planner_manager.global_progress:.3f}/{global_duration:.3f}"
            )
            print(
                f"[PlannerManagerProcess {self.agent_id}] "
                f"t={tick_input.timestamp:.3f}, "
                f"{progress_suffix}, "
                f"should_publish={result.should_publish}",
                flush=True,
            )
            self._pending_neighbor_update = False
            self._publish_visualization_snapshot(timestamp=tick_input.timestamp)

            if not result.should_publish:
                time.sleep(self.idle_sleep_time)
                continue

            command = result.command
            tick_output = PlannerTickOutput(command=command)
            put_latest(self.planner_manager_agent_queues.pm_to_agent, tick_output)
            self._publish_swarm_trajectory(command)
            print(
                f"[PlannerManagerProcess {self.agent_id}] "
                f"t={tick_input.timestamp:.3f}, "
                f"command={command.mode}",
                flush=True,
            )

            if getattr(command, "is_finish", False):
                break

    def _drain_swarm_messages(self) -> bool:
        if self.swarm_queues is None:
            return False
        if self.planner_manager is None:
            return False

        messages = drain_all(self.swarm_queues.relay_to_planner)
        if len(messages) == 0:
            return False

        latest_messages_by_agent: dict[int, NeighborTrajectoryMessage] = {}
        for message in messages:
            if not isinstance(message, NeighborTrajectoryMessage):
                continue
            current_latest = latest_messages_by_agent.get(message.agent_id)
            if current_latest is None or is_newer_neighbor_message(
                message, current_latest
            ):
                latest_messages_by_agent[message.agent_id] = message

        applied_updates = 0
        for message in latest_messages_by_agent.values():
            if self.planner_manager.update_neighbor_trajectory(message):
                applied_updates += 1

        if applied_updates > 0:
            print(
                f"[PlannerManagerProcess {self.agent_id}] "
                f"received {applied_updates} neighbor trajectory update(s).",
                flush=True,
            )
            return True

        return False

    def _publish_swarm_trajectory(self, command) -> None:
        if self.swarm_queues is None:
            return
        if not getattr(command, "is_track", False):
            return
        if command.trajectory is None:
            return
        if command.traj_id is None:
            return

        message = NeighborTrajectoryMessage(
            agent_id=self.agent_id,
            traj_id=command.traj_id,
            start_time=command.start_time,
            trajectory=command.trajectory.copy(),
            local_target_pva=(
                None
                if command.local_target_pva is None
                else np.asarray(command.local_target_pva, dtype=np.float64).copy()
            ),
        )
        put_latest(self.swarm_queues.planner_to_relay, message)
        print(
            f"[PlannerManagerProcess {self.agent_id}] "
            f"published swarm trajectory traj_id={command.traj_id}.",
            flush=True,
        )

    def _publish_visualization_snapshot(
        self,
        timestamp: float,
    ) -> None:
        if self.visualization_queue is None:
            return
        if self.planner_manager is None:
            return

        global_trajectory = self.planner_manager.global_trajectory
        global_traj_points = None
        global_progress_position = None

        if global_trajectory is not None:
            duration = float(global_trajectory.duration)
            sample_times = np.linspace(0.0, duration, 200, dtype=np.float64)
            global_traj_points = np.vstack(
                [
                    np.asarray(
                        global_trajectory.evaluate_pva(float(sample_t))[:3],
                        dtype=np.float64,
                    )
                    for sample_t in sample_times
                ]
            )
            progress_t = min(max(self.planner_manager.global_progress, 0.0), duration)
            global_progress_position = np.asarray(
                global_trajectory.evaluate_pva(progress_t)[:3],
                dtype=np.float64,
            ).reshape(3)

        snapshot = PlannerVisualizationSnapshot(
            agent_id=self.agent_id,
            timestamp=float(timestamp),
            global_traj_points=global_traj_points,
            global_progress_position=global_progress_position,
        )
        put_latest(self.visualization_queue, snapshot)
