from __future__ import annotations

import csv
import time
from multiprocessing import Process
from multiprocessing.synchronize import Event as ProcessEvent
from pathlib import Path
from typing import Any

from trajplan.config import (
    build_differential_flatness_controller_config,
    build_quadrotor_physical_config,
)
from trajplan.controller import CommonController
from trajplan.df_controller import DifferentialFlatnessController
from trajplan.planning.messages import PlannerTickInput, PlannerTickOutput
from trajplan.quadrotor.agent import QuadrotorAgent
from trajplan.quadrotor.state import QuadrotorState
from trajplan.quadrotor.model import QuadrotorPhysicalConfig
from trajplan.runtime.backend import CommonBackend
from trajplan.runtime.channels import PlannerManagerAgentQueues
from trajplan.runtime.ipc import drain_latest, put_latest
from trajplan.runtime.rk4_simulator.backend import SimBackend
from trajplan.runtime.rk4_simulator.state_provider import SimStateProvider
from trajplan.runtime.state_provider import StateProvider
from trajplan.visualization.messages import AgentVisualizationSnapshot


class AgentProcess(Process):
    def __init__(
        self,
        agent_id: int,
        planner_manager_agent_queues: PlannerManagerAgentQueues,
        quadrotor_cfg: dict[str, Any],
        stop_event: ProcessEvent,
        control_interval: float,
        idle_sleep_time: float,
        initial_state: QuadrotorState | None = None,
        visualization_queue: Any | None = None,
        trajectory_log_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.agent_id = int(agent_id)
        self.planner_manager_agent_queues = planner_manager_agent_queues
        self.quadrotor_cfg = quadrotor_cfg
        self.stop_event = stop_event
        self.control_interval = control_interval
        self.idle_sleep_time = idle_sleep_time
        self.initial_state = None if initial_state is None else initial_state.copy()
        self.visualization_queue = visualization_queue
        self.trajectory_log_path = (
            None if trajectory_log_path is None else Path(trajectory_log_path)
        )

        self.agent: QuadrotorAgent | None = None  # constructed in run()

    def _build_state_provider(self) -> SimStateProvider:
        return SimStateProvider()

    def _build_backend(
        self,
        physical_config: QuadrotorPhysicalConfig,
    ) -> CommonBackend:
        return SimBackend(
            physical_config=physical_config,
            integration_dt=self.control_interval,
        )

    def _build_controller(
        self,
        physical_config: QuadrotorPhysicalConfig,
    ) -> CommonController:
        controller_config = build_differential_flatness_controller_config(
            self.quadrotor_cfg
        )
        return DifferentialFlatnessController(
            physical_config=physical_config,
            config=controller_config,
        )

    def _build_agent(
        self,
        state_provider: StateProvider,
        backend: CommonBackend,
        controller: CommonController,
    ) -> QuadrotorAgent:
        return QuadrotorAgent(
            state_provider=state_provider,
            backend=backend,
            controller=controller,
        )

    def _wait_until_state_provider_ready(
        self,
        state_provider: StateProvider,
    ) -> bool:
        while not self.stop_event.is_set():
            if state_provider.is_initialized():
                state_provider.get_state()
                return True
            time.sleep(self.idle_sleep_time)
        return False

    def _initialize_trajectory_log(self) -> None:
        if self.trajectory_log_path is None:
            return

        self.trajectory_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trajectory_log_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "x", "y", "z"])

    def _append_trajectory_log(
        self,
        timestamp: float,
        current_state: QuadrotorState,
    ) -> None:
        if self.trajectory_log_path is None:
            return

        position = current_state.position
        with self.trajectory_log_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    float(timestamp),
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                ]
            )

    def run(self) -> None:
        """
        Runtime construction should happen in the child process.
        """
        state_provider = self._build_state_provider()
        if self.initial_state is not None:
            state_provider.set_state(self.initial_state)
        physical_config = build_quadrotor_physical_config(self.quadrotor_cfg)
        backend = self._build_backend(physical_config)
        controller = self._build_controller(physical_config)
        if not self._wait_until_state_provider_ready(state_provider):
            return
        self.agent = self._build_agent(
            state_provider=state_provider,
            backend=backend,
            controller=controller,
        )
        self._initialize_trajectory_log()

        last_loop_time = time.monotonic()

        while not self.stop_event.is_set():
            now_time = time.monotonic()
            dt = now_time - last_loop_time

            if dt < self.control_interval:
                time.sleep(self.control_interval - dt)
                continue

            last_loop_time = now_time

            planner_output = drain_latest(self.planner_manager_agent_queues.pm_to_agent)
            if isinstance(planner_output, PlannerTickOutput):
                self.agent.try_commit_command(planner_output.command)

            self.agent.step(now_time=now_time, dt=dt)

            current_state = self.agent.get_state()
            active_command = self.agent.get_active_command()
            self._publish_visualization_snapshot(
                timestamp=now_time,
                current_state=current_state,
                active_command=active_command,
            )

            if self.agent.is_finished:
                break

            self._append_trajectory_log(
                timestamp=now_time,
                current_state=current_state,
            )

            tick_input = PlannerTickInput(
                state=current_state,
                timestamp=now_time,
                active_command=active_command,
            )
            put_latest(self.planner_manager_agent_queues.agent_to_pm, tick_input)

    def _publish_visualization_snapshot(
        self,
        timestamp: float,
        current_state: QuadrotorState,
        active_command,
    ) -> None:
        if self.visualization_queue is None:
            return

        snapshot = AgentVisualizationSnapshot(
            agent_id=self.agent_id,
            timestamp=float(timestamp),
            state=current_state.copy(),
            active_command=(
                None if active_command is None else active_command.copy()
            ),
        )
        put_latest(self.visualization_queue, snapshot)
