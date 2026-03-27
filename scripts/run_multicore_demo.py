from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue
from pathlib import Path
from queue import Empty, Full
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trajplan.config import (
    build_bspline_optimizer_config,
    build_linear_mpc_config,
    build_local_planner_config,
    build_planner_manager_config,
    load_project_configs,
)
from trajplan.controller import Controller
from trajplan.map.grid_map import GridMap
from trajplan.planning.bspline_optimizer import BsplineOptimizer
from trajplan.planning.local_planner import LocalPlanner
from trajplan.planning.planner_manager import PlannerManager
from trajplan.quadrotor.agent import QuadrotorAgent
from trajplan.quadrotor.command import QuadrotorCommand
from trajplan.quadrotor.state import QuadrotorState
from trajplan.runtime.sim_backend import SimBackend
from trajplan.runtime.sim_state_provider import SimStateProvider


@dataclass(slots=True)
class PlannerTickInput:
    """
    One runtime snapshot sent from the execution side to the planner side.

    This matches the current PlannerManager interface closely.
    """

    state: QuadrotorState | None
    timestamp: float
    active_command: QuadrotorCommand | None = None


@dataclass(slots=True)
class PlannerTickOutput:
    """
    One planner response sent back to the execution side.

    Keep this small. The execution side should mostly care about the next command.
    """

    command: QuadrotorCommand


@dataclass(slots=True)
class PlannerControlMessage:
    """
    One control-plane message for configuring the planner process.

    Example commands
    ----------------
    - "add_goal"
    - "clear_goals"
    - "reset"
    - "stop"
    """

    command: str
    payload: Any | None = None


def _put_latest(queue_obj: Queue, item: Any) -> None:
    """
    Put one item into a bounded queue, dropping the old item if needed.

    This is the key difference from a naive FIFO queue: planner input should not
    build up latency. The newest snapshot is usually the only one worth keeping.
    """

    try:
        queue_obj.put_nowait(item)
        return
    except Full:
        pass

    try:
        queue_obj.get_nowait()
    except Empty:
        pass

    try:
        queue_obj.put_nowait(item)
    except Full:
        # If another producer won the race, dropping this item is acceptable.
        pass


def _drain_latest(queue_obj: Queue) -> Any | None:
    """
    Return the newest queued item and discard older ones.
    """

    latest = None
    while True:
        try:
            latest = queue_obj.get_nowait()
        except Empty:
            break
    return latest


def build_grid_map(planning_cfg: dict[str, Any]) -> GridMap:
    """
    Reuse the same map construction as the single-process demo.

    TODO
    ----
    Later this should move into a shared helper or a map-loader module so both
    demos and real integrations configure the planner the same way.
    """

    resolution = float(planning_cfg["grid_map"]["resolution"])
    origin = np.zeros(3, dtype=np.float64)
    size = np.asarray(planning_cfg["grid_map"]["map_size"], dtype=np.int64)
    grid_map = GridMap(resolution=resolution, origin=origin, map_size=size)

    grid_map.add_obstacle(
        obs_start_index=np.array([10, 30, 0], dtype=np.int64),
        obs_size=np.array([30, 10, 30], dtype=np.int64),
    )
    grid_map.add_obstacle(
        obs_start_index=np.array([20, 10, 0], dtype=np.int64),
        obs_size=np.array([20, 10, 30], dtype=np.int64),
    )

    return grid_map


def build_planner_manager_from_project_config() -> PlannerManager:
    """
    Build planner-side objects inside the planner process.

    Important design note
    ---------------------
    For a real multicore / multiprocessing design, it is cleaner to build the
    planner inside the child process instead of constructing it in the parent and
    passing a large object graph through pickling.
    """

    quadrotor_cfg, planning_cfg = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )

    bspline_optimizer = BsplineOptimizer(
        config=build_bspline_optimizer_config(
            quadrotor_cfg=quadrotor_cfg,
            planning_cfg=planning_cfg,
        )
    )
    local_planner = LocalPlanner(
        config=build_local_planner_config(
            quadrotor_cfg=quadrotor_cfg,
            planning_cfg=planning_cfg,
        ),
        optimizer=bspline_optimizer,
    )

    planner_manager = PlannerManager(
        local_planner=local_planner,
        global_lmpc_config=build_linear_mpc_config(
            quadrotor_cfg=quadrotor_cfg,
            planning_cfg=planning_cfg,
        ),
        config=build_planner_manager_config(planning_cfg),
        grid_map=build_grid_map(planning_cfg),
    )
    return planner_manager


class PlannerProcess(Process):
    """
    Draft planner worker process.

    Responsibilities
    ----------------
    - own the PlannerManager and its long-lived planning state
    - receive configuration/control messages
    - consume the latest runtime snapshot
    - publish the newest QuadrotorCommand
    """

    def __init__(
        self,
        tick_input_queue: Queue,
        tick_output_queue: Queue,
        control_queue: Queue,
        stop_event: Event,
        idle_sleep: float = 0.01,
    ) -> None:
        super().__init__(daemon=True)
        self.tick_input_queue = tick_input_queue
        self.tick_output_queue = tick_output_queue
        self.control_queue = control_queue
        self.stop_event = stop_event
        self.idle_sleep = float(idle_sleep)

    def run(self) -> None:
        planner_manager = build_planner_manager_from_project_config()

        while not self.stop_event.is_set():
            self._handle_control_messages(planner_manager)

            tick_input = _drain_latest(self.tick_input_queue)
            if tick_input is None:
                time.sleep(self.idle_sleep)
                continue

            if not isinstance(tick_input, PlannerTickInput):
                continue

            command = planner_manager.update_and_get_command(
                current_state=tick_input.state,
                current_time=tick_input.timestamp,
                active_command=tick_input.active_command,
            )
            _put_latest(
                self.tick_output_queue,
                PlannerTickOutput(command=command),
            )

    def _handle_control_messages(self, planner_manager: PlannerManager) -> None:
        while True:
            try:
                message = self.control_queue.get_nowait()
            except Empty:
                break

            if not isinstance(message, PlannerControlMessage):
                continue

            if message.command == "add_goal":
                planner_manager.add_goal(np.asarray(message.payload, dtype=np.float64))
            elif message.command == "clear_goals":
                planner_manager.clear_goals()
            elif message.command == "reset":
                planner_manager.clear_goals()
            elif message.command == "stop":
                self.stop_event.set()
            else:
                # TODO: add "set_map", "replace_goals", "update_config" when needed.
                pass


def build_execution_side() -> tuple[SimStateProvider, SimBackend, QuadrotorAgent]:
    """
    Build the execution-side stack in the parent process.

    This stays close to run_sim_demo.py on purpose. The point of this draft is
    to show the process boundary, not to redesign the whole execution stack yet.
    """

    quadrotor_cfg, _ = load_project_configs(
        quadrotor_yaml_path=PROJECT_ROOT / "configs" / "quadrotor.yaml",
        planning_yaml_path=PROJECT_ROOT / "configs" / "planning.yaml",
    )

    initial_state = QuadrotorState(
        pva=np.array(
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
    )
    state_provider = SimStateProvider(initial_state=initial_state)
    backend = SimBackend(
        state_provider=state_provider,
        max_acceleration_norm=float(
            np.max(
                np.asarray(
                    quadrotor_cfg["limits"]["max_acceleration_3d"],
                    dtype=np.float64,
                )
            )
        ),
    )
    controller = Controller(
        kp_position=2.0,
        kv_velocity=1.5,
        max_acceleration_norm=float(
            np.max(
                np.asarray(
                    quadrotor_cfg["limits"]["max_acceleration_3d"],
                    dtype=np.float64,
                )
            )
        ),
    )
    agent = QuadrotorAgent(
        controller=controller,
        state_provider=state_provider,
        backend=backend,
    )
    return state_provider, backend, agent


def main() -> None:
    """
    Draft parent process orchestration.

    Current intention
    -----------------
    - planner runs in its own process
    - simulator / agent loop runs in the parent process
    - both sides communicate through typed queues

    This is intentionally a draft. It shows structure and queue ownership, but
    leaves room for the next refactor step:
    - move PlannerTickInput / PlannerTickOutput into src/trajplan/planning
    - clean up execution-side interfaces for real-drone adapters
    """

    tick_input_queue: Queue = Queue(maxsize=1)
    tick_output_queue: Queue = Queue(maxsize=1)
    control_queue: Queue = Queue(maxsize=32)
    stop_event = Event()

    planner_process = PlannerProcess(
        tick_input_queue=tick_input_queue,
        tick_output_queue=tick_output_queue,
        control_queue=control_queue,
        stop_event=stop_event,
    )
    planner_process.start()

    state_provider, _backend, agent = build_execution_side()

    # Example mission configuration sent through the planner control plane.
    control_queue.put(
        PlannerControlMessage(
            command="add_goal",
            payload=np.array([4.5, 4.5, 1.0], dtype=np.float64),
        )
    )
    control_queue.put(
        PlannerControlMessage(
            command="add_goal",
            payload=np.array([1.0, 2.0, 0.5], dtype=np.float64),
        )
    )

    dt_exec = 0.01
    dt_plan = 1.0
    sim_time = 0.0
    next_plan_time = 0.0
    max_sim_time = 30.0

    try:
        while not agent.is_finished and sim_time <= max_sim_time:
            current_state = state_provider.get_state()

            if sim_time >= next_plan_time - 1e-12:
                _put_latest(
                    tick_input_queue,
                    PlannerTickInput(
                        state=current_state,
                        timestamp=sim_time,
                        active_command=agent.active_command,
                    ),
                )
                next_plan_time += dt_plan

            planner_output = _drain_latest(tick_output_queue)
            if isinstance(planner_output, PlannerTickOutput):
                agent.set_active_command(planner_output.command)

            agent.step(dt=dt_exec, now_time=sim_time)
            sim_time += dt_exec

    finally:
        stop_event.set()
        _put_latest(
            control_queue,
            PlannerControlMessage(command="stop"),
        )
        planner_process.join(timeout=2.0)


if __name__ == "__main__":
    main()
