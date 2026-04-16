import time
from typing import Any
import numpy as np
from multiprocessing import Process
from multiprocessing.synchronize import (
    Event as ProcessEvent,
)  # note this is only for annotation, when creating event, use from multiprocessing import Event
from trajplan.planning.messages import PlannerTickInput, PlannerTickOutput
from trajplan.runtime.channels import PlannerManagerAgentQueues
from trajplan.planning.planner_manager import PlannerManager
from trajplan.runtime.ipc import put_latest, drain_latest

from trajplan.map.grid_map import GridMap
from trajplan.config import (
    build_grid_map_config,
    build_bspline_optimizer_config,
    build_local_planner_config,
    build_planner_manager_config,
)
from trajplan.planning.bspline_optimizer import BsplineOptimizer
from trajplan.planning.local_planner import LocalPlanner


class PlannerManagerProcess(Process):
    def __init__(
        self,
        planner_manager_agent_queues: PlannerManagerAgentQueues,
        quadrotor_cfg: dict[str, Any],
        planning_cfg: dict[str, Any],
        stop_event: ProcessEvent,
        planning_interval: float,
        idle_sleep_time: float,
    ) -> None:
        super().__init__()
        self.planner_manager_agent_queues = planner_manager_agent_queues
        self.stop_event = stop_event
        self.planning_interval = planning_interval
        self.idle_sleep_time = idle_sleep_time
        self.planning_cfg = planning_cfg
        self.quadrotor_cfg = quadrotor_cfg

        # Placeholder only. The real object will be constructed in run().
        self.planner_manager: PlannerManager | None = None

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
            local_planner=local_planner,
            config=planner_manager_config,
            grid_map=grid_map,
        )
        self.planner_manager.add_goal(
            np.array([1.3, 3.0, 0.5], dtype=np.float64)
        )

        while not self.stop_event.is_set():
            tick_input = drain_latest(self.planner_manager_agent_queues.agent_to_pm)
            if not isinstance(tick_input, PlannerTickInput):
                time.sleep(self.idle_sleep_time)
                continue

            result = self.planner_manager.update_and_get_command(
                current_state=tick_input.state,
                msg_time=tick_input.timestamp,
                active_command=tick_input.active_command,
            )

            if not result.should_publish:
                time.sleep(self.idle_sleep_time)
                continue

            command = result.command
            tick_output = PlannerTickOutput(command=command)
            put_latest(self.planner_manager_agent_queues.pm_to_agent, tick_output)
            print(
                "[PlannerManagerProcess] "
                f"t={tick_input.timestamp:.3f}, "
                f"command={command.mode}",
                flush=True,
            )

            if getattr(command, "is_finish", False):
                self.stop_event.set()
                break
