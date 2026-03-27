import time
from multiprocessing import Process
from multiprocessing.synchronize import (
    Event as ProcessEvent,
)  # note this is only for annotation, when creating event, use from multiprocessing import Event
from trajplan.planning.messages import PlannerTickInput, PlannerTickOutput
from trajplan.runtime.channels import PlannerManagerAgentQueues
from trajplan.planning.planner_manager import PlannerManager
from trajplan.quadrotor.command import QuadrotorMode
from trajplan.runtime.ipc import put_latest, drain_latest


class PlannerManagerProcess(Process):
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
        planner_manager_agent_queues: PlannerManagerAgentQueues,
        planner_manager: PlannerManager,
        stop_event: ProcessEvent,
        planning_interval: float,
        idle_sleep_time: float,
    ) -> None:
        super().__init__()
        self.planner_manager = planner_manager
        self.planner_manager_agent_queues = planner_manager_agent_queues
        self.stop_event = stop_event
        self.planning_interval = planning_interval
        self.idle_sleep_time = idle_sleep_time

    def run(self) -> None:
        last_plan_time: float | None = None

        while not self.stop_event.is_set():  # when existing this while loop, this process will be not alive: planner_process.is_alive() == False
            tick_input = drain_latest(self.planner_manager_agent_queues.agent_to_pm)

            if isinstance(tick_input, PlannerTickInput) and (
                last_plan_time is None
                or (tick_input.timestamp - last_plan_time) >= self.planning_interval
            ):
                command = self.planner_manager.update_and_get_command(
                    current_state=tick_input.state,
                    current_time=tick_input.timestamp,
                    active_command=tick_input.active_command,
                )
                tick_output = PlannerTickOutput(command=command)
                put_latest(self.planner_manager_agent_queues.pm_to_agent, tick_output)
                last_plan_time = tick_input.timestamp
                if command.mode == QuadrotorMode.FINISH:
                    self.stop_event.set()
                    break
                else:
                    time.sleep(self.idle_sleep_time)
            else:
                time.sleep(self.idle_sleep_time)
