import time
from multiprocessing import Process, Queue
from queue import Empty, Full
from multiprocessing.synchronize import (
    Event as ProcessEvent,
)  # note this is only for annotation, when creating event, use from multiprocessing import Event
from trajplan.planning.messages import PlannerTickInput, PlannerTickOutput
from trajplan.runtime.channels import PlannerManagerAgentQueues
from trajplan.planning.planner_manager import PlannerManager
from trajplan.quadrotor.command import QuadrotorMode
from typing import Any


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
        last_plan_time = time.monotonic()

        while not self.stop_event.is_set():  # when existing this while loop, this process will be not alive: planner_process.is_alive() == False
            tick_input = _drain_latest(self.planner_manager_agent_queues.agent_to_pm)
            current_time = time.monotonic()

            if (
                isinstance(tick_input, PlannerTickInput)
                and (current_time - last_plan_time) >= self.planning_interval
            ):
                command = self.planner_manager.update_and_get_command(
                    current_state=tick_input.state,
                    current_time=tick_input.timestamp,
                    active_command=tick_input.active_command,
                )
                tick_output = PlannerTickOutput(command=command)
                _put_latest(self.planner_manager_agent_queues.pm_to_agent, tick_output)
                if command.mode == QuadrotorMode.TRACK:
                    last_plan_time = current_time
                elif command.mode == QuadrotorMode.FINISH:
                    self.stop_event.set()
                    break
                else:
                    time.sleep(self.idle_sleep_time)
            else:
                time.sleep(self.idle_sleep_time)


def _put_latest(queue_obj, item):
    try:
        queue_obj.put(item, block=False)
    except Full:
        try:
            queue_obj.get(block=False)
        except Empty:
            pass
        #  since the consumer does not put any information inside, so theoretically upto here, the queue should never be Full
        try:
            queue_obj.put(item, block=False)
        except Full:
            pass


def _drain_latest(queue_obj: Queue) -> Any | None:
    """
    Return the newest queued item and discard older ones.
    """

    latest = None
    while True:
        try:
            latest = queue_obj.get(block=False)
        except Empty:
            break
    return latest
