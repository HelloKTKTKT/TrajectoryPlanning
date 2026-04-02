from __future__ import annotations

import time
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing.synchronize import Event as ProcessEvent

from trajplan.planning.messages import PlannerTickInput, PlannerTickOutput
from trajplan.quadrotor.agent import QuadrotorAgent
from trajplan.runtime.channels import PlannerManagerAgentQueues
from trajplan.runtime.ipc import drain_latest, put_latest


class AgentProcess(Process):
    """
    Agent execution process.

    Responsibilities
    ----------------
    - own a long-lived QuadrotorAgent
    - receive the latest planner command
    - execute one agent step at a fixed rate
    - publish the latest state snapshot back to the planner
    """

    def __init__(
        self,
        planner_manager_agent_queues: PlannerManagerAgentQueues,
        agent: QuadrotorAgent,
        agent_stop_event: ProcessEvent,
        execution_interval: float,
        idle_sleep_time: float,
        state_log_queue: Queue | None = None,
        reference_log_queue: Queue | None = None,
    ) -> None:
        super().__init__()
        self.planner_manager_agent_queues = planner_manager_agent_queues
        self.agent = agent
        self.agent_stop_event = agent_stop_event
        self.execution_interval = float(execution_interval)
        self.idle_sleep_time = float(idle_sleep_time)
        self.state_log_queue = state_log_queue
        self.reference_log_queue = reference_log_queue

        if self.execution_interval <= 0.0:
            raise ValueError(
                f"Expected execution_interval > 0, but got {self.execution_interval}."
            )
        if self.idle_sleep_time <= 0.0:
            raise ValueError(
                f"Expected idle_sleep_time > 0, but got {self.idle_sleep_time}."
            )

    def run(self) -> None:
        last_step_time: float | None = None

        while not self.agent_stop_event.is_set():
            tick_output = drain_latest(self.planner_manager_agent_queues.pm_to_agent)
            if isinstance(tick_output, PlannerTickOutput):
                self.agent.set_active_command(tick_output.command)
                print(
                    "[AgentProcess] "
                    f"received command={tick_output.command.mode}, "
                    f"message={tick_output.command.message}",
                    flush=True,
                )

            now_time = time.monotonic()

            if last_step_time is not None:
                elapsed = now_time - last_step_time
                if elapsed < self.execution_interval:
                    time.sleep(
                        min(self.idle_sleep_time, self.execution_interval - elapsed)
                    )
                    continue

            self.agent.step(
                dt=self.execution_interval,
                now_time=now_time,
            )
            last_step_time = now_time

            tick_input = PlannerTickInput(
                state=self.agent.get_current_state(),
                timestamp=now_time,
                active_command=self.agent.active_command,
            )
            put_latest(self.planner_manager_agent_queues.agent_to_pm, tick_input)
            if self.state_log_queue is not None and tick_input.state is not None:
                self.state_log_queue.put(tick_input.state.copy())
            if self.reference_log_queue is not None:
                current_reference_pvaj = self.agent.get_current_reference_pvaj()
                if current_reference_pvaj is not None:
                    self.reference_log_queue.put(current_reference_pvaj.copy())

            if self.agent.is_finished:
                print(
                    f"[AgentProcess] finished at t={now_time:.3f}",
                    flush=True,
                )
                break

            time.sleep(self.idle_sleep_time)

        self.agent.backend.stop()
        print("[AgentProcess] backend stopped, process exiting", flush=True)
