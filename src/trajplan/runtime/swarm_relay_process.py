from __future__ import annotations

import time
from multiprocessing import Process, Queue
from multiprocessing.synchronize import Event as ProcessEvent

from trajplan.planning.messages import (
    NeighborTrajectoryMessage,
    is_newer_neighbor_message,
)
from trajplan.runtime.ipc import drain_all, put_latest


class SwarmRelayProcess(Process):
    def __init__(
        self,
        inbound_queue: Queue[NeighborTrajectoryMessage],
        outbound_queues: dict[int, Queue[NeighborTrajectoryMessage]],
        stop_event: ProcessEvent,
        idle_sleep_time: float,
    ) -> None:
        super().__init__()
        self.inbound_queue = inbound_queue
        self.outbound_queues = outbound_queues
        self.stop_event = stop_event
        self.idle_sleep_time = float(idle_sleep_time)

    def run(self) -> None:
        while not self.stop_event.is_set():
            messages = drain_all(self.inbound_queue)
            if len(messages) == 0:
                time.sleep(self.idle_sleep_time)
                continue

            for message in self._compress_latest_messages(messages):
                self._forward_message(message)

    @staticmethod
    def _compress_latest_messages(
        messages: list[object],
    ) -> list[NeighborTrajectoryMessage]:
        latest_messages_by_agent: dict[int, NeighborTrajectoryMessage] = {}
        for message in messages:
            if not isinstance(message, NeighborTrajectoryMessage):
                continue
            current_latest = latest_messages_by_agent.get(message.agent_id)
            if current_latest is None or is_newer_neighbor_message(
                message, current_latest
            ):
                latest_messages_by_agent[message.agent_id] = message

        return list(latest_messages_by_agent.values())

    def _forward_message(self, message: NeighborTrajectoryMessage) -> None:
        forwarded_agent_ids: list[int] = []
        for target_agent_id, queue_obj in self.outbound_queues.items():
            if target_agent_id == message.agent_id:
                continue
            put_latest(queue_obj, message.copy())
            forwarded_agent_ids.append(int(target_agent_id))

        if len(forwarded_agent_ids) == 0:
            return

        print(
            "[SwarmRelayProcess] "
            f"forwarded agent={message.agent_id} traj_id={message.traj_id} "
            f"to agents={forwarded_agent_ids}.",
            flush=True,
        )
