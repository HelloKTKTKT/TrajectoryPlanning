from __future__ import annotations

import time
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing.synchronize import Event as ProcessEvent
from typing import Mapping

import numpy as np

from trajplan.config import SwarmTargetScenario
from trajplan.planning.messages import TargetStateMessage
from trajplan.runtime.ipc import put_latest
from trajplan.visualization.messages import TargetVisualizationSnapshot


class TargetProcess(Process):
    def __init__(
        self,
        target_scenario: SwarmTargetScenario,
        outbound_queues: Mapping[int, Queue[TargetStateMessage]],
        stop_event: ProcessEvent,
        visualization_queue: Queue[TargetVisualizationSnapshot] | None = None,
    ) -> None:
        super().__init__()
        self.target_scenario = target_scenario
        self.outbound_queues = dict(outbound_queues)
        self.stop_event = stop_event
        self.visualization_queue = visualization_queue

    def run(self) -> None:
        update_interval = max(float(self.target_scenario.update_interval), 1e-3)
        waypoints = [
            np.asarray(self.target_scenario.initial_position, dtype=np.float64).reshape(3),
            *[
                np.asarray(waypoint, dtype=np.float64).reshape(3)
                for waypoint in self.target_scenario.waypoint_positions
            ],
        ]
        if len(waypoints) == 1:
            segment_lengths = np.zeros(0, dtype=np.float64)
            total_length = 0.0
        else:
            segment_lengths = np.asarray(
                [
                    np.linalg.norm(waypoints[i + 1] - waypoints[i])
                    for i in range(len(waypoints) - 1)
                ],
                dtype=np.float64,
            )
            total_length = float(np.sum(segment_lengths))

        speed = max(float(self.target_scenario.speed), 0.0)
        start_time = time.monotonic()

        while not self.stop_event.is_set():
            now_time = time.monotonic()
            elapsed = max(now_time - start_time, 0.0)
            position, velocity = self._sample_target_state(
                elapsed=elapsed,
                waypoints=waypoints,
                segment_lengths=segment_lengths,
                total_length=total_length,
                speed=speed,
            )

            message = TargetStateMessage(
                timestamp=now_time,
                position=position,
                velocity=velocity,
            )
            for queue_obj in self.outbound_queues.values():
                put_latest(queue_obj, message.copy())

            if self.visualization_queue is not None:
                put_latest(
                    self.visualization_queue,
                    TargetVisualizationSnapshot(
                        timestamp=now_time,
                        position=position.copy(),
                        velocity=velocity.copy(),
                    ),
                )

            time.sleep(update_interval)

    @staticmethod
    def _sample_target_state(
        elapsed: float,
        waypoints: list[np.ndarray],
        segment_lengths: np.ndarray,
        total_length: float,
        speed: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(waypoints) == 1 or speed <= 1e-9 or total_length <= 1e-9:
            return waypoints[0].copy(), np.zeros(3, dtype=np.float64)

        traveled = (elapsed * speed) % total_length
        remaining = traveled
        for segment_idx, segment_length in enumerate(segment_lengths):
            if remaining <= segment_length or segment_idx == len(segment_lengths) - 1:
                start = waypoints[segment_idx]
                end = waypoints[segment_idx + 1]
                if segment_length <= 1e-9:
                    return start.copy(), np.zeros(3, dtype=np.float64)
                alpha = remaining / segment_length
                position = (1.0 - alpha) * start + alpha * end
                direction = (end - start) / segment_length
                velocity = direction * speed
                return position.astype(np.float64), velocity.astype(np.float64)
            remaining -= float(segment_length)

        return waypoints[-1].copy(), np.zeros(3, dtype=np.float64)
