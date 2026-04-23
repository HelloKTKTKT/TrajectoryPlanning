from __future__ import annotations

import threading
import time
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing.synchronize import Event as ProcessEvent
from typing import Mapping

from trajplan.planning.messages import TargetStateMessage
from trajplan.runtime.crazyflie.state_provider import MocapStateProvider
from trajplan.runtime.ipc import put_latest
from trajplan.visualization.messages import TargetVisualizationSnapshot


class CrazyflieTargetProcess(Process):
    """
    Read the tracked target state from a real Crazyflie via mocap/odom and
    broadcast it to every planner.
    """

    def __init__(
        self,
        outbound_queues: Mapping[int, Queue[TargetStateMessage]],
        stop_event: ProcessEvent,
        cf_index: int,
        ema_alpha: float,
        update_interval: float,
        wait_for_state_timeout: float = 10.0,
        visualization_queue: Queue[TargetVisualizationSnapshot] | None = None,
    ) -> None:
        super().__init__()
        self.outbound_queues = dict(outbound_queues)
        self.stop_event = stop_event
        self.cf_index = int(cf_index)
        self.ema_alpha = float(ema_alpha)
        self.update_interval = float(update_interval)
        self.wait_for_state_timeout = float(wait_for_state_timeout)
        self.visualization_queue = visualization_queue
        self._log_prefix = f"[CrazyflieTargetProcess cf_index={self.cf_index}]"

        if self.cf_index < 0:
            raise ValueError(f"Expected cf_index >= 0, but got {self.cf_index}.")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(f"Expected ema_alpha in (0, 1], but got {self.ema_alpha}.")
        if self.update_interval <= 0.0:
            raise ValueError(
                f"Expected update_interval > 0, but got {self.update_interval}."
            )
        if self.wait_for_state_timeout <= 0.0:
            raise ValueError(
                "Expected wait_for_state_timeout > 0, "
                f"but got {self.wait_for_state_timeout}."
            )

    def run(self) -> None:
        try:
            import rclpy
            from crazyflie_py import Crazyswarm
        except ImportError as exc:
            raise ImportError(
                "crazyflie_py is not available. Make sure the Crazyswarm2 "
                "environment is sourced inside this process."
            ) from exc

        swarm = Crazyswarm()
        cf = swarm.allcfs.crazyflies[self.cf_index]
        cf_name = cf.prefix.lstrip("/")
        ros_node = cf.node

        print(
            f"{self._log_prefix} connected to {cf_name}",
            flush=True,
        )

        spin_executor = rclpy.executors.SingleThreadedExecutor()
        spin_executor.add_node(ros_node)
        spin_thread = threading.Thread(
            target=spin_executor.spin,
            daemon=True,
            name="ros-spin",
        )
        spin_thread.start()

        state_provider = MocapStateProvider(
            node=ros_node,
            cf_name=cf_name,
            ema_alpha=self.ema_alpha,
        )

        try:
            if not self._wait_for_state(
                state_provider=state_provider,
                timeout_sec=self.wait_for_state_timeout,
            ):
                print(
                    f"{self._log_prefix} ERROR: no mocap state received within timeout.",
                    flush=True,
                )
                return

            initial_state = state_provider.get_state()
            print(
                f"{self._log_prefix} initial position={initial_state.position}",
                flush=True,
            )

            while not self.stop_event.is_set():
                now_time = time.monotonic()
                try:
                    current_state = state_provider.get_state()
                except RuntimeError:
                    time.sleep(self.update_interval)
                    continue

                message = TargetStateMessage(
                    timestamp=now_time,
                    position=current_state.position.copy(),
                    velocity=current_state.velocity.copy(),
                )
                for queue_obj in self.outbound_queues.values():
                    put_latest(queue_obj, message.copy())

                if self.visualization_queue is not None:
                    put_latest(
                        self.visualization_queue,
                        TargetVisualizationSnapshot(
                            timestamp=now_time,
                            position=current_state.position.copy(),
                            velocity=current_state.velocity.copy(),
                        ),
                    )

                time.sleep(self.update_interval)
        finally:
            print(f"{self._log_prefix} exiting", flush=True)
            spin_executor.shutdown(timeout_sec=1.0)
            spin_thread.join(timeout=1.0)

    def _wait_for_state(
        self,
        state_provider: MocapStateProvider,
        timeout_sec: float,
    ) -> bool:
        start_time = time.monotonic()
        while not self.stop_event.is_set():
            if state_provider.is_initialized():
                return True
            if time.monotonic() - start_time > timeout_sec:
                return False
            time.sleep(min(self.update_interval, 0.05))
        return False
