from __future__ import annotations

import threading
import time
from multiprocessing import Process
from multiprocessing.synchronize import Event as ProcessEvent

from trajplan.crazyflie_controller import CrazyflieController
from trajplan.planning.messages import PlannerTickInput, PlannerTickOutput
from trajplan.quadrotor.agent import QuadrotorAgent
from trajplan.runtime.channels import PlannerManagerAgentQueues
from trajplan.runtime.crazyflie_backend import CrazyflieBackend
from trajplan.runtime.ipc import drain_latest, put_latest
from trajplan.runtime.mocap_state_provider import MocapStateProvider


class CrazyflieAgentProcess(Process):
    """
    Crazyflie-specific agent process.

    Runtime ROS/Crazyswarm objects are created inside ``run()`` so they never
    cross process boundaries.
    """

    def __init__(
        self,
        planner_manager_agent_queues: PlannerManagerAgentQueues,
        stop_event: ProcessEvent,
        execution_interval: float,
        idle_sleep_time: float,
        cf_index: int = 0,
        ema_alpha: float = 0.3,
        takeoff_height: float = 0.6,
        takeoff_duration: float | None = None,
        landing_height: float = 0.04,
        landing_duration: float = 2.0,
        wait_for_state_timeout: float = 10.0,
        arm_before_takeoff: bool = True,
        disarm_after_landing: bool = True,
    ) -> None:
        super().__init__()
        self.planner_manager_agent_queues = planner_manager_agent_queues
        self.stop_event = stop_event
        self.execution_interval = float(execution_interval)
        self.idle_sleep_time = float(idle_sleep_time)
        self.cf_index = int(cf_index)
        self.ema_alpha = float(ema_alpha)
        self.takeoff_height = float(takeoff_height)
        self.takeoff_duration = (
            float(takeoff_duration)
            if takeoff_duration is not None
            else float(takeoff_height + 1.5)
        )
        self.landing_height = float(landing_height)
        self.landing_duration = float(landing_duration)
        self.wait_for_state_timeout = float(wait_for_state_timeout)
        self.arm_before_takeoff = bool(arm_before_takeoff)
        self.disarm_after_landing = bool(disarm_after_landing)

        if self.execution_interval <= 0.0:
            raise ValueError(
                f"Expected execution_interval > 0, but got {self.execution_interval}."
            )
        if self.idle_sleep_time <= 0.0:
            raise ValueError(
                f"Expected idle_sleep_time > 0, but got {self.idle_sleep_time}."
            )
        if self.cf_index < 0:
            raise ValueError(f"Expected cf_index >= 0, but got {self.cf_index}.")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(f"Expected ema_alpha in (0, 1], but got {self.ema_alpha}.")
        if self.takeoff_height <= 0.0:
            raise ValueError(
                f"Expected takeoff_height > 0, but got {self.takeoff_height}."
            )
        if self.takeoff_duration <= 0.0:
            raise ValueError(
                f"Expected takeoff_duration > 0, but got {self.takeoff_duration}."
            )
        if self.landing_height < 0.0:
            raise ValueError(
                f"Expected landing_height >= 0, but got {self.landing_height}."
            )
        if self.landing_duration <= 0.0:
            raise ValueError(
                f"Expected landing_duration > 0, but got {self.landing_duration}."
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
            f"[CrazyflieAgentProcess] connected to {cf_name}",
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
        backend = CrazyflieBackend(
            cf=cf,
            landing_height=self.landing_height,
            landing_duration=self.landing_duration,
        )
        controller = CrazyflieController()
        agent = QuadrotorAgent(
            state_provider=state_provider,
            backend=backend,
            controller=controller,
        )

        try:
            if not self._wait_for_state(
                state_provider=state_provider,
                timeout_sec=self.wait_for_state_timeout,
            ):
                print(
                    "[CrazyflieAgentProcess] ERROR: no mocap state received within timeout.",
                    flush=True,
                )
                return

            initial_state = state_provider.get_state()
            print(
                "[CrazyflieAgentProcess] initial position="
                f"{initial_state.position}",
                flush=True,
            )

            try:
                if self.arm_before_takeoff:
                    try:
                        backend.arm(True)
                        time.sleep(0.5)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            "[CrazyflieAgentProcess] arm(True) failed (ignored): "
                            f"{exc}",
                            flush=True,
                        )

                print(
                    f"[CrazyflieAgentProcess] takeoff to {self.takeoff_height:.2f} m",
                    flush=True,
                )
                backend.takeoff(
                    target_height=self.takeoff_height,
                    duration=self.takeoff_duration,
                )
                time.sleep(self.takeoff_duration + 0.5)

                self._run_execution_loop(
                    agent=agent,
                    backend=backend,
                    state_provider=state_provider,
                )

            finally:
                print("[CrazyflieAgentProcess] landing ...", flush=True)
                try:
                    backend.land()
                    time.sleep(self.landing_duration + 0.5)
                finally:
                    if self.disarm_after_landing:
                        try:
                            backend.arm(False)
                        except Exception as exc:  # noqa: BLE001
                            print(
                                "[CrazyflieAgentProcess] arm(False) failed (ignored): "
                                f"{exc}",
                                flush=True,
                            )
                print("[CrazyflieAgentProcess] exiting", flush=True)
        finally:
            spin_executor.shutdown(timeout_sec=1.0)
            spin_thread.join(timeout=1.0)

    def _run_execution_loop(
        self,
        agent: QuadrotorAgent,
        backend: CrazyflieBackend,
        state_provider: MocapStateProvider,
    ) -> None:
        last_step_time: float | None = None

        while not self.stop_event.is_set():
            tick_output = drain_latest(self.planner_manager_agent_queues.pm_to_agent)
            if isinstance(tick_output, PlannerTickOutput):
                agent.try_commit_command(tick_output.command)
                print(
                    "[CrazyflieAgentProcess] "
                    f"received command={tick_output.command.mode}, "
                    f"message={tick_output.command.message}",
                    flush=True,
                )

            now_time = time.monotonic()
            dt = self.execution_interval
            if last_step_time is not None:
                elapsed = now_time - last_step_time
                if elapsed < self.execution_interval:
                    time.sleep(
                        min(self.idle_sleep_time, self.execution_interval - elapsed)
                    )
                    continue
                dt = elapsed

            agent.step(
                dt=dt,
                now_time=now_time,
            )
            last_step_time = now_time

            if agent.is_finished:
                self.stop_event.set()
                break

            tick_input = PlannerTickInput(
                state=state_provider.get_state(),
                timestamp=now_time,
                active_command=agent.get_active_command(),
            )
            put_latest(self.planner_manager_agent_queues.agent_to_pm, tick_input)

            if backend.emergency_triggered:
                print(
                    "[CrazyflieAgentProcess] backend requested emergency stop",
                    flush=True,
                )
                self.stop_event.set()
                break

        backend.stop()

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
            time.sleep(self.idle_sleep_time)
        return False
