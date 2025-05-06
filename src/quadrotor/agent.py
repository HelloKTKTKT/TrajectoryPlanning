import time
import numpy as np
import src.quadrotor.dynamics as quad_dynamics
from src.quadrotor.controller import DifferentialFlatness
from src.planner.planner_manager import PlannerManager


class QuadrotorAgent:
    def __init__(self, name, world_grid_map):
        self.name = name
        self.world_grid_map = world_grid_map
        self.dynamics = quad_dynamics.rk4  # (xin, uin, delta_t) --> xnext
        self.controller = DifferentialFlatness()
        self.xc = np.zeros((12, 1))
        self.uc = np.ones((4, 1)) * (9.81 / 4)
        self.tc = 0.0
        self.start_time = 0.0
        self.flight_log = {
            "xlog": [],
            "ulog": [],
            "tlog": []
        }
        self.global_target_pva = np.zeros((9, 1))
        self.planner_manager = PlannerManager(self.world_grid_map)
        self.lbfgs_flag_log = []
        self.xref = []

    def spawn(self, x0_pva, global_target_pva, start_time):
        x0_pva.shape = (-1, 1)
        global_target_pva.shape = (-1, 1)
        self.xc[:6] = x0_pva[:6]
        self.start_time = start_time
        self.flight_log["xlog"].append(self.xc)
        self.flight_log["ulog"].append(self.uc)
        self.flight_log["tlog"].append(self.tc)
        self.global_target_pva = global_target_pva
        self.planner_manager.generate_global_trajectory(x0_pva, global_target_pva, self.tc)
        self.planner_manager.generate_local_trajectory(type_start="ODOM", flag_new_traj=True, xc_pva=x0_pva,
                                                       start_time=0.0)

    def run(self):
        time_accu = 0.0
        while True:
            t_start = time.time()

            local_current_progress = self.planner_manager.local_trajectory.current_progress
            xref = self.planner_manager.local_trajectory.evaluate_pva(local_current_progress)
            self.xref.append(xref)

            xref_flat = np.zeros((4, 5))
            xref_flat[:3, :3] = xref.reshape((3, 3), order='F')
            u_apply = self.controller.execute(self.xc, self.uc, xref_flat)
            xnext = self.dynamics(self.xc, u_apply, 0.05)

            self.xc = xnext
            self.uc = u_apply
            self.tc += 0.05
            self.flight_log["xlog"].append(self.xc)
            self.flight_log["ulog"].append(self.uc)
            self.flight_log["tlog"].append(self.tc)

            self.planner_manager.local_trajectory.current_progress += 0.05
            xref_local_pva = self.planner_manager.local_trajectory.evaluate_pva(
                self.planner_manager.local_trajectory.current_progress)
            flag = self.planner_manager.generate_local_trajectory(type_start="LOCAL",
                                                                  flag_new_traj=False,
                                                                  xc_pva=xref_local_pva, start_time=0.0)
            self.lbfgs_flag_log.append(flag)

            if np.linalg.norm(self.xc[:6] - self.global_target_pva[:6]) < 0.2 or self.tc > 10.0:
                break

            delta_t = time.time() - t_start
            print(f'Single simulation step time (include everything): {(delta_t * 1_000):.3f} ms')

            time_accu += delta_t
        print(f'Use {time_accu:.3f}s to simulate {self.tc:.3f}s')