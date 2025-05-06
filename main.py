import numpy as np
import matplotlib.pyplot as plt
from src.map_search.grid_map import GridMap
from src.quadrotor.agent import QuadrotorAgent


def generate_grid_map():
    g_map = GridMap(map_size=np.array([50, 50, 50]))
    # g_map.update_start_target_index(agent_start_index=np.array([25, 5, 3]), agent_target_index=np.array([25, 45, 3]))
    g_map.add_obstacle(obs_start_index=np.array([20, 33, 0], dtype=int),
                       obs_size=np.array([10, 7, 30], dtype=int))
    g_map.add_obstacle(obs_start_index=np.array([24, 23, 0], dtype=int),
                       obs_size=np.array([5, 5, 30], dtype=int))
    g_map.add_obstacle(obs_start_index=np.array([5, 30, 0], dtype=int),
                       obs_size=np.array([10, 5, 30], dtype=int))
    g_map.add_obstacle(obs_start_index=np.array([15, 15, 0], dtype=int),
                       obs_size=np.array([10, 5, 30], dtype=int))
    g_map.add_obstacle(obs_start_index=np.array([35, 15, 0], dtype=int),
                       obs_size=np.array([5, 20, 30], dtype=int))
    return g_map


def get_point_set(bspline_traj):
    point_set = []
    for t in np.arange(0.0, bspline_traj.duration + 1e-3, 0.05):
        pt = bspline_traj.evaluate_pva(t)[:3]
        point_set.append(pt)
    point_set = np.hstack(point_set).T
    return point_set


def plot_traj(traj, traj_ref=None):
    fig, ax = plt.subplots()
    ax.plot(traj[:, 0], traj[:, 1], label="traj", color="black")
    ax.scatter(traj[0, 0], traj[0, 1], label="start")
    ax.scatter(traj[-1, 0], traj[-1, 1], label="end")

    if traj_ref is not None:
        ax.plot(traj_ref[:, 0], traj_ref[:, 1], label="traj_ref", color="green", linestyle="--")

    # 1st obstacle
    rect1_x = [4, 6, 6, 4, 4]
    rect1_y = [6.6, 6.6, 8, 8, 6.6]
    ax.fill(rect1_x, rect1_y, facecolor='red', alpha=0.85)

    # 2nd obstacle
    rect2_x = [4.8, 5.8, 5.8, 4.8, 4.8]
    rect2_y = [4.6, 4.6, 5.6, 5.6, 4.6]
    ax.fill(rect2_x, rect2_y, facecolor='red', alpha=0.85)

    # 3rd obstacle
    rect3_x = [1, 3, 3, 1, 1]
    rect3_y = [6, 6, 7, 7, 6]
    ax.fill(rect3_x, rect3_y, facecolor='red', alpha=0.85)

    # 4th obstacle
    rect4_x = [3, 5, 5, 3, 3]
    rect4_y = [3, 3, 4, 4, 3]
    ax.fill(rect4_x, rect4_y, facecolor='red', alpha=0.85)

    # 5th obstacle
    rect5_x = [7, 8, 8, 7, 7]
    rect5_y = [3, 3, 7, 7, 3]
    ax.fill(rect5_x, rect5_y, facecolor='red', alpha=0.85)


    ax.legend()
    ax.grid(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.show()


def run():
    grid_map = generate_grid_map()
    start_p = np.array([[5.0, 1.0, 0.6]]).T
    start_state = np.vstack((start_p, np.zeros((6, 1))))
    target_p = np.array([[5.0, 9.0, 0.6]]).T
    target_state = np.vstack((target_p, np.zeros((6, 1))))
    quad_0 = QuadrotorAgent(name="quad_0", world_grid_map=grid_map)
    quad_0.spawn(x0_pva=start_state, global_target_pva=target_state, start_time=0.0)
    quad_0.run()

    xlog = quad_0.flight_log["xlog"].copy()
    xlog = np.hstack(xlog).T
    xref = np.hstack(quad_0.xref).T
    plot_traj(xlog, xref)


if __name__ == "__main__":
    run()
