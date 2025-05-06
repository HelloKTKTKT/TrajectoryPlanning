import numpy as np

pi = np.pi


def compute_inertia_matrix(boxx, boxy, boxz, boxmass, motorw, quadposes):
    inertia = np.zeros((3, 3))
    inertia[0, 0] = boxmass/12 * (boxy**2 + boxz**2)
    inertia[1, 1] = boxmass/12 * (boxx**2 + boxz**2)
    inertia[2, 2] = boxmass/12 * (boxx**2 + boxy**2)
    for i in range(4):
        inertia[0, 0] += (quadposes[i, 1]**2 + quadposes[i, 2]**2) * motorw
        inertia[1, 1] += (quadposes[i, 0]**2 + quadposes[i, 2]**2) * motorw
        inertia[2, 2] += (quadposes[i, 0]**2 + quadposes[i, 1]**2) * motorw
    return inertia


# Rotor parameters (GWS 9X5, DJI Phantom 2, MT2212 motor)
max_rpm = 6396.667
max_rps = max_rpm/60
c_t = 0.109919  # @ 6396.667 RPM (MAX)
c_p = 0.040164  # @ 6396.667 RPM (MAX)
air_density = 1.225  # kg/m^3
D = 0.2286  # propeller_diameter
max_force = c_t * air_density * D**4 * max_rps**2
max_rotor_torque = c_p * air_density * D**5 * max_rps**2 / (2*pi)
c_tauf = max_rotor_torque / max_force

# Quadrotor parameter (F450 frame, MT2212 motor)
rotor_count = 4
arm_length = 0.2275
mass = 1
motor_w = 0.055  # motor assembly weight
box_mass = mass - rotor_count * motor_w
box_x, box_y, box_z, real_z = 0.18, 0.11, 0.04, 0.025
align_ang = pi/4
proj_len = arm_length * np.sin(align_ang)
quad_poses = np.array([[proj_len, proj_len, real_z],
                       [-proj_len, -proj_len, real_z],
                       [proj_len, -proj_len, real_z],
                       [-proj_len, proj_len, real_z]])
inertia_matrix = compute_inertia_matrix(box_x, box_y, box_z, box_mass, motor_w, quad_poses)
inertia_inv = np.linalg.inv(inertia_matrix)
rforce2pseudo = np.array([[1, 1, 1, 1],
                          [proj_len, -proj_len, proj_len, -proj_len],
                          [-proj_len, proj_len, proj_len, -proj_len],
                          [-c_tauf, -c_tauf, c_tauf, c_tauf]])
pseudo2rforce = np.linalg.inv(rforce2pseudo)
