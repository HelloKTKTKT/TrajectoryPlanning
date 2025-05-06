import numpy as np


def get_ad(ts):
    b00 = np.zeros((6, 3))
    b01 = np.eye(6)
    b10 = np.zeros((3, 3))
    b11 = np.zeros((3, 6))

    # Create two separate blocks for the first row and second row
    top_row = np.hstack((b00, b01))  # This will be 6x9
    bottom_row = np.hstack((b10, b11))  # This will be 3x9

    # Now stack them vertically
    block_matrix = np.vstack((top_row, bottom_row))  # This will be 9x9

    return np.eye(9) + block_matrix * ts


GLOBAL_MAX_DURATION = 10.0  # seconds
MPC_TS = 0.1
Ad = get_ad(ts=MPC_TS)
Bd = np.vstack((np.zeros((6, 3)), np.eye(3))) * MPC_TS
Q = np.diag([10, 10, 10, 1, 1, 1, 0.1, 0.1, 0.1])
QN = np.diag([10, 10, 10, 1, 1, 1, 0.1, 0.1, 0.1])
R = np.eye(3) * 0.01

PLANNING_HORIZON = 5.0
MAX_V = 4.0
MAX_A = 5.0
MAX_J = 10.0
MAX_V_3D = np.array([MAX_V, MAX_V, MAX_V])
MAX_A_3D = np.array([MAX_A, MAX_A, MAX_A])
MAX_J_3D = np.array([MAX_J, MAX_J, MAX_J])
FEA_TOLERANCE = 0.05

CTRL_PT_DIST = 0.4
SAFE_DIST = 0.5

# L-BFGS
LAMBDA_SMOOTH = 1e-4
LAMBDA_DIST = 10.0  # *=2 if collision still detected
LAMBDA_FEASIBILITY = 1e-3
LAMBDA_FITNESS = 5.0
MAX_RESART_NUMS_SET = 3
TOL = 1e-5
