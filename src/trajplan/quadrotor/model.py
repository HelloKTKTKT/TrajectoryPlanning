from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajplan.shared_types import Matrix, Vector


@dataclass(slots=True)
class QuadrotorPhysicalConfig:
    rotor_count: int

    mass: float
    arm_length: float
    motor_weight: float

    body_box_size: Vector
    body_real_z: float
    align_angle_rad: float

    air_density: float
    propeller_diameter: float
    thrust_coefficient: float
    power_coefficient: float
    max_rpm: float

    max_rps: float
    max_force: float
    max_rotor_torque: float
    torque_force_ratio: float

    box_mass: float
    quad_poses: Matrix

    inertia_matrix: Matrix
    inertia_inv: Matrix

    rforce2pseudo: Matrix
    pseudo2rforce: Matrix

    def __post_init__(self) -> None:
        self.rotor_count = int(self.rotor_count)

        self.mass = float(self.mass)
        self.arm_length = float(self.arm_length)
        self.motor_weight = float(self.motor_weight)

        self.body_box_size = np.asarray(
            self.body_box_size,
            dtype=np.float64,
        ).reshape(-1)
        self.body_real_z = float(self.body_real_z)
        self.align_angle_rad = float(self.align_angle_rad)

        self.air_density = float(self.air_density)
        self.propeller_diameter = float(self.propeller_diameter)
        self.thrust_coefficient = float(self.thrust_coefficient)
        self.power_coefficient = float(self.power_coefficient)
        self.max_rpm = float(self.max_rpm)

        self.max_rps = float(self.max_rps)
        self.max_force = float(self.max_force)
        self.max_rotor_torque = float(self.max_rotor_torque)
        self.torque_force_ratio = float(self.torque_force_ratio)

        self.box_mass = float(self.box_mass)

        self.quad_poses = np.asarray(self.quad_poses, dtype=np.float64)
        self.inertia_matrix = np.asarray(self.inertia_matrix, dtype=np.float64)
        self.inertia_inv = np.asarray(self.inertia_inv, dtype=np.float64)
        self.rforce2pseudo = np.asarray(self.rforce2pseudo, dtype=np.float64)
        self.pseudo2rforce = np.asarray(self.pseudo2rforce, dtype=np.float64)

        if self.rotor_count <= 0:
            raise ValueError(f"Expected rotor_count > 0, but got {self.rotor_count}.")
        if self.mass <= 0.0:
            raise ValueError(f"Expected mass > 0, but got {self.mass}.")
        if self.arm_length <= 0.0:
            raise ValueError(f"Expected arm_length > 0, but got {self.arm_length}.")
        if self.motor_weight < 0.0:
            raise ValueError(
                f"Expected motor_weight >= 0, but got {self.motor_weight}."
            )
        if self.body_box_size.shape != (3,):
            raise ValueError(
                f"Expected body_box_size shape (3,), but got {self.body_box_size.shape}."
            )
        if np.any(self.body_box_size <= 0.0):
            raise ValueError(
                f"Expected positive body_box_size entries, but got {self.body_box_size}."
            )
        if self.air_density <= 0.0:
            raise ValueError(f"Expected air_density > 0, but got {self.air_density}.")
        if self.propeller_diameter <= 0.0:
            raise ValueError(
                f"Expected propeller_diameter > 0, but got {self.propeller_diameter}."
            )
        if self.thrust_coefficient <= 0.0:
            raise ValueError(
                f"Expected thrust_coefficient > 0, but got {self.thrust_coefficient}."
            )
        if self.power_coefficient <= 0.0:
            raise ValueError(
                f"Expected power_coefficient > 0, but got {self.power_coefficient}."
            )
        if self.max_rpm <= 0.0:
            raise ValueError(f"Expected max_rpm > 0, but got {self.max_rpm}.")
        if self.max_rps <= 0.0:
            raise ValueError(f"Expected max_rps > 0, but got {self.max_rps}.")
        if self.max_force <= 0.0:
            raise ValueError(f"Expected max_force > 0, but got {self.max_force}.")
        if self.max_rotor_torque <= 0.0:
            raise ValueError(
                f"Expected max_rotor_torque > 0, but got {self.max_rotor_torque}."
            )
        if self.torque_force_ratio <= 0.0:
            raise ValueError(
                f"Expected torque_force_ratio > 0, but got {self.torque_force_ratio}."
            )
        if self.box_mass <= 0.0:
            raise ValueError(f"Expected box_mass > 0, but got {self.box_mass}.")
        if self.quad_poses.shape != (self.rotor_count, 3):
            raise ValueError(
                "Expected quad_poses shape "
                f"({self.rotor_count}, 3), but got {self.quad_poses.shape}."
            )
        if self.inertia_matrix.shape != (3, 3):
            raise ValueError(
                "Expected inertia_matrix shape (3, 3), "
                f"but got {self.inertia_matrix.shape}."
            )
        if self.inertia_inv.shape != (3, 3):
            raise ValueError(
                f"Expected inertia_inv shape (3, 3), but got {self.inertia_inv.shape}."
            )
        if self.rforce2pseudo.shape != (4, self.rotor_count):
            raise ValueError(
                "Expected rforce2pseudo shape "
                f"(4, {self.rotor_count}), but got {self.rforce2pseudo.shape}."
            )
        if self.pseudo2rforce.shape != (self.rotor_count, 4):
            raise ValueError(
                "Expected pseudo2rforce shape "
                f"({self.rotor_count}, 4), but got {self.pseudo2rforce.shape}."
            )


def compute_inertia_matrix(
    box_x: float,
    box_y: float,
    box_z: float,
    box_mass: float,
    motor_weight: float,
    quad_poses: Matrix,
) -> Matrix:
    """
    Compute the rigid-body inertia matrix of the quadrotor.

    Model
    -----
    - central body is approximated as a box
    - each motor assembly is approximated as a point mass located at quad_poses[i]
    """
    box_x = float(box_x)
    box_y = float(box_y)
    box_z = float(box_z)
    box_mass = float(box_mass)
    motor_weight = float(motor_weight)
    quad_poses = np.asarray(quad_poses, dtype=np.float64)

    if box_x <= 0.0 or box_y <= 0.0 or box_z <= 0.0:
        raise ValueError(
            f"Expected positive box dimensions, but got {(box_x, box_y, box_z)}."
        )
    if box_mass <= 0.0:
        raise ValueError(f"Expected box_mass > 0, but got {box_mass}.")
    if motor_weight < 0.0:
        raise ValueError(f"Expected motor_weight >= 0, but got {motor_weight}.")
    if quad_poses.ndim != 2 or quad_poses.shape[1] != 3:
        raise ValueError(
            f"Expected quad_poses shape (N, 3), but got {quad_poses.shape}."
        )

    inertia = np.zeros((3, 3), dtype=np.float64)

    inertia[0, 0] = box_mass / 12.0 * (box_y**2 + box_z**2)
    inertia[1, 1] = box_mass / 12.0 * (box_x**2 + box_z**2)
    inertia[2, 2] = box_mass / 12.0 * (box_x**2 + box_y**2)

    for i in range(quad_poses.shape[0]):
        inertia[0, 0] += (quad_poses[i, 1] ** 2 + quad_poses[i, 2] ** 2) * motor_weight
        inertia[1, 1] += (quad_poses[i, 0] ** 2 + quad_poses[i, 2] ** 2) * motor_weight
        inertia[2, 2] += (quad_poses[i, 0] ** 2 + quad_poses[i, 1] ** 2) * motor_weight

    return inertia
