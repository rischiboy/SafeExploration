import numpy as np


class PendulumConsts:

    # Dynamics
    G = 10.0
    M = 1.0
    L = 1.0
    B = 0.1
    DT = 0.05

    MAX_TORQUE = 2.0
    MAX_THETA = np.pi
    MIN_THETA = -np.pi
    MAX_ACTION = 2.0
    MIN_ACTION = -2.0

    # Reward
    TARGET_ANGLE = 0.0
    ANGLE_COST = 1.0
    CONTROL_COST = 0.001

    # Constraint
    MAX_SPEED = 8.0
    CONSTRAINED_SPEED = 6.0
    CONSTRAINED_MAX_SPEED = 12.0

    # Termination
    ANGLE_TOLERANCE = 0.1
    STABILITY_DURATION = 10
    MAX_STEPS = 200
    NOISE_SCALE = 0.1  # Reset noise scale


class PitchControlConsts:
    # Dynamics
    OMEGA = 56.7
    CLD = 0.313
    CMLD = 0.0139
    CW = 0.232
    CM = 0.426
    ETA = 0.0875
    STEP_SIZE = 0.05

    MAX_ACTION = 1.4
    MIN_ACTION = -1.4

    # Reward
    DESIRED_ANGLE = 0.0
    ACTION_COST = 0.02
    PITCH_COST = 2.0

    # Constraint
    MAX_ANGLE = 0.0

    # Termination
    INIT_ANGLE = -0.2
    ANGLE_TOLERANCE = 0.025
    STABILITY_DURATION = 20
    MAX_STEPS = 200
    NOISE_SCALE = 0.001  # Reset noise scale


class CarParkConsts:
    # Dynamics
    DT = 0.05
    G = 9.81
    M = 1.0
    MU_STATIC = 0.4  # Static friction coefficient
    MU_KINETIC = 0.2  # Kinetic friction coefficient

    MAX_ACTION = 6.0
    MIN_ACTION = -6.0

    MAX_POSITION = 5.0
    MIN_POSITION = -5.0

    START = 0.0  # [meters]

    # Reward
    DESTINATION = 3.0  # [meters]
    BOTTOM_MARGIN = -0.05
    TOP_MARGIN = 0.05

    REWARD_SCALE = 10.0
    REWARD_MARGIN = 0.05
    VALUE_AT_MARGIN = 0.95

    SPEED_COST = 0.01
    ACTION_COST = 0.001

    # Constraint
    MAX_SPEED = 10.0
    ROAD_LENGTH = 4.0

    # Termination
    STABILITY_DURATION = 10
    MAX_STEPS = 200
    NOISE_SCALE = 0.1  # Reset noise scale
