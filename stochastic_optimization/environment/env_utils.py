from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

""" RCCar helper functions """


def encode_angles_numpy(state: np.array, angle_idx: int) -> np.array:
    """Encodes the angle (theta) as sin(theta) and cos(theta)"""
    assert angle_idx <= state.shape[-1] - 1
    theta = state[..., angle_idx : angle_idx + 1]
    state_encoded = np.concatenate(
        [
            state[..., :angle_idx],
            np.sin(theta),
            np.cos(theta),
            state[..., angle_idx + 1 :],
        ],
        axis=-1,
    )
    assert state_encoded.shape[-1] == state.shape[-1] + 1
    return state_encoded


def encode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """Encodes the angle (theta) as sin(theta) and cos(theta)"""
    assert angle_idx <= state.shape[-1] - 1
    theta = state[..., angle_idx : angle_idx + 1]
    state_encoded = jnp.concatenate(
        [
            state[..., :angle_idx],
            jnp.sin(theta),
            jnp.cos(theta),
            state[..., angle_idx + 1 :],
        ],
        axis=-1,
    )
    assert state_encoded.shape[-1] == state.shape[-1] + 1
    return state_encoded


def decode_angles_numpy(state: np.array, angle_idx: int) -> np.array:
    """Decodes the angle (theta) from sin(theta) and cos(theta)"""
    assert angle_idx < state.shape[-1] - 1
    theta = np.arctan2(
        state[..., angle_idx : angle_idx + 1], state[..., angle_idx + 1 : angle_idx + 2]
    )
    state_decoded = np.concatenate(
        [state[..., :angle_idx], theta, state[..., angle_idx + 2 :]], axis=-1
    )
    assert state_decoded.shape[-1] == state.shape[-1] - 1
    return state_decoded


def decode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """Decodes the angle (theta) from sin(theta) and cos(theta)"""
    assert angle_idx < state.shape[-1] - 1
    theta = jnp.arctan2(
        state[..., angle_idx : angle_idx + 1], state[..., angle_idx + 1 : angle_idx + 2]
    )
    state_decoded = jnp.concatenate(
        [state[..., :angle_idx], theta, state[..., angle_idx + 2 :]], axis=-1
    )
    assert state_decoded.shape[-1] == state.shape[-1] - 1
    return state_decoded


def rotate_coordinates(state: jnp.array, encode_angle: bool = False) -> jnp.array:
    x_pos, x_vel = (
        state[..., 0:1],
        state[..., 3 + int(encode_angle) : 4 + int(encode_angle)],
    )
    y_pos, y_vel = (
        state[..., 1:2],
        state[:, 4 + int(encode_angle) : 5 + int(encode_angle)],
    )
    theta = state[..., 2 : 3 + int(encode_angle)]
    new_state = jnp.concatenate(
        [y_pos, -x_pos, theta, y_vel, -x_vel, state[..., 5 + int(encode_angle) :]],
        axis=-1,
    )
    assert state.shape == new_state.shape
    return new_state


def plot_rc_trajectory(
    traj: jnp.array,
    actions: Optional[jnp.array] = None,
    box: Optional[jnp.array] = None,  # [x, y, w, h]
    pos_domain_size: float = 5,
    show: bool = True,
    encode_angle: bool = False,
):
    """Plots the trajectory of the RC car"""
    if encode_angle:
        traj = decode_angles(traj, 2)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    scale_factor = 1.5
    if actions is None:
        fig, axes = plt.subplots(
            nrows=2, ncols=3, figsize=(scale_factor * 12, scale_factor * 8)
        )
    else:
        fig, axes = plt.subplots(
            nrows=2, ncols=4, figsize=(scale_factor * 16, scale_factor * 8)
        )
    axes[0][0].set_xlim(-pos_domain_size, pos_domain_size)
    axes[0][0].set_ylim(-pos_domain_size, pos_domain_size)
    axes[0][0].scatter(0, 0)
    # axes[0][0].plot(traj[:, 0], traj[:, 1])
    axes[0][0].set_title("x-y")

    # chaange x -> -y and y -> x
    traj = rotate_coordinates(traj, encode_angle=False)

    # Plot the velocity of the car as vectors
    total_vel = jnp.sqrt(traj[:, 3] ** 2 + traj[:, 4] ** 2)
    axes[0][0].quiver(
        traj[0:-1:3, 0],
        traj[0:-1:3, 1],
        traj[0:-1:3, 3],
        traj[0:-1:3, 4],
        total_vel[0:-1:3],
        cmap="jet",
        scale=20,
        headlength=2,
        headaxislength=2,
        headwidth=2,
        linewidth=0.2,
    )

    # Constraint box
    if box is not None:
        x, y, w, h = box
        rectangle = patches.Rectangle(
            (x, y), w, h, linewidth=1, edgecolor="r", facecolor="none"
        )
        axes[0][0].add_patch(rectangle)

    t = jnp.arange(traj.shape[0]) / 30.0
    # theta
    axes[0][1].plot(t, traj[:, 2])
    axes[0][1].set_xlabel("time")
    axes[0][1].set_ylabel("theta")
    axes[0][1].set_title("theta")

    # angular velocity
    axes[0][2].plot(t, traj[:, -1])
    axes[0][2].set_xlabel("time")
    axes[0][2].set_ylabel("angular velocity")
    axes[0][2].set_title("angular velocity")

    axes[1][0].plot(t, total_vel)
    axes[1][0].set_xlabel("time")
    axes[1][0].set_ylabel("total velocity")
    axes[1][0].set_title("velocity")

    # vel x
    axes[1][1].plot(t, traj[:, 3])
    axes[1][1].set_xlabel("time")
    axes[1][1].set_ylabel("velocity x")
    axes[1][1].set_title("velocity x")

    axes[1][2].plot(t, traj[:, 4])
    axes[1][2].set_xlabel("time")
    axes[1][2].set_ylabel("velocity y")
    axes[1][2].set_title("velocity y")

    if actions is not None:
        # steering
        axes[0][3].plot(t[: actions.shape[0]], actions[:, 0])
        axes[0][3].set_xlabel("time")
        axes[0][3].set_ylabel("steer")
        axes[0][3].set_title("steering")

        # throttle
        axes[1][3].plot(t[: actions.shape[0]], actions[:, 1])
        axes[1][3].set_xlabel("time")
        axes[1][3].set_ylabel("throttle")
        axes[1][3].set_title("throttle")

    fig.tight_layout()
    if show:
        fig.show()
    return fig, axes
