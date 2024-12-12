from functools import partial
import os
from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Union, Optional, Tuple

import jax
from jax import jit
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import random, vmap
from jaxtyping import PyTree

from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    CostModel,
    RewardModel,
    SimpleDynamicalSystem,
    SimpleDynamicsModel,
    SimpleSafeDynamicalSystem,
)
from stochastic_optimization.utils.tolerance_reward import ToleranceReward

from stochastic_optimization.environment.env_utils import (
    decode_angles,
)


DEFAULT_INIT_POSE = jnp.array([0.5, -0.5, jnp.pi])
DEFAULT_GOAL = jnp.array([0.0, 0.0, 0.0])
DEFAULT_X_BOUNDS = jnp.array([-0.25, 1.62])
DEFAULT_Y_BOUNDS = jnp.array([-1.22, 0.25])
DEFAULT_CTRL_COST_WEIGHT = 0.005
DEFAULT_DT = 1 / 15.0  # Original value: 1 / 30.0
DEFAULT_ENCODE_ANGLE = False


# class CarParams(NamedTuple):
#     """
#     d_f, d_r : Represent grip of the car. Range: [0.015, 0.025]
#     b_f, b_r: Slope of the pacejka. Range: [2.0 - 4.0].

#     delta_limit: [0.3 - 0.5] -> Limit of the steering angle.

#     c_m_1: Motor parameter. Range [0.2, 0.5]
#     c_m_1: Motor friction, Range [0.00, 0.007]
#     c_f, c_r: [1.0 2.0] # motor parameters: source https://web.mit.edu/drela/Public/web/qprop/motor1_theory.pdf,
#     https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Embedded
#     -Control-Systems/LectureNotes/6_Motor_Control.pdf # or look at:
#     https://video.ethz.ch/lectures/d-mavt/2021/spring/151-0593-00L/00718f4f-116b-4645-91da-b9482164a3c7.html :
#     lecture 2 part 2
#     c_m_1: max current of motor: [0.2 - 0.5] c_m_2: motor resistance due to shaft: [0.01 - 0.15]
#     """

#     m: Union[jax.Array, float] = jnp.array(1.65)  # [0.04, 0.08]
#     i_com: Union[jax.Array, float] = jnp.array(2.78e-05)  # [1e-6, 5e-6]
#     l_f: Union[jax.Array, float] = jnp.array(0.13)  # [0.025, 0.05]
#     l_r: Union[jax.Array, float] = jnp.array(0.17)  # [0.025, 0.05]
#     g: Union[jax.Array, float] = jnp.array(9.81)

#     d_f: Union[jax.Array, float] = jnp.array(0.02)  # [0.015, 0.025]
#     c_f: Union[jax.Array, float] = jnp.array(1.2)  # [1.0, 2.0]
#     b_f: Union[jax.Array, float] = jnp.array(2.58)  # [2.0, 4.0]

#     d_r: Union[jax.Array, float] = jnp.array(0.017)  # [0.015, 0.025]
#     c_r: Union[jax.Array, float] = jnp.array(1.27)  # [1.0, 2.0]
#     b_r: Union[jax.Array, float] = jnp.array(3.39)  # [2.0, 4.0]

#     c_m_1: Union[jax.Array, float] = jnp.array(10.431917)  # [0.2, 0.5]
#     c_m_2: Union[jax.Array, float] = jnp.array(1.5003588)  # [0.00, 0.007]
#     c_d: Union[jax.Array, float] = jnp.array(0.0)  # [0.01, 0.1]
#     steering_limit: Union[jax.Array, float] = jnp.array(0.19989373)
#     use_blend: Union[jax.Array, float] = jnp.array(
#         0.0
#     )  # 0.0 -> (only kinematics), 1.0 -> (kinematics + dynamics)

#     # parameters used to compute the blend ratio characteristics
#     blend_ratio_ub: Union[jax.Array, float] = jnp.array([0.5477225575])
#     blend_ratio_lb: Union[jax.Array, float] = jnp.array([0.4472135955])
#     angle_offset: Union[jax.Array, float] = jnp.array([0.02791893])

""" Fixed params class for the car dynamics model. """


class CarParams(NamedTuple):

    m = jnp.array(1.65)  # [0.04, 0.08]
    i_com = jnp.array(0.01)
    l_f = jnp.array(0.13)  # [0.025, 0.05]
    l_r = jnp.array(0.17)  # [0.025, 0.05]
    g = jnp.array(9.81)

    d_f = jnp.array(0.02)  # [0.015, 0.025]
    c_f = jnp.array(1.2)  # [1.0, 2.0]
    b_f = jnp.array(2.58)  # [2.0, 4.0]

    d_r = jnp.array(0.017)  # [0.015, 0.025]
    c_r = jnp.array(1.27)  # [1.0, 2.0]
    b_r = jnp.array(5.0)

    c_m_1 = jnp.array(8.0)  # [0.2, 0.5]
    c_m_2 = jnp.array(1.5)  # [0.00, 0.007]
    c_d = jnp.array(0.0)  # [0.01, 0.1]
    steering_limit = jnp.array(0.3)
    use_blend = jnp.array(0.0)

    angle_offset = jnp.array(0.00)
    blend_ratio_lb = jnp.array(0.01)
    blend_ratio_ub = jnp.array(0.01)


class DynamicsModel(ABC):
    def __init__(
        self,
        dt: float,
        x_dim: int,
        u_dim: int,
        params: PyTree,
        angle_idx: Optional[Union[int, jax.Array]] = None,
        dt_integration: float = 0.01,
    ):
        self.dt = dt
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.params = params
        self.angle_idx = angle_idx

        self.dt_integration = dt_integration
        assert dt >= dt_integration
        assert (
            dt / dt_integration - int(dt / dt_integration)
        ) < 1e-4, "dt must be multiple of dt_integration"
        self._num_steps_integrate = int(dt / dt_integration)

    def next_state(self, x: jax.Array, u: jax.Array, params: PyTree) -> jax.Array:
        def body(carry, _):
            q = carry + self.dt_integration * self.ode(carry, u, params)
            return q, None

        next_state, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        if self.angle_idx is not None:
            theta = next_state[self.angle_idx]
            sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
            next_state = next_state.at[self.angle_idx].set(
                jnp.arctan2(sin_theta, cos_theta)
            )
        return next_state

    def ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        assert x.shape[-1] == self.x_dim and u.shape[-1] == self.u_dim
        return self._ode(x, u, params)

    @abstractmethod
    def _ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        pass

    def _split_key_like_tree(self, key: jax.random.PRNGKey):
        treedef = jtu.tree_structure(self.params)
        keys = jax.random.split(key, treedef.num_leaves)
        return jtu.tree_unflatten(treedef, keys)

    def sample_params_uniform(
        self,
        key: jax.random.PRNGKey,
        sample_shape: Union[int, Tuple[int]],
        lower_bound: NamedTuple,
        upper_bound: NamedTuple,
    ):
        keys = self._split_key_like_tree(key)
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return jtu.tree_map(
            lambda key, l, u: jax.random.uniform(
                key, shape=sample_shape + l.shape, minval=l, maxval=u
            ),
            keys,
            lower_bound,
            upper_bound,
        )


class RCCarDynamics(DynamicsModel, SimpleDynamicsModel):
    """
    local_coordinates: bool
        Used to indicate if local or global coordinates shall be used.
        If local, the state x is
            x = [0, 0, theta, vel_r, vel_t, angular_velocity_z]
        else:
            x = [x, y, theta, vel_x, vel_y, angular_velocity_z]
    u = [steering_angle, throttle]
    encode_angle: bool
        Encodes angle to sin and cos if true
    """

    def __init__(
        self,
        dt=DEFAULT_DT,
        encode_angle: bool = DEFAULT_ENCODE_ANGLE,
        local_coordinates: bool = False,
        rk_integrator: bool = True,
    ):
        self.encode_angle = encode_angle
        x_dim = 6
        u_dim = 2
        super().__init__(
            dt=dt,
            x_dim=x_dim,
            u_dim=u_dim,
            params=CarParams(),
            angle_idx=2,
            dt_integration=1 / 90.0,
        )
        self.local_coordinates = local_coordinates
        self.angle_idx = 2
        self.velocity_start_idx = 4 if self.encode_angle else 3
        self.velocity_end_idx = 5 if self.encode_angle else 4
        self.rk_integrator = rk_integrator

        # Initialization as per SimpleDynamicsModel
        dim_state = x_dim + 1 if self.encode_angle else x_dim
        self.input_dim = dim_state + u_dim
        self.output_dim = dim_state
        self.default_params = CarParams()

    def rk_integration(
        self, x: jnp.array, u: jnp.array, params: CarParams
    ) -> jnp.array:
        integration_factors = jnp.asarray(
            [
                self.dt_integration / 2.0,
                self.dt_integration / 2.0,
                self.dt_integration,
                self.dt_integration,
            ]
        )
        integration_weights = jnp.asarray(
            [
                self.dt_integration / 6.0,
                self.dt_integration / 3.0,
                self.dt_integration / 3.0,
                self.dt_integration / 6.0,
            ]
        )

        def body(carry, _):
            """one step of rk integration.
            k_0 = self.ode(x, u)
            k_1 = self.ode(x + self.dt_integration / 2. * k_0, u)
            k_2 = self.ode(x + self.dt_integration / 2. * k_1, u)
            k_3 = self.ode(x + self.dt_integration * k_2, u)

            x_next = x + self.dt_integration * (k_0 / 6. + k_1 / 3. + k_2 / 3. + k_3 / 6.)
            """

            def rk_integrate(carry, ins):
                k = self.ode(carry, u, params)
                carry = carry + k * ins
                outs = k
                return carry, outs

            _, dxs = jax.lax.scan(rk_integrate, carry, xs=integration_factors, length=4)
            dx = (dxs.T * integration_weights).sum(axis=-1)
            q = carry + dx
            return q, None

        next_state, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        if self.angle_idx is not None:
            theta = next_state[self.angle_idx]
            sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
            next_state = next_state.at[self.angle_idx].set(
                jnp.arctan2(sin_theta, cos_theta)
            )
        return next_state

    def next_state(self, x: jnp.array, u: jnp.array, params: CarParams) -> jnp.array:

        if params is None:
            params = self.default_params

        theta_x = (
            jnp.arctan2(x[..., self.angle_idx], x[..., self.angle_idx + 1])
            if self.encode_angle
            else x[..., self.angle_idx]
        )
        offset = jnp.clip(params.angle_offset, -jnp.pi, jnp.pi)
        theta_x = theta_x + offset
        if not self.local_coordinates:
            # rotate velocity to local frame to compute dx
            velocity_global = x[
                ..., self.velocity_start_idx : self.velocity_end_idx + 1
            ]
            rotated_vel = self.rotate_vector(velocity_global, -theta_x)
            x = x.at[..., self.velocity_start_idx : self.velocity_end_idx + 1].set(
                rotated_vel
            )
        if self.encode_angle:
            x_reduced = self.reduce_x(x)
            if self.rk_integrator:
                x_reduced = self.rk_integration(x_reduced, u, params)
            else:
                x_reduced = super().next_state(x_reduced, u, params)
            next_theta = jnp.atleast_1d(x_reduced[..., self.angle_idx])
            next_x = jnp.concatenate(
                [
                    x_reduced[..., 0 : self.angle_idx],
                    jnp.sin(next_theta),
                    jnp.cos(next_theta),
                    x_reduced[..., self.angle_idx + 1 :],
                ],
                axis=-1,
            )
        else:
            if self.rk_integrator:
                next_x = self.rk_integration(x, u, params)
            else:
                next_x = super().next_state(x, u, params)

        if self.local_coordinates:
            # convert position to local frame
            pos = next_x[..., 0 : self.angle_idx] - x[..., 0 : self.angle_idx]
            rotated_pos = self.rotate_vector(pos, -theta_x)
            next_x = next_x.at[..., 0 : self.angle_idx].set(rotated_pos)
        else:
            # convert velocity to global frame
            new_theta_x = (
                jnp.arctan2(
                    next_x[..., self.angle_idx], next_x[..., self.angle_idx + 1]
                )
                if self.encode_angle
                else next_x[..., self.angle_idx]
            )
            new_theta_x = new_theta_x + offset
            velocity = next_x[..., self.velocity_start_idx : self.velocity_end_idx + 1]
            rotated_vel = self.rotate_vector(velocity, new_theta_x)
            next_x = next_x.at[
                ..., self.velocity_start_idx : self.velocity_end_idx + 1
            ].set(rotated_vel)

        return next_x

    def reduce_x(self, x):
        theta = jnp.arctan2(x[..., self.angle_idx], x[..., self.angle_idx + 1])

        x_reduced = jnp.concatenate(
            [
                x[..., 0 : self.angle_idx],
                jnp.atleast_1d(theta),
                x[..., self.velocity_start_idx :],
            ],
            axis=-1,
        )
        return x_reduced

    @staticmethod
    def rotate_vector(v, theta):
        v_x, v_y = v[..., 0], v[..., 1]
        rot_x = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        rot_y = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        return jnp.concatenate([jnp.atleast_1d(rot_x), jnp.atleast_1d(rot_y)], axis=-1)

    def _accelerations(self, x, u, params: CarParams):
        """Compute acceleration forces for dynamic model.
        Inputs
        -------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, velocity_r, velocity_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        acceleration: jnp.ndarray,
            shape = (3, ) -> [a_r, a_t, a_theta]
        """
        if params is None:
            params = self.default_params

        i_com = params.i_com
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        d_f = params.d_f * params.g
        d_r = params.d_r * params.g
        c_f = params.c_f
        c_r = params.c_r
        b_f = params.b_f
        b_r = params.b_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2

        c_d = params.c_d

        delta, d = u[0], u[1]

        alpha_f = -jnp.arctan((w * l_f + v_y) / (v_x + 1e-6)) + delta
        alpha_r = jnp.arctan((w * l_r - v_y) / (v_x + 1e-6))
        f_f_y = d_f * jnp.sin(c_f * jnp.arctan(b_f * alpha_f))
        f_r_y = d_r * jnp.sin(c_r * jnp.arctan(b_r * alpha_r))
        f_r_x = c_m_1 * d - (c_m_2**2) * v_x - (c_d**2) * (v_x * jnp.abs(v_x))

        v_x_dot = (f_r_x - f_f_y * jnp.sin(delta) + m * v_y * w) / m
        v_y_dot = (f_r_y + f_f_y * jnp.cos(delta) - m * v_x * w) / m
        w_dot = (f_f_y * l_f * jnp.cos(delta) - f_r_y * l_r) / i_com

        acceleration = jnp.array([v_x_dot, v_y_dot, w_dot])
        return acceleration

    def _ode_dyn(self, x, u, params: CarParams):
        """Compute derivative using dynamic model.
        Inputs
        -------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, velocity_r, velocity_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        x_dot: jnp.ndarray,
            shape = (6, ) -> time derivative of x

        """
        if params is None:
            params = self.default_params

        # state = [p_x, p_y, theta, v_x, v_y, w]. Velocities are in local coordinate frame.
        # Inputs: [\delta, d] -> \delta steering angle and d duty cycle of the electric motor.
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        p_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        theta_dot = w
        p_x_dot = jnp.array([p_x_dot, p_y_dot, theta_dot])

        accelerations = self._accelerations(x, u, params)

        x_dot = jnp.concatenate([p_x_dot, accelerations], axis=-1)
        return x_dot

    def _compute_dx_kin(self, x, u, params: CarParams):
        """Compute kinematics derivative for localized state.
        Inputs
        -----
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, v_x, v_y, w], velocities in local frame
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        dx_kin: jnp.ndarray,
            shape = (6, ) -> derivative of x

        Assumption: \dot{\delta} = 0.
        """
        if params is None:
            params = self.default_params

        p_x, p_y, theta, v_x, v_y, w = x[0], x[1], x[2], x[3], x[4], x[5]  # progress
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_d = params.c_d
        delta, d = u[0], u[1]
        v_r = v_x
        v_r_dot = (c_m_1 * d - (c_m_2**2) * v_r - (c_d**2) * (v_r * jnp.abs(v_r))) / m
        beta = jnp.arctan(jnp.tan(delta) * 1 / (l_r + l_f))
        v_x_dot = v_r_dot * jnp.cos(beta)
        # Determine accelerations from the kinematic model using FD.
        v_y_dot = (v_r * jnp.sin(beta) * l_r - v_y) / self.dt_integration
        # v_x_dot = (v_r_dot + v_y * w)
        # v_y_dot = - v_x * w
        w_dot = (jnp.sin(beta) * v_r - w) / self.dt_integration
        p_g_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_g_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        dx_kin = jnp.asarray([p_g_x_dot, p_g_y_dot, w, v_x_dot, v_y_dot, w_dot])
        return dx_kin

    def _compute_dx(self, x, u, params: CarParams):
        """Calculate time derivative of state.
        Inputs:
        ------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: jnp.ndarray, derivative of x


        If params.use_blend <= 0.5 --> only kinematic model is used, else a blend between nonlinear model
        and kinematic is used.
        """
        if params is None:
            params = self.default_params

        use_kin = params.use_blend <= 0.5
        v_x = x[3]
        blend_ratio_ub = jnp.square(params.blend_ratio_ub)
        blend_ratio_lb = jnp.square(params.blend_ratio_lb)
        blend_ratio = (v_x - blend_ratio_ub) / (blend_ratio_lb + 1e-6)
        blend_ratio = blend_ratio.squeeze()
        lambda_blend = jnp.min(jnp.asarray([jnp.max(jnp.asarray([blend_ratio, 0])), 1]))
        dx_kin_full = self._compute_dx_kin(x, u, params)
        dx_dyn = self._ode_dyn(x=x, u=u, params=params)
        dx_blend = lambda_blend * dx_dyn + (1 - lambda_blend) * dx_kin_full
        dx = (1 - use_kin) * dx_blend + use_kin * dx_kin_full
        return dx

    def _ode(self, x, u, params: CarParams):
        """
        Using kinematic model with blending: https://arxiv.org/pdf/1905.05150.pdf
        Code based on: https://github.com/alexliniger/gym-racecar/

        Inputs:
        ------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: jnp.ndarray, derivative of x
        """
        if params is None:
            params = self.default_params

        delta, d = u[0], u[1]
        delta = jnp.clip(delta, a_min=-1, a_max=1) * params.steering_limit
        d = jnp.clip(d, a_min=-1.0, a_max=1)  # throttle
        u = u.at[0].set(delta)
        u = u.at[1].set(d)
        dx = self._compute_dx(x, u, params)
        return dx


# ------------------- Reward ------------------- #


class RCCarRewardParams(NamedTuple):
    goal: jnp.array
    encode_angle: bool
    ctrl_cost_weight: jnp.array
    bound: float = jnp.array(0.1)
    margin_factor: float = jnp.array(10.0)


class RCCarEnvReward:
    _angle_idx: int = 2
    dim_action: Tuple[int] = (2,)

    def __init__(
        self,
        goal: jnp.array,
        encode_angle: bool = DEFAULT_ENCODE_ANGLE,
        ctrl_cost_weight: float = DEFAULT_CTRL_COST_WEIGHT,
        bound: float = 0.1,
        margin_factor: float = 10.0,
    ):
        self.goal = goal
        self.ctrl_cost_weight = ctrl_cost_weight
        self.encode_angle = encode_angle
        # Margin 20 seems to work even better (maybe try at some point)
        self.tolerance_reward = ToleranceReward(
            bounds=(0.0, bound),
            margin=margin_factor * bound,
            value_at_margin=0.1,
            sigmoid="long_tail",
        )

    def forward(self, obs: jnp.array, action: jnp.array, next_obs: jnp.array):
        """Computes the reward for the given transition"""
        reward_ctrl = self.action_reward(action)
        reward_state = self.state_reward(obs, next_obs)
        reward = reward_state + self.ctrl_cost_weight * reward_ctrl
        return reward

    @staticmethod
    def action_reward(action: jnp.array) -> jnp.array:
        """Computes the reward/penalty for the given action"""
        return -(action**2).sum(-1)

    def state_reward(self, obs: jnp.array, next_obs: jnp.array) -> jnp.array:
        """Computes the reward for the given observations"""
        if self.encode_angle:
            next_obs = decode_angles(next_obs, angle_idx=self._angle_idx)
        pos_diff = next_obs[..., :2] - self.goal[:2]
        theta_diff = next_obs[..., 2] - self.goal[2]
        pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_diff), axis=-1))
        theta_dist = jnp.abs(((theta_diff + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        total_dist = jnp.sqrt(pos_dist**2 + theta_dist**2)
        reward = self.tolerance_reward(total_dist)
        return reward

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)


class RCCarRewardModel(RewardModel):
    def __init__(
        self,
        goal: jnp.array = DEFAULT_GOAL,
        encode_angle: bool = DEFAULT_ENCODE_ANGLE,
        ctrl_cost_weight: float = DEFAULT_CTRL_COST_WEIGHT,
    ) -> None:
        super().__init__()

        self.default_params = RCCarRewardParams(
            goal=jnp.array(goal),
            encode_angle=encode_angle,
            ctrl_cost_weight=jnp.array(ctrl_cost_weight),
        )
        self.reward_cls = RCCarEnvReward(*self.default_params)

    def init(self, key=None):
        return self.default_params

    @partial(jit, static_argnums=(0, 4))
    def predict(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        reward_params: RCCarRewardParams = None,
    ) -> Any:

        reward = self.reward_cls.forward(obs, action, next_obs)

        return reward


# ------------------- Cost ------------------- #


class RCCarCostParams(NamedTuple):
    init_pose: jnp.array
    goal: jnp.array
    x_bounds: jnp.array
    y_bounds: jnp.array


class RCCarCost(CostModel):

    def __init__(
        self,
        init_pose: jnp.array = DEFAULT_INIT_POSE,
        goal: jnp.array = DEFAULT_GOAL,
        x_bounds: jnp.array = DEFAULT_X_BOUNDS,
        y_bounds: jnp.array = DEFAULT_Y_BOUNDS,
    ) -> None:
        super().__init__()

        self.default_params = RCCarCostParams(
            init_pose=jnp.array(init_pose),
            goal=jnp.array(goal),
            x_bounds=jnp.array(x_bounds),
            y_bounds=jnp.array(y_bounds),
        )

    def init(self, key=None):
        return self.default_params

    @staticmethod
    def get_cost_params(cost_params: RCCarCostParams):
        init_pose = cost_params.init_pose
        goal = cost_params.goal
        x_bounds = cost_params.x_bounds
        y_bounds = cost_params.y_bounds

        return (init_pose, goal, x_bounds, y_bounds)

    @partial(jit, static_argnums=(0, 4))
    def compute_violation_dist(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        cost_params: RCCarCostParams = None,
    ):

        if not cost_params:
            cost_params = self.default_params

        init_pose, goal, x_bounds, y_bounds = self.get_cost_params(cost_params)

        if next_obs.shape[-1] == 6:
            x, y, theta, velocity_r, velocity_t, angular_velocity_z = next_obs
        else:
            x, y, cos_theta, sin_theta, velocity_r, velocity_t, angular_velocity_z = (
                next_obs
            )

        curr_pos = jnp.array([x, y])

        # Bounding box around init_pose and goal with error_bound
        min_x, max_x = x_bounds
        min_y, max_y = y_bounds

        bottom_left = jnp.array([min_x, min_y])
        top_right = jnp.array([max_x, max_y])

        # Stay within bounding box
        condition = jnp.all(curr_pos >= bottom_left) & jnp.all(curr_pos <= top_right)

        def true_branch(curr_pos):
            return jnp.array(0.0)

        def false_branch(curr_pos):
            dist_x = jnp.max(jnp.array([min_x - curr_pos[0], curr_pos[0] - max_x]))
            dist_y = jnp.max(jnp.array([min_y - curr_pos[1], curr_pos[1] - max_y]))

            l_inf_dist = jnp.max(jnp.array([dist_x, dist_y]))
            l2_dist = jnp.linalg.norm(jnp.array([dist_x, dist_y]))

            return l_inf_dist

        dist = jax.lax.cond(condition, true_branch, false_branch, curr_pos)

        return dist

    def predict(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        cost_params: RCCarCostParams = None,
    ) -> jnp.ndarray:

        violation_dist = self.compute_violation_dist(obs, action, next_obs, cost_params)

        # Stay within bounding box
        condition = violation_dist == 0.0

        def true_branch(curr_pos):
            return jnp.array(0.0)

        def false_branch(curr_pos):
            return jnp.array(1.0)

        cost = jax.lax.cond(condition, true_branch, false_branch, violation_dist)

        return cost


class RCCarSystem(SimpleDynamicalSystem):
    def __init__(self, env=None, dynamics=RCCarDynamics(), reward=RCCarRewardModel()):
        super().__init__(dynamics, reward)

        self.env = env

        # For evaluation purposes
        cost_model = RCCarCost()

        self.constraint = (
            lambda x: cost_model.predict(obs=None, action=None, next_obs=x) == 1.0
        )

        self.constraint_deviation = lambda x: cost_model.compute_violation_dist(
            obs=None, action=None, next_obs=x
        )


class SafeRCCarSystem(SimpleSafeDynamicalSystem):

    def __init__(
        self,
        env=None,
        dynamics=RCCarDynamics(),
        reward=RCCarRewardModel(),
        cost=RCCarCost(),
    ):
        super().__init__(dynamics, reward, cost)

        self.env = env

        self.constraint = (
            lambda x: cost.predict(obs=None, action=None, next_obs=x) == 1.0
        )

        self.constraint_deviation = lambda x: cost.compute_violation_dist(
            obs=None, action=None, next_obs=x
        )
