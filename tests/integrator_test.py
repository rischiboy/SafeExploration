import jax
import pytest
import numpy as np
from scipy import integrate
from jax import numpy as jnp

from stochastic_optimization.utils.integrator import *
from stochastic_optimization.dynamical_system.pitch_control_system import (
    PitchControlSystem,
)
from stochastic_optimization.environment.pitch_control_env import PitchControlEnv

# ------------------- Fixtures ------------------- #


@pytest.fixture
def env():
    return PitchControlEnv()


@pytest.fixture
def system():
    return PitchControlSystem()


# Scipy integrator to compare with
@pytest.fixture
def integrator():
    return integrate.RK45


# ------------------- ODE functions ------------------- #

k = 1.0


# Define the differential equation system
def harmonic_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = -k * x
    return jnp.array([dxdt, dvdt])


# ------------------- Test parameters ------------------- #

x0 = 1.0
v0 = 0.0
y0 = np.array([x0, v0])

t0 = 0
dt = 0.1
num_steps = 1

# Set function to test
f = harmonic_oscillator

# Tolerance for comparing results
one_step_tol = 1e-6
total_tol = 5e-3

# ------------------- Tests ------------------- #


def test_runge_kutta_step(integrator):
    y_next, y_next_hat = runge_kutta_step(f, y0, t0, dt)

    integrator = integrator(
        fun=f,
        t0=t0,
        y0=y0,
        t_bound=dt,
    )

    while integrator.status == "running":
        integrator.step()
    scipy_y_next = integrator.y

    assert y_next.shape == scipy_y_next.shape
    assert np.allclose(y_next, scipy_y_next, atol=one_step_tol)


def test_runge_kutta(integrator):
    y_next = runge_kutta_45(f, t0, y0, dt, num_steps)

    t_bound = dt * num_steps
    integrator = integrator(
        fun=f,
        t0=t0,
        y0=y0,
        t_bound=t_bound,
    )

    while integrator.status == "running":
        integrator.step()
    scipy_y_next = integrator.y

    assert y_next.shape == scipy_y_next.shape
    assert np.allclose(y_next, scipy_y_next, atol=total_tol)


def test_pitch_control_dynamics(env, system):
    env = PitchControlEnv()
    obs, _ = env.reset()
    action = env.action_space.sample()

    rng = jax.random.PRNGKey(seed=313)

    # Test environment -- uses scipy integrator
    env_next_obs, env_reward, done, truncate, info = env.step(action)

    # Test system -- uses custom integrator
    sys_next_obs, sys_reward = system(obs, action, rng)

    assert np.allclose(env_next_obs, sys_next_obs, atol=total_tol)
    assert env_reward == sys_reward
