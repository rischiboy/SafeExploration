import jax.numpy as jnp
from jax.lax import scan, cond
from typing import Callable


def runge_kutta_step(f, y, t, h):
    k1 = h * f(t, y)

    k2 = h * f(t + h / 4, y + k1 / 4)
    k3 = h * f(t + 3 * h / 8, y + 3 * k1 / 32 + 9 * k2 / 32)
    k4 = h * f(
        t + 12 * h / 13, y + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197
    )
    k5 = h * f(
        t + h,
        y + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104,
    )
    k6 = h * f(
        t + h / 2,
        y - 8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40,
    )

    y_next = y + 25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5
    y_next_hat = (
        y
        + 16 * k1 / 135
        + 6656 * k3 / 12825
        + 28561 * k4 / 56430
        - 9 * k5 / 50
        + 2 * k6 / 55
    )

    return y_next, y_next_hat


def runge_kutta_45(
    func: Callable, t0: float, y0: jnp.ndarray, step_size: float, num_steps: int
):
    def body(carry, _):
        t, y = carry
        y_next, y_next_hat = runge_kutta_step(func, y, t, step_size)

        # Do not require to update time since ODE does not depend on time.
        # t = t + step_size

        carry = (t, y_next)
        outs = y_next

        return carry, outs

    init = (t0, y0)
    _, y_next = scan(body, init, xs=None, length=num_steps)

    return y_next[-1]


# rk_coeffs = jnp.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])

# butcher_table = jnp.array(
#     [
#         [0, 0, 0, 0, 0, 0],
#         [1 / 4, 0, 0, 0, 0, 0],
#         [3 / 32, 9 / 32, 0, 0, 0, 0],
#         [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
#         [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
#         [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
#     ]
# )

# weights = jnp.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
# derivative_weights = jnp.array(
#     [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]
# )

# def runge_kutta_step(carry, ins):
#     t, y, k_array, it = carry
#     # y = y + jnp.dot(butcher_table[it], k_array)
#     y_i = y + jnp.dot(ins, k_array)
#     t_i = t + rk_coeffs[it] * step_size
#     k = step_size * func(y_i, t_i)
#     k_array = k_array.at[it].set(k)

#     carry = (t, y, k_array, it + 1)
#     outs = k_array

#     return carry, outs
