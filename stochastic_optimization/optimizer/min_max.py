from functools import partial
from stochastic_optimization.optimizer.cem import CrossEntropyMethod
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.lax import scan
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union
import time

from stochastic_optimization.optimizer.icem import ICEM

DEBUG = True


class OptVarConstants(NamedTuple):
    # Constants for CEM optimizer
    action_dim: Tuple
    lower_bound: float
    upper_bound: float
    num_elites: int  # For CEM optimization
    num_fixed_elites: int  # Describe the fixed quantity in min-max optimization
    num_iter: int
    num_samples: int
    minimum: bool
    exponent: float = None  # For ICEM optimization


class OptVarParams:
    def __init__(
        self,
        consts: OptVarConstants,
        mean: jnp.ndarray = None,
        std: float = 5.0,
        fixed_elites: jnp.ndarray = None,
    ):
        if mean is None:
            mean = jnp.zeros(consts.action_dim)

        std = jnp.ones(consts.action_dim) * std

        self.mean = mean
        self.std = std

        if fixed_elites is None:
            fixed_elites = jnp.zeros((consts.num_fixed_elites, *consts.action_dim))
        self.fixed_elites = fixed_elites

        self.init_mean = mean
        self.init_std = std
        self.init_fixed_elites = fixed_elites

        self.consts = consts

        assert (
            consts.num_fixed_elites <= consts.num_elites
        ), "Number of fixed elites exceeds number of total elites."

    def reset(self):
        self.mean = self.init_mean
        self.std = self.init_std
        self.fixed_elites = self.init_fixed_elites
        return

    @property
    def get_elites(self):
        return self.fixed_elites

    def get_mean_std(self):
        return self.mean, self.std

    def get_params(self):
        dict = {"mean": self.mean, "std": self.std, "fixed_elites": self.fixed_elites}

        return dict

    def set_elites(self, elites: jnp.ndarray):
        assert elites is not None, 'Value for argument "elites" is of NoneType'

        elites_dim = (self.consts.num_fixed_elites, *self.consts.action_dim)
        self.fixed_elites = elites.reshape(elites_dim)

    def update_from_reference(self, ref: Dict):
        self.mean = ref["mean"]
        self.std = ref["std"]
        self.fixed_elites = ref["fixed_elites"]

        return

    @staticmethod
    def create_params_dict(
        mean: jnp.ndarray, std: Union[float, jnp.ndarray], fixed_elites: jnp.ndarray
    ):
        dict = {"mean": mean, "std": std, "fixed_elites": fixed_elites}

        return dict


class MinMaxOptimizer:
    def __init__(self, var_x: OptVarParams, var_y: OptVarParams, seed: int = 0):
        self.var_x = var_x
        self.var_y = var_y
        self.seed = seed

        self.x_optimizer = self._init_optimizer(var_x.consts)
        self.y_optimizer = self._init_optimizer(var_y.consts)

    def _init_optimizer(self, consts: OptVarConstants) -> CrossEntropyMethod:

        if consts.exponent is None:
            config = {
                "action_dim": consts.action_dim,
                "num_elites": consts.num_elites,
                "num_iter": consts.num_iter,
                "num_samples": consts.num_samples,
                "lower_bound": consts.lower_bound,
                "upper_bound": consts.upper_bound,
            }
            optimizer = CrossEntropyMethod(**config)
        else:
            config = {
                "action_dim": consts.action_dim,
                "num_elites": consts.num_elites,
                "num_iter": consts.num_iter,
                "num_samples": consts.num_samples,
                "exponent": consts.exponent,
                "lower_bound": consts.lower_bound,
                "upper_bound": consts.upper_bound,
            }
            optimizer = ICEM(**config)

        return optimizer

    def reset(self):
        self.var_x.reset()
        self.var_y.reset()

    @partial(jit, static_argnums=(0, 1, 3))
    def update(self, func, rng=None, iterations: int = 3):
        def step(
            func: Callable,
            optimizer: CrossEntropyMethod,
            var_param: Dict,
            consts: OptVarConstants,
            x_fixed,
            y_fixed,
            sample_key,
        ):
            if consts.minimum:
                obj_func = func
            else:
                obj_func = lambda x, y: -func(x, y)

            partial_cost_fn = get_partial_cost_fn(
                func=obj_func, x_fixed=x_fixed, y_fixed=y_fixed
            )

            mean, std = var_param["mean"], var_param["std"]

            # results = optimizer.update(partial_cost_fn, mean, std, sample_key)
            results = optimizer.update(partial_cost_fn, mean, rng=sample_key)

            best_action, best_cost, best_elites, elites_costs = results

            # Update parameters
            mean = jnp.mean(best_elites, axis=0).reshape(consts.action_dim)
            std = jnp.std(best_elites, axis=0).reshape(consts.action_dim)

            if not consts.minimum:
                best_cost = -best_cost
                elites_costs = -elites_costs

            num_fixed_elites = consts.num_fixed_elites
            fixed_elites = best_elites[:num_fixed_elites]
            fixed_elites_costs = elites_costs[:num_fixed_elites]

            updated_params = OptVarParams.create_params_dict(
                mean=mean, std=std, fixed_elites=fixed_elites
            )

            # results = (best_action, best_cost, fixed_elites, fixed_elites_costs)
            results = (best_action, best_cost)
            outs = (updated_params, results)

            return outs

        def loop_body(carry, ins):
            key, x_params, y_params = carry
            # sample_key, x_params, y_params = carry

            # Optimize y and Fix x
            x_fixed = x_params["fixed_elites"]

            key, sample_key = jax.random.split(key, 2)
            results = step(
                func=func,
                optimizer=self.y_optimizer,
                var_param=y_params,
                consts=y_consts,
                x_fixed=x_fixed,
                y_fixed=None,
                sample_key=sample_key,
            )

            y_params, opt_results = results
            best_action_y, best_cost_y = opt_results

            # Optimize x and Fix y
            y_fixed = y_params["fixed_elites"]

            key, sample_key = jax.random.split(key, 2)
            results = step(
                func=func,
                optimizer=self.x_optimizer,
                var_param=x_params,
                consts=x_consts,
                x_fixed=None,
                y_fixed=y_fixed,
                sample_key=sample_key,
            )

            x_params, opt_results = results
            best_action_x, best_cost_x = opt_results

            carry = (key, x_params, y_params)
            # carry = (sample_key, x_params, y_params)
            outs = (best_action_x, best_cost_x, best_action_y, best_cost_y)

            # outs = (
            #     best_action_x,
            #     best_cost_x,
            #     best_elites_x,
            #     best_action_y,
            #     best_cost_y,
            #     best_elites_y,
            # )
            return carry, outs

        if rng is None:
            rng = jax.random.PRNGKey(self.seed)

        x_params = self.var_x.get_params()
        y_params = self.var_y.get_params()
        x_consts = self.var_x.consts
        y_consts = self.var_y.consts

        carry = (rng, x_params, y_params)

        carry, outs = scan(loop_body, carry, xs=None, length=iterations)

        # Maintain state from previous update invocation
        _, x_params, y_params = carry

        best_action_x, best_cost_x, best_action_y, best_cost_y = outs
        # best_action_x, best_cost_x, elites_x, best_action_y, best_cost_y, elites_y = (
        #     outs
        # )
        # best_action_x = best_action_x[-1]
        # best_action_y = best_action_y[-1]

        return best_action_x, best_action_y, x_params, y_params


def get_partial_cost_fn(func, x_fixed, y_fixed):
    assert (
        x_fixed is None or y_fixed is None
    ), "Only one variable can be fixed to obtain a partial function."
    assert not (x_fixed is None and y_fixed is None), "No fixed values provided."

    if x_fixed is not None:

        def partial_cost_fn_x(y):
            out = vmap(func, in_axes=(0, None))(x_fixed, y)
            return out.mean()

        return partial_cost_fn_x

    if y_fixed is not None:

        def partial_cost_fn_y(x):
            out = vmap(func, in_axes=(None, 0))(x, y_fixed)
            return out.mean()

        return partial_cost_fn_y

    return None


def print_results(results):
    # best_action_x, best_cost_x, _, best_action_y, best_cost_y, _ = results
    best_action_x, best_action_y, _, _ = results

    iterations = len(best_action_x)

    for i in range(iterations):
        print(
            f"Iteration {i} -- Best x: {best_action_x[i]} \t Best y: {best_action_y[i]}"
        )

    return


###########################
### Objective functions ###
###########################


def separable_obj_fun(a, b):
    func = lambda x, y: -((x - a) ** 2) + (y - b) ** 2
    return func


def coupled_obj_fun_1(x, y):
    return x * y - (x - 1) ** 2 + jnp.sin(y)


def coupled_obj_fun_2(x, y):
    return jnp.sin(3 * x) ** 2 + 2 * y + jnp.sin(4 * y) ** 2 + x**2


def mix_of_gaussians(x, y):
    term1 = 0.25 * jnp.exp(-((x - 0.75) ** 2 + (y - 0.75) ** 2) / 0.1)
    term2 = 0.5 * jnp.exp(-((x + 0.75) ** 2 + (y + 0.75) ** 2) / 0.1)
    term3 = 0.75 * jnp.exp(-((x - 0.75) ** 2 + (y + 0.75) ** 2) / 0.1)
    term4 = jnp.exp(-((x + 0.75) ** 2 + (y - 0.75) ** 2) / 0.1)
    term5 = jnp.exp(-((x + 0) ** 2 + (y - 0) ** 2) / 0.5)

    return term5


def modified_rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2 + jnp.sin(x) + jnp.sin(y)


def vector_obj_fun(x, y):
    return jnp.linalg.norm(x - y)
    # return jnp.linalg.norm(x) + jnp.linalg.norm(y)


######################
### Test Optimizer ###
######################

if __name__ == "__main__":
    x_config = {
        "action_dim": (1,),
        "lower_bound": -5,
        "upper_bound": 5,
        "num_elites": 10,
        "num_fixed_elites": 5,
        "num_iter": 3,
        "num_samples": 100,
        "minimum": False,
    }

    y_config = {
        "action_dim": (1,),
        "lower_bound": -5,
        "upper_bound": 5,
        "num_elites": 10,
        "num_fixed_elites": 5,
        "num_iter": 3,
        "num_samples": 100,
        "minimum": True,
    }

    iterations = 10

    x_consts = OptVarConstants(**x_config)
    y_consts = OptVarConstants(**y_config)
    var_x = OptVarParams(x_consts)
    var_y = OptVarParams(y_consts)

    min_max_optimizer = MinMaxOptimizer(var_x, var_y)

    # objective_fun = separable_obj_fun(a=2, b=3)
    objective_fun = coupled_obj_fun_1
    # objective_fun = mix_of_gaussians
    # objective_fun = vector_obj_fun
    # objective_fun = modified_rosenbrock

    # start = time.time()
    # min_max_optimizer.optimize(func=objective_fun, iterations=iterations)
    # end = time.time()
    # elapsed_time = end - start
    # print(f"Execution time without parallelism: {elapsed_time:.3f} s")

    # print("")
    # min_max_optimizer.reset()

    start = time.time()
    results = min_max_optimizer.update(func=objective_fun, iterations=iterations)
    end = time.time()
    elapsed_time = end - start

    x_mean, x_std = min_max_optimizer.var_x.get_mean_std()
    y_mean, y_std = min_max_optimizer.var_y.get_mean_std()

    best_x, best_y, x_params, y_params = results

    print("Before:")

    print(f"Mean x: {x_mean} \t Std x: {x_std}")
    print(f"Mean y: {y_mean} \t Std y: {y_std}")

    print("After:")
    print("Mean x: ", x_params["mean"], "\t Std x: ", x_params["std"])
    print("Mean y: ", y_params["mean"], "\t Std y: ", y_params["std"])

    print(f"Execution time with parallelism: {elapsed_time:.3f} s")

    print_results(results)

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_fun(X, Y)

    # Create contour plot
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="Function Value")

    trajectory_x, trajectory_y, _, _ = results

    # Plot trajectory
    plt.plot(
        trajectory_x,
        trajectory_y,
        marker="o",
        color="red",
        label="Optimization Trajectory",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Contour Plot with Optimization Trajectory")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()
