from stochastic_optimization.optimizer.min_max import *
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os


def separable_obj_fun(a, b):
    func = lambda x, y: -((x - a) ** 2) + (y - b) ** 2
    return func


def coupled_obj_fun_1(x, y):
    return x * y - (x - 1) ** 2 + jnp.sin(y)


def coupled_obj_fun_2(x, y):
    return jnp.sin(3 * x) ** 2 + 2 * y + jnp.sin(4 * y) ** 2 + x**2


def vector_obj_fun(x, y):
    return jnp.norm(x - y)


def mix_of_gaussians(x, y):
    term1 = -0.25 * jnp.exp(-((x - 0.75) ** 2 + (y - 0.75) ** 2) / 0.1)
    term2 = -0.5 * jnp.exp(-((x + 0.75) ** 2 + (y + 0.75) ** 2) / 0.1)
    term3 = -0.75 * jnp.exp(-((x - 0.75) ** 2 + (y + 0.75) ** 2) / 0.1)
    term4 = -jnp.exp(-((x + 0.75) ** 2 + (y - 0.75) ** 2) / 0.1)
    term5 = -jnp.exp(-((x + 0) ** 2 + (y - 0) ** 2) / 0.5)

    return term1 + term2 + term3 + term4 + term5


def create_plot_dir(path: str):
    # Check if the directory exists, and if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    return


"""
Cost of the best action after every CEM update iteration
"""


def cost_iteration_plot(
    costs, num_updates: int, total_iter: int, save_fig: bool = False
):
    total_iter = 2 * num_updates * total_iter
    x = np.array(range(total_iter))
    y = costs

    # Create a plot
    plt.plot(x, y)

    # Color partitions on the x-axis
    for i in range(total_iter):
        if i == 0:
            x_label = {"label": "Update x"}
            y_label = {"label": "Update y"}
        else:
            x_label = {}
            y_label = {}

        plt.axvspan(
            2 * i * num_updates,
            (2 * i + 1) * num_updates,
            color="lightgray",
            alpha=0.5,
            **y_label,
        )
        plt.axvspan(
            (2 * i + 1) * num_updates,
            (2 * i + 2) * num_updates,
            color="lightblue",
            alpha=0.5,
            **x_label,
        )

    # Add labels and legend
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()

    x_ticks = np.arange(x[0], x[-1], step=5)
    plt.xticks(x_ticks)
    plt.xlim(left=x[0], right=x[-1])

    dir_path = "./plots/cost_iteration"
    create_plot_dir(path=dir_path)
    file_name = f"cem_updates_{num_updates}.png"
    file_path = os.path.join(dir_path, file_name)

    if save_fig:
        plt.savefig(file_path)

    # Show the plot
    plt.show()

    return


def cost_samples_plot(costs, times, num_samples, save_fig: bool = False):
    x = np.array(num_samples)
    y = np.array(costs)

    fig = plt.figure(figsize=(12, 6))

    # Create a subplot 1
    ax = fig.add_subplot(121)
    ax.plot(x, y)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Cost")

    x_ticks = np.linspace(x[0], x[-1], num=10 + 1, dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xlim(left=x[0], right=x[-1])

    y = np.array(times)

    # Create a subplot 2
    ax2 = fig.add_subplot(122)
    ax2.plot(x, y)
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Time [s]")

    x_ticks = np.linspace(x[0], x[-1], num=10 + 1, dtype=int)
    ax2.set_xticks(x_ticks)
    ax2.set_xlim(left=x[0], right=x[-1])

    dir_path = "./plots/"
    create_plot_dir(path=dir_path)
    # file_name = f"cost_samples.png"
    file_name = f"cost_samples_one_elite.png"
    file_path = os.path.join(dir_path, file_name)

    if save_fig:
        plt.savefig(file_path)

    # Show the plot
    plt.show()

    return


def cost_elites_plot(costs, times, num_elites, save_fig: bool = False):
    x = np.array(num_elites)
    y = np.array(costs)

    fig = plt.figure(figsize=(12, 6))

    # Create a subplot 1
    ax = fig.add_subplot(121)
    ax.plot(x, y)
    ax.set_xlabel("Elites")
    ax.set_ylabel("Cost")

    x_ticks = np.linspace(x[0], x[-1], num=10 + 1, dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xlim(left=x[0], right=x[-1])

    y = np.array(times)

    # Create a subplot 2
    ax2 = fig.add_subplot(122)
    ax2.plot(x, y)
    ax2.set_xlabel("Elites")
    ax2.set_ylabel("Time [s]")

    x_ticks = np.linspace(x[0], x[-1], num=10 + 1, dtype=int)
    ax2.set_xticks(x_ticks)
    ax2.set_xlim(left=x[0], right=x[-1])

    dir_path = "./plots/"
    create_plot_dir(path=dir_path)
    file_name = f"cost_elites.png"
    file_path = os.path.join(dir_path, file_name)

    if save_fig:
        plt.savefig(file_path)

    # Show the plot
    plt.show()

    return


def test_num_iter(
    obj_fun: Callable,
    x_config: Dict,
    y_config: Dict,
    num_iter: List,
    iterations=10,
    save_fig=False,
):
    kwargs = {"num_iter": num_iter}

    optimizers = create_optimizers(x_config, y_config, **kwargs)

    for it, optimizer in zip(num_iter, optimizers):
        costs, _ = run_optimizer(optimizer, obj_fun, iterations, runs=1)
        cost_iteration_plot(
            costs=costs, num_updates=it, total_iter=iterations, save_fig=save_fig
        )

    return


def test_num_samples(
    obj_fun: Callable,
    x_config: Dict,
    y_config: Dict,
    num_samples: List,
    elite_ratio: float = 0.1,
    iterations=10,
    save_fig=False,
):
    num_elites_list = []

    for x in num_samples:
        # num_elites = int(max(1, elite_ratio * x))
        # num_elites_list.append(num_elites)
        num_elites_list.append(1)

    kwargs = {
        "num_samples": num_samples,
        "num_elites": num_elites_list,
        "num_fixed_elites": num_elites_list,
    }

    optimizers = create_optimizers(x_config, y_config, **kwargs)

    final_cost = []
    execution_times = []
    for optimizer in optimizers:
        costs, time = run_optimizer(optimizer, obj_fun, iterations)

        # final_cost.append(costs[-1])
        final_cost.append(np.mean(costs))
        execution_times.append(time)

    cost_samples_plot(
        costs=final_cost,
        times=execution_times,
        num_samples=num_samples,
        save_fig=save_fig,
    )
    return


def test_num_elites(
    obj_fun: Callable,
    x_config: Dict,
    y_config: Dict,
    num_elites: List,
    iterations=10,
    save_fig=False,
):
    kwargs = {"num_elites": num_elites, "num_fixed_elites": num_elites}

    optimizers = create_optimizers(x_config, y_config, **kwargs)

    final_cost = []
    execution_times = []
    for optimizer in optimizers:
        costs, time = run_optimizer(optimizer, obj_fun, iterations)

        final_cost.append(np.mean(costs))
        execution_times.append(time)

    cost_elites_plot(
        costs=final_cost,
        times=execution_times,
        num_elites=num_elites,
        save_fig=save_fig,
    )
    return


def create_optimizers(x_config: Dict, y_config: Dict, **kwargs):
    # assert len(kwargs) == 1, "Too many arguments provided"

    if not bool(kwargs):
        x_consts = OptVarConstants(**x_config)
        y_consts = OptVarConstants(**y_config)
        var_x = OptVarParams(x_consts)
        var_y = OptVarParams(y_consts)

        min_max_optimizer = MinMaxOptimizer(var_x, var_y)

        optimizers.append(min_max_optimizer)

        return optimizers

    assert bool(kwargs), "No arguments provided"

    _, first_val = list(kwargs.items())[0]
    num_vals = len(first_val) if type(first_val) is list else 1

    for name, val in kwargs.items():
        if type(val) is not list:
            kwargs[name] = [val]

        val = kwargs[name]
        assert (
            name in x_config and name in y_config
        ), f"Argument {name} does not exist in config"
        assert len(val) == num_vals, f"Arguments have different number of values"

    optimizers = []

    for i in range(num_vals):
        for name, val in kwargs.items():
            x_config[name] = val[i]
            y_config[name] = val[i]

        x_consts = OptVarConstants(**x_config)
        y_consts = OptVarConstants(**y_config)
        var_x = OptVarParams(x_consts)
        var_y = OptVarParams(y_consts)

        min_max_optimizer = MinMaxOptimizer(var_x, var_y)

        optimizers.append(min_max_optimizer)

    return optimizers


def run_optimizer(
    optimizer: MinMaxOptimizer, obj_fun: Callable, iterations: int, runs: int = 10
):
    times = []

    for i in range(runs):
        start = time.time()
        results = optimizer.update(func=obj_fun, iterations=iterations)
        end = time.time()
        elapsed_time = end - start
        times.append(elapsed_time)
        optimizer.reset()

    execution_time = np.mean(times)
    print(f"Execution time: {execution_time:.3f} s")

    _, best_cost_x, _, best_cost_y = results

    best_cost = intertwine_arrays(
        best_cost_y, best_cost_x
    )  # y is optimized first and then x is optimized

    return best_cost, execution_time


def intertwine_arrays(arr_1, arr_2):
    result = np.concatenate((arr_1, arr_2), axis=1).flatten()

    return result


if __name__ == "__main__":
    x_config = {
        "action_dim": (1,),
        "bounds": (-1, 1),
        "num_elites": 20,
        "num_fixed_elites": 1,
        "num_iter": 1,
        "num_samples": 200,
        "minimum": False,
    }

    y_config = {
        "action_dim": (1,),
        "bounds": (-1, 1),
        "num_elites": 20,
        "num_fixed_elites": 1,
        "num_iter": 1,
        "num_samples": 200,
        "minimum": True,
    }

    # objective_fun = separable_obj_fun(a=0.5, b=0.75)
    # objective_fun = coupled_obj_fun_1
    objective_fun = mix_of_gaussians

    num_iter_list = [1, 2, 5]
    test_num_iter(
        objective_fun, x_config, y_config, num_iter_list, iterations=10, save_fig=True
    )

    num_samples_list = [1, 10, 50, 100, 200, 300, 400, 500]
    test_num_samples(
        objective_fun,
        x_config,
        y_config,
        num_samples_list,
        elite_ratio=0.1,
        iterations=10,
        save_fig=True,
    )

    num_elites_list = [1, 2, 5, 10, 20, 50, 100, 200]
    test_num_elites(
        objective_fun, x_config, y_config, num_elites_list, iterations=10, save_fig=True
    )
