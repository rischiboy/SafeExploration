import gym
import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
from typing import Callable, Optional

from stochastic_optimization.agent.min_max_agent import MinMaxAgent

from bsm.utils.type_aliases import (
    ModelState,
    StatisticalModelState,
)
from bsm.bayesian_regression.bayesian_neural_networks.bnn import BNNState

from stochastic_optimization.optimizer.opt_min_max_planner import (
    OptMinMaxPlanner,
    simulate_trajectory,
)
from stochastic_optimization.utils.scheduler import Scheduler


class OptMinMaxAgent(MinMaxAgent):

    def __init__(
        self, opt_alpha_scheduler: Optional[Scheduler] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.policy_optimizer: OptMinMaxPlanner
        self.opt_alpha_scheduler = opt_alpha_scheduler

    def update_alpha(self, step: int):
        if self.pes_alpha_scheduler is not None:
            new_alpha = self.pes_alpha_scheduler.update(step)
            self.policy_optimizer.set_pes_alpha(new_alpha)

        if self.opt_alpha_scheduler is not None:
            new_alpha = self.opt_alpha_scheduler.update(step)
            self.policy_optimizer.set_opt_alpha(new_alpha)

        return

    def select_best_action(
        self,
        obs: jnp.ndarray,
        model_params: StatisticalModelState[BNNState],
        rng: jnp.ndarray = None,
        return_both: bool = False,
    ):
        """
        Pick the best action for a given observation by optimizing the trajectory that admits the lowest cost.
        """

        rng, eval_key, optimizer_key = jax.random.split(rng, 3)

        optimize_trajectory = self.policy_optimizer.optimize_trajectory

        best_seq, best_opt_seq, best_pes_seq = optimize_trajectory(
            optimize_fn=self.optimize_fn,
            init_obs=obs,
            model_params=model_params,
            eval_key=eval_key,
            optimizer_key=optimizer_key,
        )
        best_action = best_seq[0]

        if return_both:
            best_opt_action = best_opt_seq[0]
            best_hal_action = best_pes_seq[0]
            return best_action, best_opt_action, best_hal_action

        return best_action

    def predict_violations(
        self,
        constraint: Callable,
        init_obs: jnp.ndarray,
        actions: jnp.ndarray,
        opt_actions: jnp.ndarray,
        pes_actions: jnp.ndarray,
        model_params: StatisticalModelState[ModelState],
        rng: jnp.ndarray,
    ):

        ext_actions = jnp.concatenate([opt_actions, actions], axis=-1)

        pes_alpha = self.policy_optimizer.get_pes_alpha()
        opt_alpha = self.policy_optimizer.get_opt_alpha()

        results = simulate_trajectory(
            system=self.dynamical_system,
            model_params=model_params,
            init_obs=init_obs,
            opt_action_seq=ext_actions,
            pes_action_seq=pes_actions,
            opt_alpha=opt_alpha,
            pes_alpha=pes_alpha,
            key=rng,
        )

        opt_trajectory, pes_trajectory, rewards, costs = results
        violations = vmap(constraint, in_axes=(0))(pes_trajectory)
        num_violations = jnp.sum(violations)

        return num_violations
