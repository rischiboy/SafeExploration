from functools import partial
import jax
from jax import vmap
import jax.numpy as jnp
from jax import jit

from bsm.statistical_model.gp_statistical_model import GPStatisticalModel
from bsm.bayesian_regression.gaussian_processes.gaussian_processes import GPModelState

from stochastic_optimization.dynamical_system.abstract_dynamical_system import (
    compute_state_from_diff,
)
from stochastic_optimization.utils.type_utils import SamplingMode
from stochastic_optimization.utils.trainer_utils import (
    sample,
    sample_from_particle_dist,
)

from bsm.utils.type_aliases import (
    ModelState,
    StatisticalModelOutput,
    StatisticalModelState,
)


class GPDynamicsModel(GPStatisticalModel):
    def __init__(
        self,
        seed: int = 123,
        sampling_mode: SamplingMode = SamplingMode.MEAN,
        diff_states: bool = False,
        *args,
        **kwargs
    ):
        self.seed = seed
        self.sampling_mode = sampling_mode
        self.diff_states = diff_states

        super().__init__(*args, **kwargs)

    def __call__(self, input, dynamics_params, rng):
        # assert input.shape == (self.input_dim,)
        outs = self.predict(input, dynamics_params, rng)

        return outs

    def predict(
        self,
        input: jnp.ndarray,
        dynamics_params: StatisticalModelState[GPModelState],
        rng: jnp.ndarray,
    ):
        if input.ndim > 1:
            num_inputs = input.shape[0]
            pred = self.predict_batch(input, dynamics_params, rng, num_inputs)
        else:
            pred = self.predict_single(input, dynamics_params, rng)

        if self.diff_states:
            next_obs = compute_state_from_diff(input, pred, self.output_dim)
        else:
            next_obs = pred

        return next_obs

    def validate(
        self,
        input: jnp.ndarray,
        dynamics_params: StatisticalModelState[GPModelState],
        rng: jnp.ndarray,
    ):
        if input.ndim > 1:
            num_inputs = input.shape[0]
            model_out, pred = self.validate_batch(
                input, dynamics_params, rng, num_inputs
            )
        else:
            model_out, pred = self.validate_single(input, dynamics_params, rng)

        if self.diff_states:
            next_obs = compute_state_from_diff(input, pred, self.output_dim)
            model_mean = compute_state_from_diff(input, model_out.mean, self.output_dim)
            model_out.mean = model_mean
        else:
            next_obs = pred

        return model_out, next_obs

    @partial(jit, static_argnums=(0))
    def predict_single(
        self,
        input: jnp.ndarray,
        dynamics_params: StatisticalModelState[GPModelState],
        rng: jnp.ndarray,
    ):
        model_out = super(GPStatisticalModel, self).__call__(input, dynamics_params)

        mode = self.sampling_mode.value
        next_obs = sample(model_out, mode, rng)
        return next_obs

    @partial(jit, static_argnums=(0, 4))
    def predict_batch(
        self,
        input: jnp.ndarray,
        dynamics_params: StatisticalModelState[GPModelState],
        rng: jnp.ndarray,
        num_inputs: int,
    ):
        model_out = vmap(
            self._predict,
            in_axes=(0, self.vmap_input_axis(0)),
            out_axes=self.vmap_output_axis(0),
        )(input, dynamics_params)

        sample_keys = jax.random.split(rng, num=(num_inputs))
        mode = self.sampling_mode.value

        next_obs = vmap(
            sample,
            in_axes=(self.vmap_output_axis(0), None, 0),
        )(model_out, mode, sample_keys)

        return next_obs

    @partial(jit, static_argnums=(0))
    def validate_single(
        self,
        input: jnp.ndarray,
        dynamics_params: StatisticalModelState[GPModelState],
        rng: jnp.ndarray,
    ):
        model_out = super(GPStatisticalModel, self).__call__(input, dynamics_params)

        mode = self.sampling_mode.value
        next_obs = sample(model_out, mode, rng)
        return model_out, next_obs

    @partial(jit, static_argnums=(0, 4))
    def validate_batch(
        self,
        input: jnp.ndarray,
        dynamics_params: StatisticalModelState[GPModelState],
        rng: jnp.ndarray,
        num_inputs: int,
    ):
        model_out = vmap(
            self._predict,
            in_axes=(0, self.vmap_input_axis(0)),
            out_axes=self.vmap_output_axis(0),
        )(input, dynamics_params)

        sample_keys = jax.random.split(rng, num=num_inputs)
        mode = self.sampling_mode.value

        next_obs = vmap(
            sample,
            in_axes=(self.vmap_output_axis(0), None, 0),
        )(model_out, mode, sample_keys)
        return model_out, next_obs
