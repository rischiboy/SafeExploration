from functools import partial
import jax
from jax import vmap
import jax.numpy as jnp
from jax import jit

from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from bsm.utils.type_aliases import ModelState, StatisticalModelOutput
from bsm.bayesian_regression.bayesian_neural_networks.bnn import BNNState

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


class BNNDynamicsModel(BNNStatisticalModel):
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
        dynamics_params: StatisticalModelState[BNNState],
        rng: jnp.ndarray,
    ):
        if input.ndim > 1:
            num_inputs = input.shape[0]
            if self.sampling_mode.value == SamplingMode.TS.value:
                pred = self.predict_batch_from_dist(
                    input, dynamics_params, rng, num_inputs
                )
            else:
                pred = self.predict_batch(input, dynamics_params, rng, num_inputs)
        else:
            if self.sampling_mode.value == SamplingMode.TS.value:
                pred = self.predict_from_dist(input, dynamics_params, rng)
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
        dynamics_params: StatisticalModelState[BNNState],
        rng: jnp.ndarray,
    ):
        if input.ndim > 1:
            num_inputs = input.shape[0]
            if self.sampling_mode.value == SamplingMode.TS.value:
                model_out, pred = self.validate_batch_from_dist(
                    input, dynamics_params, rng, num_inputs
                )
            else:
                model_out, pred = self.validate_batch(
                    input, dynamics_params, rng, num_inputs
                )
        else:
            if self.sampling_mode.value == SamplingMode.TS.value:
                model_out, pred = self.validate_from_dist(input, dynamics_params, rng)
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
        dynamics_params: StatisticalModelState[BNNState],
        rng: jnp.ndarray,
    ):
        model_out = super(BNNDynamicsModel, self).__call__(input, dynamics_params)

        mode = self.sampling_mode.value
        next_obs = sample(model_out, mode, rng)
        return next_obs

    @partial(jit, static_argnums=(0, 4))
    def predict_batch(
        self,
        input: jnp.ndarray,
        dynamics_params: StatisticalModelState[BNNState],
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
        dynamics_params: StatisticalModelState[BNNState],
        rng: jnp.ndarray,
    ):
        model_out = super(BNNDynamicsModel, self).__call__(input, dynamics_params)

        mode = self.sampling_mode.value
        next_obs = sample(model_out, mode, rng)
        return model_out, next_obs

    @partial(jit, static_argnums=(0, 4))
    def validate_batch(
        self,
        input: jnp.ndarray,
        dynamics_params: StatisticalModelState[BNNState],
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

    @partial(jit, static_argnums=(0))
    def predict_from_dist(
        self,
        input: jnp.ndarray,
        dynamics_params: StatisticalModelState[BNNState],
        rng: jnp.ndarray,
    ):
        f_dist, y_dist = self.model.posterior(input, dynamics_params.model_state)
        next_obs = sample_from_particle_dist(y_dist, rng)
        return next_obs

    @partial(jit, static_argnums=(0, 4))
    def predict_batch_from_dist(
        self,
        input: jnp.ndarray,
        dynamics_params: StatisticalModelState[BNNState],
        rng: jnp.ndarray,
        num_inputs: int,
    ):
        input_axis = BNNState(
            vmapped_params=None, data_stats=None, calibration_alpha=None
        )
        f_dist, y_dist = vmap(self.model.posterior, in_axes=(0, input_axis))(
            input, dynamics_params.model_state
        )
        sample_keys = jax.random.split(rng, num=num_inputs)

        next_obs = vmap(
            sample_from_particle_dist,
            in_axes=(0, 0),
        )(y_dist, sample_keys)

        return next_obs

    @partial(jit, static_argnums=(0))
    def validate_from_dist(
        self,
        input: jnp.ndarray,
        dynamics_params: StatisticalModelState[BNNState],
        rng: jnp.ndarray,
    ):
        f_dist, y_dist = self.model.posterior(input, dynamics_params.model_state)

        next_obs = sample_from_particle_dist(y_dist, rng)

        statistical_model = StatisticalModelOutput(
            mean=f_dist.mean(),
            epistemic_std=f_dist.stddev(),
            aleatoric_std=y_dist.aleatoric_std(),
            statistical_model_state=dynamics_params,
        )

        return statistical_model, next_obs

    @partial(jit, static_argnums=(0, 4))
    def validate_batch_from_dist(
        self,
        input: jnp.ndarray,
        dynamics_params: StatisticalModelState[BNNState],
        rng: jnp.ndarray,
        num_inputs: int,
    ):
        input_axis = BNNState(
            vmapped_params=None, data_stats=None, calibration_alpha=None
        )
        f_dist, y_dist = vmap(self.model.posterior, in_axes=(0, input_axis))(
            input, dynamics_params.model_state
        )
        sample_keys = jax.random.split(rng, num=num_inputs)

        next_obs = vmap(
            sample_from_particle_dist,
            in_axes=(0, 0),
        )(y_dist, sample_keys)

        statistical_model = StatisticalModelOutput(
            mean=f_dist.mean(),
            epistemic_std=f_dist.stddev(),
            aleatoric_std=y_dist.aleatoric_std(),
            statistical_model_state=dynamics_params,
        )

        return statistical_model, next_obs
