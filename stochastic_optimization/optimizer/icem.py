from functools import partial
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Optional, Tuple, Union
from jax.random import multivariate_normal
from jax.numpy import sqrt, newaxis
from jax.numpy.fft import irfft, rfftfreq
from functools import partial
from typing import Callable, Union, Optional, NamedTuple

from stochastic_optimization.optimizer.cem import CrossEntropyMethod

MIN_STD = 1e-6


@partial(jax.jit, static_argnums=(0, 1, 3))
def powerlaw_psd_gaussian(
    exponent: float, size: int, rng: jax.random.PRNGKey, fmin: float = 0
) -> jax.Array:
    """Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    size : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper.

        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.
    random_state :  int, numpy.integer, numpy.random.Generator, numpy.random.RandomState,
                    optional
        Optionally sets the state of NumPy's underlying random number generator.
        Integer-compatible values or None are passed to np.random.default_rng.
        np.random.RandomState or np.random.Generator are used directly.
        Default: None.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1.0 / samples)  # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Build scaling factors for all frequencies

    # s_scale = f
    # ix = npsum(s_scale < fmin)  # Index of the cutoff
    # if ix and ix < len(s_scale):
    #    s_scale[:ix] = s_scale[ix]
    # s_scale = s_scale ** (-exponent / 2.)
    s_scale = f
    ix = jnp.sum(s_scale < fmin)  # Index of the cutoff

    def cutoff(x, idx):
        x_idx = jax.lax.dynamic_slice(x, start_indices=(idx,), slice_sizes=(1,))
        y = jnp.ones_like(x) * x_idx
        indexes = jnp.arange(0, x.shape[0], step=1)
        first_idx = indexes < idx
        z = (1 - first_idx) * x + first_idx * y
        return z

    def no_cutoff(x, idx):
        return x

    s_scale = jax.lax.cond(
        jnp.logical_and(ix < len(s_scale), ix), cutoff, no_cutoff, s_scale, ix
    )
    s_scale = s_scale ** (-exponent / 2.0)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w = w.at[-1].set(w[-1] * (1 + (samples % 2)) / 2.0)  # correct f = +-0.5
    sigma = 2 * sqrt(jnp.sum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

    # prepare random number generator
    key_sr, key_si, rng = jax.random.split(rng, 3)
    sr = jax.random.normal(key=key_sr, shape=s_scale.shape) * s_scale
    si = jax.random.normal(key=key_si, shape=s_scale.shape) * s_scale

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si = si.at[..., -1].set(0)
        sr = sr.at[..., -1].set(sr[..., -1] * sqrt(2))  # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si = si.at[..., 0].set(0)
    sr = sr.at[..., 0].set(sr[..., 0] * sqrt(2))  # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1j * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma
    return y


class ICEM(CrossEntropyMethod):
    def __init__(
        self,
        exponent: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.exponent = exponent

    @partial(jit, static_argnums=(0, 1))
    def update(
        self,
        cost_fn,
        init_mean: Optional[jax.Array] = None,
        init_std: Optional[Union[float, jax.Array]] = 5.0,
        rng=None,
    ):
        if init_mean is None:
            mean = jnp.zeros(self.sample_dim, dtype=float)
        else:
            mean = init_mean.astype(float)

        assert mean.shape == self.sample_dim
        mean = mean.squeeze()

        init_std = jax.lax.max(init_std, MIN_STD)
        std = jnp.ones(self.sample_dim) * init_std
        std = std.squeeze()

        if rng is None:
            rng = jax.random.PRNGKey(self.seed)

        # best_cost = jnp.inf
        # best_action = mean
        best_elites = jnp.zeros((self.num_elites, *self.sample_dim))
        elites_costs = jnp.ones(self.num_elites) * jnp.inf

        get_curr_best_action = lambda best_cost, best_action, cost, action: (
            cost,
            action,
        )
        get_best_action = lambda best_cost, best_action, cost, action: (
            best_cost,
            best_action,
        )

        def step(carry, ins):
            key, mean, std, best_elites, elites_costs = carry
            best_action = best_elites[0]
            best_cost = elites_costs[0]

            key, step_key = jax.random.split(key, 2)
            sample_key = jax.random.split(step_key, self.num_samples)
            mean = mean.reshape(-1, 1).flatten()
            std = std.reshape(-1, 1).flatten()

            sample_size = np.prod(self.sample_dim)

            # Sample trajectory (vector of iid actions) over a fixed horizon
            colored_samples = jax.vmap(
                lambda rng: powerlaw_psd_gaussian(
                    exponent=self.exponent, size=sample_size, rng=rng
                )
            )(sample_key)
            action_samples = mean + colored_samples * std
            action_samples = self.clip_actions(action_samples)
            action_samples = action_samples.reshape(
                (self.num_samples, *self.sample_dim)
            )
            action_samples = jnp.concatenate([action_samples, best_elites], axis=0)

            # Pick K best-performing samples according to the cost_fn
            sorted_samples, sorted_costs = self.evaluate_samples(
                cost_fn, action_samples
            )
            elites = sorted_samples[: self.num_elites].reshape(best_elites.shape)
            costs = sorted_costs[: self.num_elites].reshape(elites_costs.shape)

            # Updated mean and std according to elites
            elites_mean = jnp.mean(elites, axis=0)
            elites_var = jnp.var(elites, axis=0)

            # Momentum term
            mean = (
                self.alpha * mean.reshape(self.sample_dim)
                + (1 - self.alpha) * elites_mean
            ).squeeze()
            var = (
                self.alpha * jnp.square(std).reshape(self.sample_dim)
                + (1 - self.alpha) * elites_var
            ).squeeze()
            std = jnp.sqrt(var)

            # Corresponds to the elite admitting the lowest cost
            curr_best_action = elites[0].squeeze()
            curr_best_cost = costs[0].squeeze()

            # Update best seen action and cost
            # best = jax.lax.cond(
            #     curr_best_cost <= best_cost,
            #     get_curr_best_action,
            #     get_best_action,
            #     best_cost,
            #     best_action,
            #     curr_best_cost,
            #     curr_best_action,
            # )

            best = jax.lax.cond(
                curr_best_cost <= best_cost,
                get_curr_best_action,
                get_best_action,
                elites_costs,
                best_elites,
                costs,
                elites,
            )

            elites_costs, best_elites = best
            best_cost = elites_costs[0]
            best_action = best_elites[0]

            carry = (key, mean, std, best_elites, elites_costs)
            outs = (best_action, best_cost, best_elites, elites_costs)

            return carry, outs

        carry = (rng, mean, std, best_elites, elites_costs)
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.num_iter)

        best_action, best_cost, elites, costs = outs

        # Return the state of the best elites and corresponding costs after the final iteration
        return best_action[-1], best_cost, elites[-1], costs
