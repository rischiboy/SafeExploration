from abc import abstractmethod
from functools import partial
from jax import jit, vmap
import jax
import jax.numpy as jnp
from bsm.statistical_model.abstract_statistical_model import StatisticalModel
from bsm.utils.type_aliases import StatisticalModelOutput
import numpy as np


class AbstractDynamicsModel(StatisticalModel):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input, dynamics_params, rng):
        return self.predict(input, dynamics_params, rng)

    @abstractmethod
    def predict(self, input, dynamics_params, rng):
        pass

    @abstractmethod
    def validate(self, input, dynamics_params, rng):
        pass


class SimpleDynamicsModel(AbstractDynamicsModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Dummy values (Overridden by the subclass)
        self.input_dim = 1
        self.output_dim = 1
        self.default_params = None

    def predict(self, input, dynamics_params=None, rng=None):
        if input.ndim > 1:
            next_obs = self.predict_batch(input, dynamics_params, rng)
        else:
            next_obs = self.predict_single(input, dynamics_params, rng)

        return next_obs

    @partial(jit, static_argnums=(0, 2))
    def predict_single(self, input, dynamics_params=None, rng=None):
        out_dim = self.output_dim
        obs = input[:out_dim]
        action = input[out_dim:]
        next_obs = self.next_state(obs, action, dynamics_params)
        return next_obs

    @partial(jit, static_argnums=(0, 2))
    def predict_batch(self, input, dynamics_params=None, rng=None):
        next_obs = vmap(self.predict_single, in_axes=(0, None))(input, dynamics_params)
        return next_obs

    def validate(self, input, dynamics_params=None, rng=None):
        if input.ndim > 1:
            model_out, next_obs = self.validate_batch(input, dynamics_params, rng)
        else:
            model_out, next_obs = self.validate_single(input, dynamics_params, rng)

        return model_out, next_obs

    @partial(jit, static_argnums=(0, 2))
    def validate_single(self, input, dynamics_params=None, rng=None):
        next_obs = self.predict(input, dynamics_params, rng)
        model_out = StatisticalModelOutput(
            mean=next_obs,
            epistemic_std=np.zeros(self.output_dim),
            aleatoric_std=np.zeros(self.output_dim),
            statistical_model_state=dynamics_params,
        )
        return model_out, next_obs

    @partial(jit, static_argnums=(0, 2))
    def validate_batch(self, input, dynamics_params=None, rng=None):
        output_axis = StatisticalModel.vmap_output_axis(0)
        model_out, next_obs = vmap(
            self.validate_single, in_axes=(0, None), out_axes=(output_axis, 0)
        )(input, dynamics_params)
        return model_out, next_obs

    def next_state(self, obs, action, dynamics_params=None):
        pass

    def update(self, stats_model_state, data):
        return self.default_params

    def init(self, key=None):
        return self.default_params


class CostModel(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, obs, action, next_obs, cost_params=None):
        return self.predict(obs, action, next_obs, cost_params)

    @abstractmethod
    def init(key):
        pass

    @abstractmethod
    def predict(self, obs, action, next_obs, cost_params=None):
        pass


class RewardModel(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, obs, action, next_obs, reward_params=None):
        return self.predict(obs, action, next_obs, reward_params)

    @abstractmethod
    def init(key):
        pass

    @abstractmethod
    def predict(self, obs, action, next_obs, reward_params=None):
        pass


class DynamicalSystem(object):
    def __init__(
        self, dynamics: StatisticalModel, reward: RewardModel, *args, **kwargs
    ):
        self.dynamics = dynamics
        self.reward = reward

    def __call__(self, obs, action, rng, dynamics_params=None, reward_params=None):
        return self.evaluate(obs, action, rng, dynamics_params, reward_params)

    @abstractmethod
    def init(self, rng):
        pass

    # @abstractmethod
    # def predict(self, obs, action, rng=None):
    #     pass

    @abstractmethod
    def evaluate(self, obs, action, rng, dynamics_params, reward_params):
        pass


class SafeDynamicalSystem(object):
    def __init__(
        self,
        dynamics: StatisticalModel,
        reward: RewardModel,
        cost: CostModel,
        *args,
        **kwargs
    ):
        self.dynamics = dynamics
        self.reward = reward
        self.cost = cost

        # Default constraint is always True
        self.constraint = lambda x: True

    def __call__(
        self,
        obs,
        action,
        rng,
        dynamics_params=None,
        reward_params=None,
        cost_params=None,
    ):
        return self.evaluate(
            obs, action, rng, dynamics_params, reward_params, cost_params
        )

    @abstractmethod
    def init(self, rng):
        pass

    # @abstractmethod
    # def predict(self, obs, action, rng=None):
    #     pass

    @abstractmethod
    def evaluate(self, obs, action, rng, dynamics_params, reward_params, cost_params):
        pass

    @abstractmethod
    def evaluate_with_eta(
        self,
        obs,
        action,
        alpha,
        eta,
        dynamics_params=None,
        reward_params=None,
        cost_params=None,
    ):
        pass


class SimpleDynamicalSystem(DynamicalSystem):
    def __init__(
        self, dynamics: StatisticalModel, reward: RewardModel, *args, **kwargs
    ):
        super().__init__(dynamics, reward, *args, **kwargs)
        self.env = None

    def init(self, key=None):
        dynamics_params = self.dynamics.init(key)
        reward_params = self.reward.init(key)

        return dynamics_params, reward_params

    # @partial(jax.jit, static_argnums=(0, 3, 4))
    def evaluate(
        self,
        obs,
        action,
        rng,
        dynamics_params=None,
        reward_params=None,
    ):

        input = jnp.concatenate([obs, action], axis=-1)
        next_obs = self.dynamics(input, dynamics_params, rng)

        # Clip to observation space
        if self.env is not None:
            next_obs = jnp.clip(
                next_obs,
                self.env.observation_space.low,
                self.env.observation_space.high,
            )

        reward = self.reward(obs, action, next_obs, reward_params)
        return next_obs, reward


class SimpleSafeDynamicalSystem(SafeDynamicalSystem):
    def __init__(
        self,
        dynamics: StatisticalModel,
        reward: RewardModel,
        cost: CostModel,
        *args,
        **kwargs
    ):
        super().__init__(dynamics, reward, cost, *args, **kwargs)
        self.env = None

    def init(self, key=None):
        dynamics_params = self.dynamics.init(key)
        reward_params = self.reward.init(key)
        cost_params = self.cost.init(key)

        return dynamics_params, reward_params, cost_params

    # @partial(jax.jit, static_argnums=(0, 3, 4))
    def evaluate(
        self,
        obs,
        action,
        rng,
        dynamics_params=None,
        reward_params=None,
        cost_params=None,
    ):

        input = jnp.concatenate([obs, action], axis=-1)
        next_obs = self.dynamics(input, dynamics_params, rng)

        # Clip to observation space
        if self.env is not None:
            next_obs = jnp.clip(
                next_obs,
                self.env.observation_space.low,
                self.env.observation_space.high,
            )

        reward = self.reward(obs, action, next_obs, reward_params)
        cost = self.cost(obs, action, next_obs, cost_params)
        return next_obs, reward, cost

    def evaluate_with_eta(
        self,
        obs,
        action,
        alpha,
        eta,
        dynamics_params=None,
        reward_params=None,
        cost_params=None,
    ):

        # Not used but just for compatibility
        rng = jax.random.PRNGKey(0)

        input = jnp.concatenate([obs, action], axis=-1)
        (
            model_out,
            _,
        ) = self.dynamics.validate(input, dynamics_params, rng)

        mean = model_out.mean
        eps_std = model_out.epistemic_std
        next_obs = mean + alpha * eps_std * eta

        # Clip to observation space
        if self.env is not None:
            next_obs = jnp.clip(
                next_obs,
                self.env.observation_space.low,
                self.env.observation_space.high,
            )

        reward = self.reward(obs, action, next_obs, reward_params)
        cost = self.cost(obs, action, next_obs, cost_params)

        return (next_obs, reward, cost)


def compute_state_from_diff(input: jnp.ndarray, diff: jnp.ndarray, out_dim: int):

    if input.ndim > 1:
        obs = input[:, :out_dim]
    else:
        obs = input[:out_dim]

    next_obs = obs + diff
    return next_obs
