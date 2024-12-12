import pytest
import jax
import jax.numpy as jnp

from brax.training.replay_buffers import UniformSamplingQueue
from stochastic_optimization.model_based_trainer import ModelBasedTrainer

from stochastic_optimization.utils.trainer_utils import (
    get_dummy_transition,
    prepare_model_input,
    rollout_trajectory,
)
from stochastic_optimization.dynamical_system.pendulum_system import (
    PendulumTrueDynamics,
    PendulumDynamicsParams,
    PendulumRewardParams,
    PendulumSystem,
)
from stochastic_optimization.environment.pendulum_env import (
    CustomPendulum,
    ConstrainedPendulum,
)
from stochastic_optimization.optimizer.cem import CrossEntropyMethod
from stochastic_optimization.optimizer.cem_planner import CEMPlanner
from stochastic_optimization.optimizer.utils import mean_reward, plan
from stochastic_optimization.agent.cem_agent import CEMAgent
from stochastic_optimization.dynamical_system.dynamics_model import BNNDynamicsModel

from bsm.utils.type_aliases import StatisticalModelState, StatisticalModelOutput
from bsm.bayesian_regression.bayesian_neural_networks.fsvgd_ensemble import (
    ProbabilisticFSVGDEnsemble,
)


# ------------------- Test parameters ------------------ #
seed = 14823
buffer_size = 1000
sample_batch_size = 256
num_steps = 200

precision = 1e-4
opt_seed = 14
opt_key = jax.random.PRNGKey(seed=opt_seed)

# ------------------- Fixtures ------------------ #


@pytest.fixture
def env():
    return ConstrainedPendulum()


@pytest.fixture
def true_dynamics():
    return PendulumTrueDynamics()


@pytest.fixture
def bnn_dynamics():
    data_std = 0.1 * jnp.ones((3,))
    statistical_model = BNNDynamicsModel(
        input_dim=4,
        output_dim=3,
        output_stds=data_std,
        logging_wandb=False,
        beta=jnp.array([1.0, 1.0, 1.0]),
        num_particles=1,
        features=[64, 64],
        bnn_type=ProbabilisticFSVGDEnsemble,
        train_share=0.6,
        num_training_steps=2,
        weight_decay=1e-4,
    )
    return statistical_model


@pytest.fixture
def cem_optimizer(env):
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    # CEM parameters for Planning
    horizon = 20  # Planning horizon
    num_elites = 50
    num_iter = 10
    num_samples = 500  # Number of random control sequences to sample

    config = {
        "action_dim": action_dim,
        "horizon": horizon,
        "num_elites": num_elites,
        "num_iter": num_iter,
        "num_samples": num_samples,
        "lower_bound": -2,
        "upper_bound": 2,
    }

    cem_optimizer = CrossEntropyMethod(**config)
    return cem_optimizer


@pytest.fixture
def true_dynamical_system(true_dynamics):
    return PendulumSystem(dynamics=true_dynamics)


@pytest.fixture
def bnn_dynamical_system(bnn_dynamics):
    return PendulumSystem(dynamics=bnn_dynamics)


@pytest.fixture
def true_cem_planner(true_dynamical_system, cem_optimizer):
    return CEMPlanner(true_dynamical_system, cem_optimizer)


@pytest.fixture
def bnn_cem_planner(bnn_dynamical_system, cem_optimizer):
    return CEMPlanner(bnn_dynamical_system, cem_optimizer)


@pytest.fixture
def true_agent(env, true_cem_planner, true_dynamical_system):
    return CEMAgent(
        env.action_space,
        env.observation_space,
        mean_reward,
        true_cem_planner,
        true_dynamical_system,
    )


@pytest.fixture
def bnn_agent(env, bnn_cem_planner, bnn_dynamical_system):
    return CEMAgent(
        env.action_space,
        env.observation_space,
        mean_reward,
        bnn_cem_planner,
        bnn_dynamical_system,
    )


@pytest.fixture
def true_trainer(env, true_agent):
    return ModelBasedTrainer(env=env, agent=true_agent)


@pytest.fixture
def bnn_trainer(env, bnn_agent):
    return ModelBasedTrainer(env=env, agent=bnn_agent)


@pytest.fixture(params=["true_cem_planner", "bnn_cem_planner"])
def cem_planner(request):
    # Access the fixtures dynamically by name
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["true_agent", "bnn_agent"])
def agent(request):
    # Access the fixtures dynamically by name
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["true_trainer", "bnn_trainer"])
def trainer(request):
    # Access the fixtures dynamically by name
    return request.getfixturevalue(request.param)


# ------------------- Helper tests ------------------ #

# Planning parameters
num_last_obs = 5
target_angle = 0
tolerance = 0.05


def common_planning_checks(env, cem_planner, model_params):
    def is_upright(obs, target_angle, tolerance=0.05):
        theta = jnp.arctan2(obs[:, 1], obs[:, 0])
        return jnp.allclose(theta, target_angle, atol=tolerance)

    init_obs, _ = env.reset()
    transitions, _ = plan(
        env=env,
        planner=cem_planner,
        optimize_fn=mean_reward,
        init_obs=init_obs,
        model_params=model_params,
        rng=opt_key,
        num_steps=num_steps,
    )

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    obs = transitions.observation
    actions = transitions.action
    rewards = transitions.reward
    next_obs = transitions.next_observation

    assert obs.shape == next_obs.shape and obs.shape == (num_steps, *obs_dim)
    assert actions.shape == (num_steps, *act_dim)
    assert rewards.shape == (num_steps,)

    assert jnp.logical_and(
        actions >= env.action_space.low, actions <= env.action_space.high
    ).all()

    evaluate_states = next_obs[-num_last_obs:, :]

    assert is_upright(evaluate_states, target_angle, tolerance=tolerance)


def common_prediction_checks(env, agent, model_params, eval_key):
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, reward, _, _, _ = env.step(action)

    pred_next_obs, pred_reward, pred_cost = agent.predict(
        obs, action, model_params, eval_key
    )

    assert (
        type(model_params) is PendulumDynamicsParams
        or type(model_params) is StatisticalModelState
    )

    assert isinstance(pred_next_obs, jnp.ndarray)
    assert next_obs.shape == env.observation_space.shape
    assert jnp.allclose(next_obs, pred_next_obs, atol=precision)


# ------------------- Main tests ------------------ #


# True Dynamics
def test_rollout_trajectory(env):
    # assert (
    #     type(model_params) is PendulumDynamicsParams
    #     or type(model_params) is StatisticalModelState
    # )

    # assert type(reward_params) is PendulumRewardParams

    init_obs, _ = env.reset()
    policy = lambda obs, params, key: env.action_space.sample()
    transitions, last_obs = rollout_trajectory(
        env, policy, init_obs, model_params=None, optimizer_rng=None, num_steps=20
    )

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    assert jnp.all(transitions.observation[0] == init_obs)
    assert transitions.observation.shape == (20, *obs_dim)
    assert transitions.action.shape == (20, *act_dim)
    assert transitions.reward.shape == (20,)
    assert transitions.discount.shape == (20,)
    assert transitions.next_observation.shape == (20, *obs_dim)
    assert jnp.all(transitions.next_observation[-1] == last_obs)


def test_prepare_model_input(agent):
    observations = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    actions = jnp.asarray([[0.1, 0.2], [0.3, 0.4]])
    expected_input = jnp.asarray([[1, 2, 3, 0.1, 0.2], [4, 5, 6, 0.3, 0.4]])
    input = prepare_model_input(observations, actions)
    assert jnp.all(input == expected_input)
    assert input.shape == expected_input.shape


def test_predict(env, true_agent):
    key = jax.random.PRNGKey(seed=seed)
    key, model_key = jax.random.split(key=key, num=2)
    model_params, reward_params = true_agent.dynamical_system.init(model_key)

    key, eval_key = jax.random.split(key=key, num=2)
    common_prediction_checks(env, true_agent, model_params, eval_key)


def test_true_agent_train_step(env, true_agent):
    key = jax.random.PRNGKey(seed=seed)
    dummy_data = get_dummy_transition(env)
    buffer = UniformSamplingQueue(
        max_replay_size=buffer_size,
        dummy_data_sample=dummy_data,
        sample_batch_size=sample_batch_size,
    )
    key, buffer_key = jax.random.split(key=key, num=2)
    buffer_state = buffer.init(key=buffer_key)
    key, model_key = jax.random.split(key=key, num=2)
    model_params, reward_params = true_agent.dynamical_system.init(model_key)

    init_obs, _ = env.reset()
    policy = true_agent.select_best_action
    key, optimizer_key = jax.random.split(key=key, num=2)
    transitions, obs = rollout_trajectory(
        env, policy, init_obs, model_params, optimizer_key, num_steps=200
    )
    buffer_state = buffer.insert(buffer_state, transitions)

    trained_model_params = true_agent.train_step(buffer, buffer_state, model_params)
    assert trained_model_params == model_params

    key, eval_key = jax.random.split(key=key, num=2)
    common_prediction_checks(env, true_agent, model_params, eval_key)


def test_train(env, trainer):
    model_params = trainer.train()

    key = jax.random.PRNGKey(seed=seed)
    key, eval_key = jax.random.split(key=key, num=2)
    common_prediction_checks(env, trainer.agent, model_params, eval_key)
    common_planning_checks(env, trainer.agent.policy_optimizer, model_params)
