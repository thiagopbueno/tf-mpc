import pytest
import tensorflow as tf

from tests import conftest
from tests.conftest import sample_state, sample_action


@pytest.fixture(scope="module", params=["hvac", "reservoir", "navigation_1_zone", "navigation_2_zones"])
def env(request):
    yield getattr(conftest, request.param)()


@pytest.fixture(scope="module", params=[1, 10], ids=["horizon=1", "horizon=10"])
def batch_size(request):
    yield request.param


def test_transition_batch(env, batch_size):
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)

    next_state = env.transition(state, action, batch=True)
    assert next_state.shape == [batch_size, env.state_size, 1]

    for i, (s, a) in enumerate(zip(state, action)):
        next_s = env.transition(s, a, batch=False)
        assert tf.reduce_all(next_s == next_state[i])


def test_cost_batch(env, batch_size):
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)

    cost = env.cost(state, action, batch=True)
    assert cost.shape == [batch_size,]
    assert tf.reduce_all(cost >= 0.0)

    for i, (s, a) in enumerate(zip(state, action)):
        c = env.cost(s, a, batch=False)
        assert tf.reduce_all(c == cost[i])


def test_linear_transition_batch(env, batch_size):
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)

    model = env.get_linear_transition(state, action, batch=True)

    for i, (s, a) in enumerate(zip(state, action)):
        m_i = env.get_linear_transition(s, a, batch=False)
        assert len(m_i) == len(model) == 3
        for f1, f2 in zip(model, m_i):
            assert f1.shape[0] == batch_size
            assert f1[i].shape == f2.shape
            assert tf.reduce_all(tf.abs(f1[i] - f2) < 1e-3)


def test_quadratic_cost_batch(env, batch_size):
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)

    model = env.get_quadratic_cost(state, action, batch=True)

    for i, (s, a) in enumerate(zip(state, action)):
        m_i = env.get_quadratic_cost(s, a, batch=False)
        assert len(model) == len(m_i) == 7
        for l1, l2 in zip(model, m_i):
            assert l1.shape[0] == batch_size
            assert l1[i].shape == l2.shape
            assert tf.reduce_all(tf.abs(l1[i] - l2) < 1e-3)
