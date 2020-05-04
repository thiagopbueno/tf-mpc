import pytest
import tensorflow as tf

from tfmpc.envs import navigation
from tfmpc.solvers import ilqr


@pytest.fixture
def solver():
    goal = tf.constant([[5.5], [-9.0]])
    beta = 5.0
    env = navigation.Navigation(goal, beta)
    return ilqr.iLQR(env)


@pytest.fixture
def initial_state():
    return tf.constant([[0.0], [0.0]])


@pytest.fixture
def horizon():
    return tf.constant(10)


def test_start(solver, initial_state, horizon):
    x0 = initial_state
    T = horizon
    states, actions = solver.start(x0, T)

    assert isinstance(states, tf.Tensor)
    assert states.shape == [T + 1, *x0.shape]
    assert tf.reduce_all(states[0] == x0)

    assert isinstance(actions, tf.Tensor)
    assert actions.shape == [T, *x0.shape]

    for t in range(T):
        x, u = states[t], actions[t]
        assert tf.reduce_all(states[t + 1] == solver.env.transition(x, u))


def test_backward(solver, initial_state, horizon):
    states, actions = solver.start(initial_state, horizon)
    K, k = solver.backward(states, actions)

    action_size = solver.env.action_size
    state_size = solver.env.state_size

    assert isinstance(K, tf.Tensor)
    assert K.shape == [horizon, action_size, state_size]

    assert isinstance(k, tf.Tensor)
    assert k.shape == [horizon, action_size, 1]


def test_forward(solver, initial_state, horizon):
    x, u = solver.start(initial_state, horizon)
    K, k = solver.backward(x, u)
    states, actions = solver.forward(x, u, K, k)

    assert states.shape == x.shape
    assert actions.shape == u.shape

    for t in range(horizon):
        x, u = states[t], actions[t]
        assert tf.reduce_all(states[t + 1] == solver.env.transition(x, u))
