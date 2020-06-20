import pytest
import tensorflow as tf
import tensorflow.compat.v1.logging as tf_logging

from tfmpc.envs.lqr import navigation
from tfmpc.solvers import ilqr

tf_logging.set_verbosity(tf_logging.ERROR)


@pytest.fixture(params=[[[5.5], [-9.0]]])
def goal(request):
    return request.param


@pytest.fixture(params=[0.0, 5.0], ids=["beta=0.0", "beta=5.0"])
def beta(request):
    return request.param


@pytest.fixture(params=[None, [-1.0, 1.0]], ids=["unconstrained", "box"])
def bounds(request):
    return request.param


@pytest.fixture
def initial_state():
    return tf.constant([[0.0], [0.0]])


@pytest.fixture
def horizon():
    return tf.constant(10)


@pytest.fixture(scope="function")
def solver(goal, beta, bounds):
    goal = tf.constant(goal)
    beta = tf.constant(beta)
    low = high = None
    if bounds:
        low, high = bounds

    env = navigation.NavigationLQR(goal, beta, low, high)
    return ilqr.iLQR(env)


def test_start(solver, initial_state, horizon):
    x0 = initial_state
    T = horizon
    states, actions, costs = solver.start(x0, T)

    assert isinstance(states, tf.Tensor)
    assert states.shape == [T + 1, *x0.shape]
    assert tf.reduce_all(states[0] == x0)

    assert isinstance(actions, tf.Tensor)
    assert actions.shape == [T, *x0.shape]

    for t in range(T):
        x, u = states[t], actions[t]
        assert tf.reduce_all(states[t + 1] == solver.env.transition(x, u))


def test_derivatives(solver, initial_state, horizon):
    states, actions, costs = solver.start(initial_state, horizon)
    models = solver.derivatives(states, actions)

    assert len(models) == 3
    transition_model, cost_model, final_cost_model = models

    assert all(tf.shape(g)[0] == horizon for g in transition_model)
    assert all(tf.shape(g)[0] == horizon for g in cost_model)
    #assert all(tf.shape(g)[0] == horizon for g in final_cost_model)


def test_backward(solver, initial_state, horizon):
    states, actions, costs = solver.start(initial_state, horizon)
    transition_model, cost_model, final_cost_model = solver.derivatives(states, actions)
    K, k, J, dV1, dV2 = solver.backward(horizon, actions, transition_model, cost_model, final_cost_model)

    action_size = solver.env.action_size
    state_size = solver.env.state_size

    assert isinstance(K, tf.Tensor)
    assert K.shape == [horizon, action_size, state_size]

    assert isinstance(k, tf.Tensor)
    assert k.shape == [horizon, action_size, 1]


def test_forward(solver, initial_state, horizon):
    x, u, c = solver.start(initial_state, horizon)
    transition_model, cost_model, final_cost_model = solver.derivatives(x, u)
    K, k, J, dV1, dV2 = solver.backward(horizon, u, transition_model, cost_model, final_cost_model)
    states, actions, costs, J_hat, residual = solver.forward(x, u, K, k)

    assert states.shape == x.shape
    assert actions.shape == u.shape
    assert costs.shape == [horizon + 1]

    assert tf.reduce_all(x[0] == states[0])

    for t in range(horizon):
        x, u, c = states[t], actions[t], costs[t]
        assert tf.reduce_all(states[t + 1] == solver.env.transition(x, u))
        assert c == solver.env.cost(x, u)

    assert costs[horizon] == solver.env.final_cost(states[horizon])


def test_solve(solver, initial_state, horizon):
    trajectory, iterations = solver.solve(initial_state, horizon, show_progress=False)
    #assert iterations == 2
    print()
    print(trajectory)
