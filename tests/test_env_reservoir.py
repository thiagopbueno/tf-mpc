import pytest
import tensorflow as tf

from tests.conftest import sample_state, sample_action

MAX_RES_CAP = 100


def test_repr(reservoir):
    print(repr(reservoir))


def test_str(reservoir):
    print()
    print(str(reservoir))


def test_bounds(reservoir):
    assert reservoir.lower_bound.shape == [reservoir.state_size, 1]
    assert reservoir.upper_bound.shape == [reservoir.state_size, 1]
    assert tf.reduce_all(reservoir.lower_bound < reservoir.upper_bound)
    assert tf.reduce_all(reservoir.upper_bound < MAX_RES_CAP)


def test_downstream(reservoir):
    assert reservoir.downstream.shape == [reservoir.state_size, reservoir.state_size]
    assert tf.reduce_all(tf.reduce_sum(reservoir.downstream, axis=1) >= 0)
    assert tf.reduce_all(tf.reduce_sum(reservoir.downstream, axis=1) <= 1)


def test_rain(reservoir):
    rain_mean = reservoir.rain_shape * reservoir.rain_scale
    assert tf.reduce_all(rain_mean >= 0.0)


def test_vaporated(reservoir):
    rlevel = sample_state(reservoir)
    assert rlevel.shape == [reservoir.state_size, 1]

    value = reservoir._vaporated(rlevel)
    assert value.shape == rlevel.shape
    assert tf.reduce_all(value > 0.0)


def test_vaporated_batch(reservoir):
    batch_size = 10
    rlevel = sample_state(reservoir, batch_size=batch_size)
    assert rlevel.shape == [batch_size, reservoir.state_size, 1]

    value = reservoir._vaporated(rlevel)
    assert value.shape == rlevel.shape
    assert tf.reduce_all(value > 0.0)


def test_rainfall(reservoir):
    rlevel = sample_state(reservoir)
    rainfall = reservoir._rainfall(cec=True)
    assert rainfall.shape == rlevel.shape

    rainfall = reservoir._rainfall(cec=False)
    assert rainfall.shape == rlevel.shape


def test_inflow(reservoir):
    outflow = sample_action(reservoir)
    assert outflow.shape == [reservoir.action_size, 1]
    assert tf.reduce_all(outflow >= 0.0)
    assert tf.reduce_all(outflow <= 1.0)

    value = reservoir._inflow(outflow)
    assert value.shape == [reservoir.state_size, 1]

    assert tf.reduce_all(value[0] == 0.0)
    assert tf.reduce_all(value[1:] == outflow[:-1])

    balance = (
        tf.reduce_sum(value)
        + outflow[-1]
        - tf.reduce_sum(outflow)
    )
    assert tf.abs(balance) < 1e-3


def test_inflow_batch(reservoir):
    batch_size = 10
    outflow = sample_action(reservoir, batch_size=batch_size)
    assert outflow.shape == [batch_size, reservoir.action_size, 1]
    assert tf.reduce_all(outflow >= 0.0)
    assert tf.reduce_all(outflow <= 1.0)

    value = reservoir._inflow(outflow)
    assert value.shape == [batch_size, reservoir.state_size, 1]

    assert tf.reduce_all(value[:, 0] == 0.0)
    assert tf.reduce_all(value[:, 1:] == outflow[:, :-1])

    balance = (
        tf.reduce_sum(value, axis=1)
        + outflow[:,-1]
        - tf.reduce_sum(outflow, axis=1)
    )
    assert tf.reduce_all(tf.abs(balance) < 1e-3)


def test_outflow(reservoir):
    state = sample_state(reservoir)
    action = sample_action(reservoir)

    outflow = reservoir._outflow(action, state)
    assert outflow.shape == [reservoir.state_size, 1]

    assert tf.reduce_all(tf.abs(outflow / state - action) < 1e-3)


def test_outflow_batch(reservoir):
    batch_size = 10
    state = sample_state(reservoir, batch_size)
    action = sample_action(reservoir, batch_size)

    outflow = reservoir._outflow(action, state)
    assert outflow.shape == [batch_size, reservoir.state_size, 1]

    assert tf.reduce_all(tf.abs(outflow / state - action) < 1e-3)


def test_transition(reservoir):
    state = sample_state(reservoir)
    action = sample_action(reservoir)

    next_state = reservoir.transition(state, action, batch=False)
    assert next_state.shape == state.shape

    balance = (
        tf.reduce_sum(state)
        + tf.reduce_sum(reservoir.rain_shape * reservoir.rain_scale)
        - tf.reduce_sum(reservoir._vaporated(state))
        - action[-1] * state[-1]
        - tf.reduce_sum(next_state)
    )
    assert tf.abs(balance) < 1e-3


def test_probabilistic_transition(reservoir):
    state = sample_state(reservoir)
    action = sample_action(reservoir)

    next_state = reservoir.transition(state, action, batch=False, cec=False)
    assert next_state.shape == state.shape


def test_linear_transition_model(reservoir):
    state = sample_state(reservoir)
    action = sample_action(reservoir)
    next_state = reservoir.transition(state, action, batch=False)

    model = reservoir.get_linear_transition(state, action, batch=False)
    f = model.f
    f_x = model.f_x
    f_u = model.f_u

    assert tf.reduce_all(tf.abs(f - next_state) < 1e-3)

    a_t = tf.squeeze(action)
    s_t = tf.squeeze(state)
    C = tf.squeeze(reservoir.max_res_cap)
    grad_v_s = tf.linalg.diag(1/2 * (tf.cos(s_t/C) * s_t/C + tf.sin(s_t/C)))
    grad_F_s = tf.linalg.diag(a_t)
    grad_I_s = tf.matmul(reservoir.downstream, tf.linalg.diag(a_t), transpose_a=True)
    f_x_expected = tf.eye(reservoir.state_size) - grad_v_s - grad_F_s + grad_I_s
    assert f_x.shape == f_x_expected.shape
    assert tf.reduce_all(tf.abs(f_x - f_x_expected) < 1e-3)

    grad_F_a = tf.linalg.diag(s_t)
    grad_I_a = tf.matmul(reservoir.downstream, tf.linalg.diag(s_t), transpose_a=True)
    f_u_expected = - grad_F_a + grad_I_a
    assert f_u.shape == f_u_expected.shape
    assert tf.reduce_all(tf.abs(f_u - f_u_expected) < 1e-3)


def test_cost(reservoir):
    state = sample_state(reservoir)
    action = sample_action(reservoir)

    cost = reservoir.cost(state, action)
    assert cost.shape == []
    assert cost >= 0.0


def test_quadratic_cost_model(reservoir):
    state = sample_state(reservoir)
    action = sample_action(reservoir)
    cost = reservoir.cost(state, action)

    model = reservoir.get_quadratic_cost(state, action, batch=False)
    l = model.l
    l_x = model.l_x
    l_u = model.l_u
    l_xx = model.l_xx
    l_uu = model.l_uu
    l_ux = model.l_ux
    l_xu = model.l_xu

    assert l == cost

    s_t = tf.squeeze(state)
    LB = tf.squeeze(reservoir.lower_bound)
    UB = tf.squeeze(reservoir.upper_bound)
    HP = -tf.squeeze(reservoir.high_penalty)
    LP = -tf.squeeze(reservoir.low_penalty)
    SPP = -tf.squeeze(reservoir.set_point_penalty)

    e1 = HP * tf.cast(((s_t - UB) > 0.0), tf.float32)
    e2 = -LP * tf.cast(((LB - s_t) > 0.0), tf.float32)
    e3 = -SPP * tf.sign((UB + LB) / 2 - s_t)
    l_x_expected = tf.expand_dims(e1 + e2 + e3, -1)

    assert l_x.shape == state.shape
    assert tf.reduce_all(tf.abs(l_x - l_x_expected) < 1e-3)

    assert l_u.shape == action.shape
    assert tf.reduce_all(l_u == tf.zeros_like(action))

    assert l_xx.shape == [reservoir.state_size, reservoir.state_size]
    assert tf.reduce_all(l_xx == tf.zeros([reservoir.state_size, reservoir.state_size]))

    assert l_uu.shape == [reservoir.action_size, reservoir.action_size]
    assert tf.reduce_all(l_uu == tf.zeros([reservoir.action_size, reservoir.action_size]))

    assert l_ux.shape == [reservoir.action_size, reservoir.state_size]
    assert tf.reduce_all(l_ux == tf.zeros([reservoir.action_size, reservoir.state_size]))

    assert l_xu.shape == [reservoir.state_size, reservoir.action_size]
    assert tf.reduce_all(l_xu == tf.zeros([reservoir.state_size, reservoir.action_size]))
