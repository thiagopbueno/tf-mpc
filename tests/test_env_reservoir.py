import pytest
import tensorflow as tf

from tests.conftest import sample_state, sample_action

def test_repr(reservoir):
    print(repr(reservoir))


def test_str(reservoir):
    print()
    print(str(reservoir))


def test_bounds(reservoir):
    assert reservoir.lower_bound.shape == [reservoir.state_size, 1]
    assert reservoir.upper_bound.shape == [reservoir.state_size, 1]
    assert tf.reduce_all(reservoir.lower_bound < reservoir.upper_bound)
    assert tf.reduce_all(reservoir.upper_bound < reservoir.max_res_cap)


def test_downstream(reservoir):
    assert reservoir.downstream.shape == [reservoir.state_size, reservoir.state_size]
    assert tf.reduce_all(tf.reduce_sum(reservoir.downstream, axis=1) >= 0)
    assert tf.reduce_all(tf.reduce_sum(reservoir.downstream, axis=1) <= 1)


def test_rain(reservoir):
    rain_mean = reservoir.rain_shape * reservoir.rain_scale
    assert tf.reduce_all(rain_mean >= 0.0)


def test_vaporated(reservoir):
    batch_size = 10

    rlevel1 = sample_state(reservoir)
    rlevel2 = sample_state(reservoir, batch_size=batch_size)

    for rlevel in [rlevel1, rlevel2]:
        batch_size = tf.shape(rlevel)[0]

        assert rlevel.shape == [batch_size, reservoir.state_size, 1]

        value = reservoir._vaporated(rlevel)
        assert value.shape == rlevel.shape
        assert tf.reduce_all(value > 0.0)

    fn = reservoir._vaporated
    cfn1 = fn.get_concrete_function(rlevel1)
    cfn2 = fn.get_concrete_function(rlevel2)
    assert cfn1 == cfn2


def test_inflow(reservoir):
    batch_size = 10

    outflow1 = sample_action(reservoir)
    outflow2 = sample_action(reservoir, batch_size=batch_size)

    for outflow in [outflow1, outflow2]:
        batch_size = tf.shape(outflow)[0]

        assert outflow.shape == [batch_size, reservoir.action_size, 1]
        assert tf.reduce_all(outflow >= 0.0)
        assert tf.reduce_all(outflow <= 1.0)

        value = reservoir._inflow(outflow)
        assert value.shape == [batch_size, reservoir.state_size, 1]

        assert tf.reduce_all(value[:, 0] == 0.0)
        assert tf.reduce_all(value[:, 1:] == outflow[:, :-1])

        balance = (
            tf.reduce_sum(value, axis=1)
            + outflow[:, -1]
            - tf.reduce_sum(outflow, axis=1)
        )
        assert tf.reduce_all(tf.abs(balance) < 1e-3)

    fn = reservoir._inflow
    assert fn.get_concrete_function(outflow1) == fn.get_concrete_function(outflow2)


def test_rainfall(reservoir):
    batch_size = 10

    rlevel1 = sample_action(reservoir)
    rlevel2 = sample_action(reservoir, batch_size=batch_size)

    for rlevel in [rlevel1, rlevel2]:
        batch_size = tf.shape(rlevel)[0]

        rainfall = reservoir._rainfall(batch_size)
        assert rainfall.shape == rlevel.shape

        rainfall = reservoir._rainfall(batch_size, cec=tf.constant(False))
        assert rainfall.shape == rlevel.shape

    fn = reservoir._rainfall
    batch_size1 = tf.shape(rlevel1)[0]
    batch_size2 = tf.shape(rlevel2)[0]
    cfn1 = fn.get_concrete_function(batch_size1)
    cfn2 = fn.get_concrete_function(batch_size1, cec=tf.constant(False))
    cfn3 = fn.get_concrete_function(batch_size2)
    cfn4 = fn.get_concrete_function(batch_size2, cec=tf.constant(False))
    assert cfn1 == cfn2 == cfn3 == cfn4


def test_outflow(reservoir):
    batch_size = 10

    flow1 = sample_action(reservoir)
    rlevel1 = sample_state(reservoir)

    flow2 = sample_action(reservoir, batch_size=batch_size)
    rlevel2 = sample_state(reservoir, batch_size=batch_size)

    for flow, rlevel in [(flow1, rlevel1), (flow2, rlevel2)]:
        assert flow.shape == rlevel.shape

        batch_size = tf.shape(rlevel)[0]

        outflow = reservoir._outflow(flow, rlevel)
        assert outflow.shape == [batch_size, reservoir.state_size, 1]

        assert tf.reduce_all(tf.abs(outflow / rlevel - flow) < 1e-3)

    fn = reservoir._outflow
    cfn1 = fn.get_concrete_function(flow1, rlevel1)
    cfn2 = fn.get_concrete_function(flow2, rlevel2)
    assert cfn1 == cfn2


def test_transition(reservoir):
    batch_size = 10

    state1 = sample_state(reservoir)
    action1 = sample_action(reservoir)

    state2 = sample_state(reservoir, batch_size=batch_size)
    action2 = sample_action(reservoir, batch_size=batch_size)

    for state, action in [(state1, action1), (state2, action2)]:
        assert state.shape == action.shape

        cec = tf.constant(False)
        next_state = reservoir.transition(state, action, cec)
        assert next_state.shape == state.shape

        cec = tf.constant(True)
        next_state = reservoir.transition(state, action, cec)
        assert next_state.shape == state.shape

        rain_mean = reservoir.rain_shape * reservoir.rain_scale

        balance = (
            tf.reduce_sum(state, axis=1)
            + tf.expand_dims(tf.reduce_sum(rain_mean), axis=0)
            - tf.reduce_sum(reservoir._vaporated(state), axis=1)
            - action[:, -1] * state[:, -1]
            - tf.reduce_sum(next_state, axis=1)
        )
        assert tf.reduce_all(tf.abs(balance) < 1e-3)

    fn = reservoir.transition
    cfn1 = fn.get_concrete_function(state1, action1, tf.constant(True))
    cfn2 = fn.get_concrete_function(state2, action2, tf.constant(True))
    cfn3 = fn.get_concrete_function(state1, action1, tf.constant(False))
    cfn4 = fn.get_concrete_function(state2, action2, tf.constant(False))
    assert cfn1 == cfn2 == cfn3 == cfn4


def test_linear_transition_model(reservoir):
    batch_size = 10

    cec = tf.constant(True)

    state1 = sample_state(reservoir)
    action1 = sample_action(reservoir)

    state2 = sample_state(reservoir, batch_size=batch_size)
    action2 = sample_action(reservoir, batch_size=batch_size)

    for state, action in [(state1, action1), (state2, action2)]:
        assert state.shape == action.shape
        batch_size = tf.shape(state)[0]

        next_state = reservoir.transition(state, action, cec)

        model = reservoir.get_linear_transition(state, action)
        f = model.f
        f_x = model.f_x
        f_u = model.f_u

        assert tf.reduce_all(tf.abs(f - next_state) < 1e-3)

        a_t = tf.squeeze(action, axis=-1)
        s_t = tf.squeeze(state, axis=-1)
        C = tf.squeeze(reservoir.max_res_cap, axis=-1)
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

    fn = reservoir.get_linear_transition
    cfn1 = fn.get_concrete_function(state1, action1)
    cfn2 = fn.get_concrete_function(state2, action2)
    assert cfn1 == cfn2


def test_cost(reservoir):
    batch_size = 10

    state1 = sample_state(reservoir)
    action1 = sample_action(reservoir)

    state2 = sample_state(reservoir, batch_size=batch_size)
    action2 = sample_action(reservoir, batch_size=batch_size)

    for state, action in [(state1, action1), (state2, action2)]:
        batch_size = tf.shape(state)[0]

        cost = reservoir.cost(state, action)
        assert cost.shape == [batch_size,]
        assert tf.reduce_all(cost >= 0.0)

    fn = reservoir.cost
    cfn1 = fn.get_concrete_function(state1, action1)
    cfn2 = fn.get_concrete_function(state2, action2)
    assert cfn1 == cfn2


def test_final_cost(reservoir):
    batch_size = 10

    state1 = sample_state(reservoir)
    state2 = sample_state(reservoir, batch_size=batch_size)

    for state in [state1, state2]:
        batch_size = tf.shape(state)[0]

        cost = reservoir.final_cost(state)
        assert cost.shape == [batch_size,]
        assert tf.reduce_all(cost >= 0.0)

    fn = reservoir.final_cost
    cfn1 = fn.get_concrete_function(state1)
    cfn2 = fn.get_concrete_function(state2)
    assert cfn1 == cfn2


def test_quadratic_cost_model(reservoir):
    batch_size = 10

    state1 = sample_state(reservoir)
    action1 = sample_action(reservoir)

    state2 = sample_state(reservoir, batch_size=batch_size)
    action2 = sample_action(reservoir, batch_size=batch_size)

    for state, action in [(state1, action1), (state2, action2)]:
        assert state.shape == action.shape
        batch_size = tf.shape(state)[0]

        cost = reservoir.cost(state, action)

        model = reservoir.get_quadratic_cost(state, action)
        l = model.l
        l_x = model.l_x
        l_u = model.l_u
        l_xx = model.l_xx
        l_uu = model.l_uu
        l_ux = model.l_ux
        l_xu = model.l_xu

        assert tf.reduce_all(tf.abs(l - cost) < 1e-3)

        s_t = tf.squeeze(state, axis=-1)
        LB = tf.squeeze(reservoir.lower_bound, axis=-1)
        UB = tf.squeeze(reservoir.upper_bound, axis=-1)
        HP = -tf.squeeze(reservoir.high_penalty, axis=-1)
        LP = -tf.squeeze(reservoir.low_penalty, axis=-1)
        SPP = -tf.squeeze(reservoir.set_point_penalty, axis=-1)

        e1 = HP * tf.cast(((s_t - UB) > 0.0), tf.float32)
        e2 = -LP * tf.cast(((LB - s_t) > 0.0), tf.float32)
        e3 = -SPP * tf.sign((UB + LB) / 2 - s_t)
        l_x_expected = tf.expand_dims(e1 + e2 + e3, -1)

        assert l_x.shape == state.shape
        assert tf.reduce_all(tf.abs(l_x - l_x_expected) < 1e-3)

        assert l_u.shape == action.shape
        assert tf.reduce_all(l_u == tf.zeros_like(action))

        assert l_xx.shape == [batch_size, reservoir.state_size, reservoir.state_size]
        assert tf.reduce_all(l_xx == tf.zeros([batch_size, reservoir.state_size, reservoir.state_size]))

        assert l_uu.shape == [batch_size, reservoir.action_size, reservoir.action_size]
        assert tf.reduce_all(l_uu == tf.zeros([batch_size, reservoir.action_size, reservoir.action_size]))

        assert l_ux.shape == [batch_size, reservoir.action_size, reservoir.state_size]
        assert tf.reduce_all(l_ux == tf.zeros([batch_size, reservoir.action_size, reservoir.state_size]))

        assert l_xu.shape == [batch_size, reservoir.state_size, reservoir.action_size]
        assert tf.reduce_all(l_xu == tf.zeros([batch_size, reservoir.state_size, reservoir.action_size]))

    fn = reservoir.get_quadratic_cost
    cfn1 = fn.get_concrete_function(state1, action1)
    cfn2 = fn.get_concrete_function(state2, action2)
    assert cfn1 == cfn2
