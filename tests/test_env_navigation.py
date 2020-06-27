import numpy as np
import pytest
import tensorflow as tf

from tests.conftest import sample_state, sample_action


def test_deceleration(navigation):
    center = navigation.deceleration["center"]
    decay = navigation.deceleration["decay"]

    state1 = sample_state(navigation)
    state2 = sample_state(navigation, batch_size=10)

    for state in [state1, state2]:
        batch_size = tf.shape(state)[0]

        lambda_ = navigation._deceleration(state)
        lambda_ = navigation._deceleration(state)

        assert lambda_.shape == [batch_size,]

        for i in range(batch_size):
            state_ = state[i]
            lambda_i = lambda_[i]

            expected = 1.0
            for c, d in zip(center, decay):
                delta = np.squeeze(state_ - c)
                distance = np.sqrt(np.sum(delta ** 2, axis=-1))
                expected *= 2 / (1.0 + np.exp(-d * distance)) - 1.0

            assert isinstance(lambda_i, tf.Tensor)
            assert lambda_i.shape == []
            assert lambda_i.dtype == tf.float32
            assert tf.reduce_all(tf.abs(lambda_i - expected) < 1e-4)

    fn = navigation._deceleration
    cfn1 = fn.get_concrete_function(state1)
    cfn2 = fn.get_concrete_function(state2)
    assert cfn1 == cfn2


def test_transition(navigation):
    state1 = sample_state(navigation)
    action1 = sample_action(navigation)

    batch_size = 10
    state2 = sample_state(navigation, batch_size=batch_size)
    action2 = sample_action(navigation, batch_size=batch_size)

    for state, action in [(state1, action1), (state2, action2)]:
        batch_size = tf.shape(state)[0]

        lambda_ = navigation._deceleration(state)

        cec = tf.constant(False)
        next_state = navigation.transition(state, action, cec)
        assert next_state.shape == state.shape

        next_state2 = navigation.transition(state, action, cec)
        assert tf.reduce_any(next_state != next_state2)

        cec = tf.constant(True)
        next_state = navigation.transition(state, action, cec)
        assert next_state.shape == state.shape

        next_state2 = navigation.transition(state, action, cec)
        assert tf.reduce_all(next_state == next_state2)

        for i in range(batch_size):
            assert tf.reduce_all(next_state[i] == state[i] + lambda_[i] * action[i])

    fn = navigation.transition

    cec = tf.constant(True)
    cfn1 = fn.get_concrete_function(state1, action1, cec)
    cfn2 = fn.get_concrete_function(state2, action2, cec)

    cec = tf.constant(False)
    cfn3 = fn.get_concrete_function(state1, action1, cec)
    cfn4 = fn.get_concrete_function(state2, action2, cec)

    assert cfn1 == cfn2 == cfn3 == cfn4


def test_linear_transition(navigation):
    goal = navigation.goal
    decay = navigation.deceleration["decay"]
    center = navigation.deceleration["center"]

    state1 = sample_state(navigation)
    action1 = sample_action(navigation)

    batch_size = 10
    state2 = sample_state(navigation, batch_size=batch_size)
    action2 = sample_action(navigation, batch_size=batch_size)

    for state, action in [(state1, action1), (state2, action2)]:
        batch_size = tf.shape(state)[0]

        lambda_ = navigation._deceleration(state)

        model = navigation.get_linear_transition(state, action)
        f = model.f
        f_x = model.f_x
        f_u = model.f_u

        assert f.shape == state.shape

        cec = tf.constant(True)
        assert tf.reduce_all(f == navigation.transition(state, action, cec))

        assert f_u.shape == [batch_size] + [navigation.action_size] * 2
        assert tf.reduce_all(f_u == tf.reshape(lambda_, [-1, 1, 1]) * tf.eye(2))
        assert f_x.shape == [batch_size] + [navigation.state_size] * 2

        for i in range(batch_size):
            state_ = state[i]
            action_ = action[i]

            lambda_x = tf.zeros_like(state_)
            for c, d in zip(center, decay):
                dist = tf.norm(state_ - c)
                l = 2 / (1.0 + tf.exp(-d * dist)) - 1.0

                h = 2 * d * tf.exp(-d * dist) / ((1 + tf.exp(-d * dist)) ** 2)
                lambda_x += h * (state_ - c) / dist * lambda_[i] / l

            expected_f_x = (tf.eye(navigation.state_size)
                            + tf.tensordot(tf.squeeze(action_),
                                           tf.squeeze(lambda_x), axes=0))

            assert tf.reduce_all(tf.abs(f_x[i] - expected_f_x) < 1e-4)

    fn = navigation.get_linear_transition
    cfn1 = fn.get_concrete_function(state1, action1)
    cfn2 = fn.get_concrete_function(state2, action2)
    assert cfn1 == cfn2


def test_cost(navigation):
    goal = navigation.goal

    state1 = sample_state(navigation)
    action1 = sample_action(navigation)

    batch_size = 10
    state2 = sample_state(navigation, batch_size=batch_size)
    action2 = sample_action(navigation, batch_size=batch_size)

    for state, action in [(state1, action1), (state2, action2)]:
        batch_size = tf.shape(state)[0]

        cost = navigation.cost(state, action)

        assert isinstance(cost, tf.Tensor)
        assert cost.shape == [batch_size,]
        assert tf.reduce_all(cost >= 0.0)

        cost2 = navigation.cost(tf.expand_dims(goal, axis=0), action)
        assert tf.reduce_all(cost2 == 0.0)

    fn = navigation.cost
    cfn1 = fn.get_concrete_function(state1, action1)
    cfn2 = fn.get_concrete_function(state2, action2)
    assert cfn1 == cfn2


def test_quadratic_cost(navigation):
    goal = navigation.goal

    state1 = sample_state(navigation)
    action1 = sample_action(navigation)

    batch_size = 10
    state2 = sample_state(navigation, batch_size=batch_size)
    action2 = sample_action(navigation, batch_size=batch_size)

    for state, action in [(state1, action1), (state2, action2)]:
        batch_size = tf.shape(state)[0]
        state_size = navigation.state_size
        action_size = navigation.action_size

        model = navigation.get_quadratic_cost(state, action)
        l = model.l
        l_x = model.l_x
        l_u = model.l_u
        l_xx = model.l_xx
        l_uu = model.l_uu
        l_xu = model.l_xu
        l_ux = model.l_ux

        assert l.shape == [batch_size,]
        assert tf.reduce_all(l == navigation.cost(state, action))

        assert tf.reduce_all(l_x == 2 * (state - goal))

        assert tf.reduce_all(l_u == tf.zeros_like(action))

        assert tf.reduce_all(l_xx == tf.reshape(
            tf.tile(2 * tf.eye(state_size), [batch_size, 1]),
            [batch_size, state_size, state_size]))

        assert tf.reduce_all(l_uu == tf.zeros([batch_size, action_size, action_size]))

        assert tf.reduce_all(l_ux == tf.zeros([batch_size, state_size, action_size]))

        assert tf.reduce_all(l_xu == tf.zeros([batch_size, action_size, state_size]))

    fn = navigation.get_quadratic_cost
    cfn1 = fn.get_concrete_function(state1, action1)
    cfn2 = fn.get_concrete_function(state2, action2)
    assert cfn1 == cfn2


def test_quadratic_final_cost(navigation):
    goal = navigation.goal

    state = sample_state(navigation)[0]

    model = navigation.get_quadratic_final_cost(state)
    l = model.l
    l_x = model.l_x
    l_xx = model.l_xx

    assert l.shape == []
    assert tf.reduce_all(l == navigation.final_cost(state))

    assert tf.reduce_all(l_x == 2 * (state - goal))
    assert tf.reduce_all(l_xx == 2 * tf.eye(navigation.state_size))

