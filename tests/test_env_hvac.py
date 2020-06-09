import pytest
import tensorflow as tf

from tests import conftest
from tfmpc.envs.hvac import HVAC


def test_str(hvac):
    print()
    print(hvac)


def test_repr(hvac):
    print()
    print(repr(hvac))


def test_temp_bounds(hvac):
    assert hvac.temp_lower_bound.shape == [hvac.state_size, 1]
    assert hvac.temp_upper_bound.shape == [hvac.state_size, 1]
    assert tf.reduce_all(hvac.temp_lower_bound <= hvac.temp_upper_bound)


def test_thermal_conductance(hvac):
    assert hvac.R_outside.shape == [hvac.state_size, 1]
    assert hvac.R_hall.shape == [hvac.state_size, 1]
    assert hvac.R_wall.shape == [hvac.state_size, hvac.state_size]

    assert tf.reduce_all(hvac.R_outside > 0.0)
    assert tf.reduce_all(hvac.R_hall > 0.0)
    assert tf.reduce_all(hvac.R_wall > 0.0)


def test_adj(hvac):
    adj = hvac.adj
    assert adj.shape == [hvac.state_size, hvac.state_size]
    assert tf.reduce_all(tf.linalg.diag_part(adj) == False)
    assert tf.reduce_all(tf.linalg.band_part(adj, -1, 0) == False)


def test_transition(hvac):
    state = conftest.sample_state(hvac)
    action = conftest.sample_action(hvac)

    next_state = hvac.transition(state, action)
    assert next_state.shape == state.shape


def test_linear_transition(hvac):
    state = conftest.sample_state(hvac)
    action = conftest.sample_action(hvac)

    model = hvac.get_linear_transition(state, action, batch=False)

    # check f
    next_state = hvac.transition(state, action, batch=False)
    assert model.f.shape == next_state.shape
    assert tf.reduce_all(model.f == next_state)

    # check f_x
    air = action * hvac.air_max
    heating_x = - tf.linalg.diag(tf.squeeze(air * hvac.CAP_AIR))

    adj = tf.logical_or(hvac.adj, tf.transpose(hvac.adj))
    adj = tf.cast(adj, tf.float32)
    conduction_between_rooms_x = - adj / hvac.R_wall * (-tf.ones_like(adj))

    adj_outside = tf.cast(hvac.adj_outside, tf.float32)
    conduction_with_outside_x = - tf.linalg.diag(
        tf.squeeze(adj_outside / hvac.R_outside))

    adj_hall = tf.cast(hvac.adj_hall, tf.float32)
    conduction_with_hall_x = - tf.linalg.diag(
        tf.squeeze(adj_hall / hvac.R_hall))

    C = tf.linalg.diag(tf.squeeze(hvac.TIME_DELTA / hvac.capacity))
    expected_f_x = (
        tf.eye(hvac.state_size)
        + tf.matmul(C,
            heating_x
            + conduction_between_rooms_x
            + conduction_with_outside_x
            + conduction_with_hall_x
        )
    )

    # print(expected_f_x)
    # print(model.f_x)
    assert expected_f_x.shape == model.f_x.shape
    # assert tf.reduce_all(tf.abs(model.f_x - expected_f_x) < 1e-3)

    # check f_u
    temp = state
    expected_f_u = tf.linalg.diag(
        tf.squeeze(
            hvac.TIME_DELTA / hvac.capacity * hvac.air_max * hvac.CAP_AIR * (hvac.TEMP_AIR - temp)))
    assert model.f_u.shape == expected_f_u.shape
    assert tf.reduce_all(tf.abs(model.f_u - expected_f_u) < 1e-3)


def test_conduction_between_rooms(hvac):
    state = conftest.sample_state(hvac)

    conduction = hvac._conduction_between_rooms(state)
    assert conduction.shape == state.shape

    adj = tf.logical_or(hvac.adj, tf.transpose(hvac.adj))
    R_wall = hvac.R_wall
    temp = tf.squeeze(state)
    expected = []
    for s in range(hvac.state_size):
        c = 0.0
        for p in range(hvac.state_size):
            c += float(adj[s, p]) * (temp[p] - temp[s]) / R_wall[s, p]
        expected.append(c.numpy())
    expected = tf.constant(expected, shape=[hvac.state_size, 1], dtype=tf.float32)
    assert expected.shape == conduction.shape
    assert tf.reduce_all(tf.abs(conduction - expected) < 1e-3)


def test_conduction_with_outside(hvac):
    state = conftest.sample_state(hvac)

    conduction = hvac._conduction_with_outside(state)
    assert conduction.shape == state.shape

    idx1 = tf.where(tf.squeeze(hvac.adj_outside))
    cond = tf.gather(tf.squeeze(conduction), idx1)
    temp_diff = tf.gather(tf.squeeze(hvac.temp_outside - state), idx1)
    assert tf.reduce_all(tf.sign(cond) == tf.sign(temp_diff))

    idx2 = tf.where(tf.squeeze(tf.logical_not(hvac.adj_outside)))
    cond = tf.gather(tf.squeeze(conduction), idx2)
    assert tf.reduce_all(cond == 0.0)


def test_conduction_with_hall(hvac):
    state = conftest.sample_state(hvac)

    conduction = hvac._conduction_with_hall(state)
    assert conduction.shape == state.shape

    idx1 = tf.where(tf.squeeze(hvac.adj_hall))
    cond = tf.gather(tf.squeeze(conduction), idx1)
    temp_diff = tf.gather(tf.squeeze(hvac.temp_hall - state), idx1)
    assert tf.reduce_all(tf.sign(cond) == tf.sign(temp_diff))

    idx2 = tf.where(tf.squeeze(tf.logical_not(hvac.adj_hall)))
    cond = tf.gather(tf.squeeze(conduction), idx2)
    assert tf.reduce_all(cond == 0.0)


def test_cost(hvac):
    state = conftest.sample_state(hvac)
    action = conftest.sample_action(hvac)

    cost = hvac.cost(state, action)
    assert cost.shape == []
    assert cost >= 0.0


def test_final_cost(hvac):
    state = conftest.sample_state(hvac)

    final_cost = hvac.final_cost(state)
    assert final_cost.shape == []
    assert final_cost >= 0.0


def test_quadratic_cost(hvac):
    state = conftest.sample_state(hvac)
    action = conftest.sample_action(hvac)

    model = hvac.get_quadratic_cost(state, action, batch=False)

    # check l
    assert model.l == hvac.cost(state, action, batch=False)

    # check l_x
    temp = state

    below_limit = - tf.linalg.diag(
        tf.squeeze(tf.sign(tf.maximum(0.0, hvac.temp_lower_bound - temp))))
    above_limit = tf.linalg.diag(
        tf.squeeze(tf.sign(tf.maximum(0.0, temp - hvac.temp_upper_bound))))
    out_of_bounds_penalty_x = (
        hvac.PENALTY * (below_limit + above_limit)
    )

    set_point_penalty_x = - hvac.SET_POINT_PENALTY * tf.linalg.diag(
        tf.squeeze(tf.sign(
            (hvac.temp_lower_bound + hvac.temp_upper_bound) / 2 - temp)))

    expected_l_x = tf.reduce_sum(
        out_of_bounds_penalty_x + set_point_penalty_x,
        axis=-1,
        keepdims=True)

    assert model.l_x.shape == expected_l_x.shape
    assert tf.reduce_all(tf.abs(model.l_x - expected_l_x) < 1e-3)

    # check l_u
    expected_l_u = hvac.air_max * hvac.COST_AIR
    assert model.l_u.shape == expected_l_u.shape
    assert tf.reduce_all(tf.abs(model.l_u - expected_l_u) < 1e-3)

    # check all 2nd order derivatives are null
    assert tf.reduce_all(model.l_xx == tf.zeros([hvac.state_size, hvac.state_size]))
    assert tf.reduce_all(model.l_uu == tf.zeros([hvac.state_size, hvac.state_size]))
    assert tf.reduce_all(model.l_xu == tf.zeros([hvac.state_size, hvac.state_size]))
    assert tf.reduce_all(model.l_ux == tf.zeros([hvac.state_size, hvac.state_size]))
