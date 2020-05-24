import pytest
import tensorflow as tf

from tfmpc.envs.hvac import HVAC


def sample_adjacency_matrix(n):
    adj = (tf.random.uniform(shape=[n, n], maxval=1.0) >= 0.4)
    adj = tf.logical_and(adj, tf.eye(n) <= 0)
    adj = tf.linalg.band_part(adj, 0, -1) # upper triangular
    return adj


def sample_state(env, batch_size=None):
    state_size = env.state_size
    shape = [batch_size, state_size, 1] if batch_size else [state_size, 1]
    return tf.random.normal(shape=shape, mean=10.0, dtype=tf.float32)


def sample_action(env, batch_size=None):
    action_size = env.action_size
    shape = [batch_size, action_size, 1] if batch_size else [action_size, 1]
    return tf.random.uniform(shape, dtype=tf.float32)


@pytest.fixture(scope="module")
def env():
    n_rooms = tf.random.uniform(shape=[], minval=2, maxval=6, dtype=tf.int32)

    temp_outside = tf.random.normal(shape=[n_rooms, 1], mean=6.0, dtype=tf.float32)
    temp_hall = tf.random.normal(shape=[n_rooms, 1], mean=10.0, dtype=tf.float32)

    temp_lower_bound = tf.random.normal(shape=[n_rooms, 1], mean=20.0, stddev=1.5, dtype=tf.float32)
    temp_upper_bound = temp_lower_bound + tf.random.uniform(shape=[], minval=3.5, maxval=55.5)

    R_outside = tf.random.normal(shape=[n_rooms, 1], mean=4.0, dtype=tf.float32)
    R_hall = tf.random.normal(shape=[n_rooms, 1], mean=2.0, stddev=0.5, dtype=tf.float32)
    R_wall = tf.random.normal(shape=[n_rooms, n_rooms], mean=1.5, stddev=0.1, dtype=tf.float32)
    R_wall = 1 / 2 * (R_wall + tf.transpose(R_wall))

    capacity = tf.random.normal(shape=[n_rooms, 1], mean=80.0, stddev=2.0, dtype=tf.float32)

    air_max = tf.random.normal(shape=[n_rooms, 1], mean=15.0, dtype=tf.float32)

    adj = sample_adjacency_matrix(int(n_rooms))
    adj_outside = (tf.random.normal(shape=[n_rooms, 1]) >= 0.0)
    adj_hall = (tf.random.normal(shape=[n_rooms, 1]) >= 0.0)

    return HVAC(
        temp_outside, temp_hall,
        temp_lower_bound, temp_upper_bound,
        R_outside, R_hall, R_wall,
        capacity, air_max,
        adj, adj_outside, adj_hall
    )


def test_str(env):
    print()
    print(env)


def test_repr(env):
    print()
    print(repr(env))


def test_temp_bounds(env):
    assert env.temp_lower_bound.shape == [env.state_size, 1]
    assert env.temp_upper_bound.shape == [env.state_size, 1]
    assert tf.reduce_all(env.temp_lower_bound <= env.temp_upper_bound)


def test_thermal_conductance(env):
    assert env.R_outside.shape == [env.state_size, 1]
    assert env.R_hall.shape == [env.state_size, 1]
    assert env.R_wall.shape == [env.state_size, env.state_size]

    assert tf.reduce_all(env.R_outside > 0.0)
    assert tf.reduce_all(env.R_hall > 0.0)
    assert tf.reduce_all(env.R_wall > 0.0)


def test_adj(env):
    adj = env.adj
    assert adj.shape == [env.state_size, env.state_size]
    assert tf.reduce_all(tf.linalg.diag_part(adj) == False)
    assert tf.reduce_all(tf.linalg.band_part(adj, -1, 0) == False)


def test_transition(env):
    state = sample_state(env)
    action = sample_action(env)

    next_state = env.transition(state, action)
    assert next_state.shape == state.shape


def test_conduction_between_rooms(env):
    state = sample_state(env)

    conduction = env._conduction_between_rooms(state)
    assert conduction.shape == state.shape

    adj = tf.logical_or(env.adj, tf.transpose(env.adj))
    R_wall = env.R_wall
    temp = tf.squeeze(state)
    expected = []
    for s in range(env.state_size):
        c = 0.0
        for p in range(env.state_size):
            c += float(adj[s, p]) * (temp[p] - temp[s]) / R_wall[s, p]
        expected.append(c.numpy())
    expected = tf.constant(expected, shape=[env.state_size, 1], dtype=tf.float32)
    assert expected.shape == conduction.shape
    assert tf.reduce_all(tf.abs(conduction - expected) < 1e-3)


def test_conduction_with_outside(env):
    state = sample_state(env)

    conduction = env._conduction_with_outside(state)
    assert conduction.shape == state.shape

    idx1 = tf.where(tf.squeeze(env.adj_outside))
    cond = tf.gather(tf.squeeze(conduction), idx1)
    temp_diff = tf.gather(tf.squeeze(env.temp_outside - state), idx1)
    assert tf.reduce_all(tf.sign(cond) == tf.sign(temp_diff))

    idx2 = tf.where(tf.squeeze(tf.logical_not(env.adj_outside)))
    cond = tf.gather(tf.squeeze(conduction), idx2)
    assert tf.reduce_all(cond == 0.0)


def test_conduction_with_hall(env):
    state = sample_state(env)

    conduction = env._conduction_with_hall(state)
    assert conduction.shape == state.shape

    idx1 = tf.where(tf.squeeze(env.adj_hall))
    cond = tf.gather(tf.squeeze(conduction), idx1)
    temp_diff = tf.gather(tf.squeeze(env.temp_hall - state), idx1)
    assert tf.reduce_all(tf.sign(cond) == tf.sign(temp_diff))

    idx2 = tf.where(tf.squeeze(tf.logical_not(env.adj_hall)))
    cond = tf.gather(tf.squeeze(conduction), idx2)
    assert tf.reduce_all(cond == 0.0)


def test_cost(env):
    state = sample_state(env)
    action = sample_action(env)

    cost = env.cost(state, action)
    assert cost.shape == []
    assert cost >= 0.0


def test_final_cost(env):
    state = sample_state(env)

    final_cost = env.final_cost(state)
    assert final_cost.shape == []
    assert final_cost >= 0.0
