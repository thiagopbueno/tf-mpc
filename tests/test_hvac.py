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


def test_transition_batch(env):
    batch_size = 10
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)

    next_state = env.transition(state, action, batch=True)
    assert next_state.shape == [batch_size, env.state_size, 1]

    for i, (s, a) in enumerate(zip(state, action)):
        next_s = env.transition(s, a, batch=False)
        assert tf.reduce_all(next_s == next_state[i])


def test_linear_transition(env):
    state = sample_state(env)
    action = sample_action(env)

    model = env.get_linear_transition(state, action, batch=False)

    # check f
    next_state = env.transition(state, action, batch=False)
    assert model.f.shape == next_state.shape
    assert tf.reduce_all(model.f == next_state)

    # check f_x
    air = action * env.air_max
    heating_x = - tf.linalg.diag(tf.squeeze(air * env.CAP_AIR))

    adj = tf.logical_or(env.adj, tf.transpose(env.adj))
    adj = tf.cast(adj, tf.float32)
    conduction_between_rooms_x = - adj / env.R_wall * (-tf.ones_like(adj))

    adj_outside = tf.cast(env.adj_outside, tf.float32)
    conduction_with_outside_x = - tf.linalg.diag(
        tf.squeeze(adj_outside / env.R_outside))

    adj_hall = tf.cast(env.adj_hall, tf.float32)
    conduction_with_hall_x = - tf.linalg.diag(
        tf.squeeze(adj_hall / env.R_hall))

    C = tf.linalg.diag(tf.squeeze(env.TIME_DELTA / env.capacity))
    expected_f_x = (
        tf.eye(env.state_size)
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
            env.TIME_DELTA / env.capacity * env.air_max * env.CAP_AIR * (env.TEMP_AIR - temp)))
    assert model.f_u.shape == expected_f_u.shape
    assert tf.reduce_all(tf.abs(model.f_u - expected_f_u) < 1e-3)


def test_linear_transition_batch(env):
    batch_size = 10
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


def test_cost_batch(env):
    batch_size = 10
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)

    cost = env.cost(state, action, batch=True)
    assert cost.shape == [batch_size,]
    assert tf.reduce_all(cost >= 0.0)

    for i, (s, a) in enumerate(zip(state, action)):
        c = env.cost(s, a, batch=False)
        assert tf.reduce_all(c == cost[i])


def test_final_cost(env):
    state = sample_state(env)

    final_cost = env.final_cost(state)
    assert final_cost.shape == []
    assert final_cost >= 0.0


def test_quadratic_cost(env):
    state = sample_state(env)
    action = sample_action(env)

    model = env.get_quadratic_cost(state, action, batch=False)

    # check l
    assert model.l == env.cost(state, action, batch=False)

    # check l_x
    temp = state

    below_limit = - tf.linalg.diag(
        tf.squeeze(tf.sign(tf.maximum(0.0, env.temp_lower_bound - temp))))
    above_limit = tf.linalg.diag(
        tf.squeeze(tf.sign(tf.maximum(0.0, temp - env.temp_upper_bound))))
    out_of_bounds_penalty_x = (
        env.PENALTY * (below_limit + above_limit)
    )

    set_point_penalty_x = - env.SET_POINT_PENALTY * tf.linalg.diag(
        tf.squeeze(tf.sign(
            (env.temp_lower_bound + env.temp_upper_bound) / 2 - temp)))

    expected_l_x = tf.reduce_sum(
        out_of_bounds_penalty_x + set_point_penalty_x,
        axis=-1,
        keepdims=True)

    assert model.l_x.shape == expected_l_x.shape
    assert tf.reduce_all(tf.abs(model.l_x - expected_l_x) < 1e-3)

    # check l_u
    expected_l_u = env.air_max * env.COST_AIR
    assert model.l_u.shape == expected_l_u.shape
    assert tf.reduce_all(tf.abs(model.l_u - expected_l_u) < 1e-3)

    # check all 2nd order derivatives are null
    assert tf.reduce_all(model.l_xx == tf.zeros([env.state_size, env.state_size]))
    assert tf.reduce_all(model.l_uu == tf.zeros([env.state_size, env.state_size]))
    assert tf.reduce_all(model.l_xu == tf.zeros([env.state_size, env.state_size]))
    assert tf.reduce_all(model.l_ux == tf.zeros([env.state_size, env.state_size]))


def test_quadratic_cost_batch(env):
    batch_size = 10
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
