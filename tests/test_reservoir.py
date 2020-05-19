import pytest
import tensorflow as tf

from tfmpc.envs import reservoir


MAX_RES_CAP = 100


def linear_topology(n):
    row = tf.constant([1] + [0] * (n - 1), dtype=tf.float32)
    rows = [tf.roll(row, shift=i, axis=0) for i in range(1, n)]
    rows.append(tf.zeros([n,], dtype=tf.float32))
    rows = tf.stack(rows, axis=0)
    return rows


def sample_state(env, batch_size=None):
    state_size = env.state_size
    shape = [batch_size, state_size, 1] if batch_size else [state_size, 1]
    return tf.random.uniform(shape=shape, maxval=MAX_RES_CAP, dtype=tf.float32)


def sample_action(env, batch_size=None):
    action_size = env.action_size
    shape = [batch_size, action_size, 1] if batch_size else [action_size, 1]
    return tf.random.uniform(shape, dtype=tf.float32)


@pytest.fixture(scope="module")
def env():
    n_reservoirs = tf.random.uniform(
        shape=[], minval=2, maxval=5, dtype=tf.int32)

    lower_bound = MAX_RES_CAP * tf.random.uniform(shape=[n_reservoirs, 1],
                                                  maxval=0.5,
                                                  dtype=tf.float32)
    upper_bound = MAX_RES_CAP * tf.random.uniform(shape=[n_reservoirs, 1],
                                                  minval=0.5,
                                                  dtype=tf.float32)

    downstream = linear_topology(int(n_reservoirs))

    rain = tf.random.gamma(
        shape=[n_reservoirs, 1], alpha=20.0, beta=1.0)

    return reservoir.Reservoir(lower_bound, upper_bound, downstream, rain)


def test_env_repr(env):
    print(repr(env))


def test_env_str(env):
    print()
    print(str(env))


def test_bounds(env):
    assert env.lower_bound.shape == [env.state_size, 1]
    assert env.upper_bound.shape == [env.state_size, 1]
    assert tf.reduce_all(env.lower_bound < env.upper_bound)
    assert tf.reduce_all(env.upper_bound < MAX_RES_CAP)


def test_downstream(env):
    assert env.downstream.shape == [env.state_size, env.state_size]
    assert tf.reduce_all(tf.reduce_sum(env.downstream, axis=1) >= 0)
    assert tf.reduce_all(tf.reduce_sum(env.downstream, axis=1) <= 1)


def test_rain(env):
    assert tf.reduce_all(env.rain >= 0.0)


def test_vaporated(env):
    rlevel = sample_state(env)
    assert rlevel.shape == [env.state_size, 1]

    value = env._vaporated(rlevel)
    assert value.shape == rlevel.shape
    assert tf.reduce_all(value > 0.0)


def test_vaporated_batch(env):
    batch_size = 10
    rlevel = sample_state(env, batch_size=batch_size)
    assert rlevel.shape == [batch_size, env.state_size, 1]

    value = env._vaporated(rlevel)
    assert value.shape == rlevel.shape
    assert tf.reduce_all(value > 0.0)


def test_inflow(env):
    outflow = sample_action(env)
    assert outflow.shape == [env.action_size, 1]
    assert tf.reduce_all(outflow >= 0.0)
    assert tf.reduce_all(outflow <= 1.0)

    value = env._inflow(outflow)
    assert value.shape == [env.state_size, 1]

    assert tf.reduce_all(value[0] == 0.0)
    assert tf.reduce_all(value[1:] == outflow[:-1])

    balance = (
        tf.reduce_sum(value)
        + outflow[-1]
        - tf.reduce_sum(outflow)
    )
    assert tf.abs(balance) < 1e-3


def test_inflow_batch(env):
    batch_size = 10
    outflow = sample_action(env, batch_size=batch_size)
    assert outflow.shape == [batch_size, env.action_size, 1]
    assert tf.reduce_all(outflow >= 0.0)
    assert tf.reduce_all(outflow <= 1.0)

    value = env._inflow(outflow)
    assert value.shape == [batch_size, env.state_size, 1]

    assert tf.reduce_all(value[:, 0] == 0.0)
    assert tf.reduce_all(value[:, 1:] == outflow[:, :-1])

    balance = (
        tf.reduce_sum(value, axis=1)
        + outflow[:,-1]
        - tf.reduce_sum(outflow, axis=1)
    )
    assert tf.reduce_all(tf.abs(balance) < 1e-3)


def test_outflow(env):
    state = sample_state(env)
    action = sample_action(env)

    outflow = env._outflow(action, state)
    assert outflow.shape == [env.state_size, 1]

    assert tf.reduce_all(tf.abs(outflow / state - action) < 1e-3)


def test_outflow_batch(env):
    batch_size = 10
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)

    outflow = env._outflow(action, state)
    assert outflow.shape == [batch_size, env.state_size, 1]

    assert tf.reduce_all(tf.abs(outflow / state - action) < 1e-3)


def test_transition(env):
    state = sample_state(env)
    action = sample_action(env)

    next_state = env.transition(state, action)
    assert next_state.shape == state.shape

    balance = (
        tf.reduce_sum(state)
        + tf.reduce_sum(env.rain)
        - tf.reduce_sum(env._vaporated(state))
        - action[-1] * state[-1]
        - tf.reduce_sum(next_state)
    )
    assert tf.abs(balance) < 1e-3


def test_transition_batch(env):
    batch_size = 10
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)

    next_state = env.transition(state, action, batch=tf.constant(True))
    assert next_state.shape == [batch_size, env.state_size, 1]

    balance = (
        tf.reduce_sum(state, axis=1)
        + tf.reduce_sum(env.rain)
        - tf.reduce_sum(env._vaporated(state), axis=1)
        - action[:, -1] * state[:, -1]
        - tf.reduce_sum(next_state, axis=1)
    )
    assert tf.reduce_all(tf.abs(balance) < 1e-3)


def test_cost(env):
    state = sample_state(env)
    action = sample_action(env)

    cost = env.cost(state, action)
    assert cost.shape == []


def test_cost_batch(env):
    batch_size = 10
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)

    cost = env.cost(state, action, batch=tf.constant(True))
    assert cost.shape == [batch_size,]
