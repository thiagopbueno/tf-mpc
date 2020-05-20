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

    next_state = env.transition(state, action, batch=False)
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

    next_state = env.transition(state, action, batch=True)
    assert next_state.shape == [batch_size, env.state_size, 1]

    balance = (
        tf.reduce_sum(state, axis=1)
        + tf.reduce_sum(env.rain)
        - tf.reduce_sum(env._vaporated(state), axis=1)
        - action[:, -1] * state[:, -1]
        - tf.reduce_sum(next_state, axis=1)
    )
    assert tf.reduce_all(tf.abs(balance) < 1e-3)


def test_linear_transition_model(env):
    state = sample_state(env)
    action = sample_action(env)
    next_state = env.transition(state, action, batch=False)

    model = env.get_linear_transition(state, action, batch=False)
    f = model.f
    f_x = model.f_x
    f_u = model.f_u

    assert tf.reduce_all(tf.abs(f - next_state) < 1e-3)

    a_t = tf.squeeze(action)
    s_t = tf.squeeze(state)
    C = env.BIGGESTMAXCAP
    grad_v_s = tf.linalg.diag(1/2 * (tf.cos(s_t/C) * s_t/C + tf.sin(s_t/C)))
    grad_F_s = tf.linalg.diag(a_t)
    grad_I_s = tf.matmul(env.downstream, tf.linalg.diag(a_t), transpose_a=True)
    f_x_expected = tf.eye(env.state_size) - grad_v_s - grad_F_s + grad_I_s
    assert f_x.shape == f_x_expected.shape
    assert tf.reduce_all(tf.abs(f_x - f_x_expected) < 1e-3)

    grad_F_a = tf.linalg.diag(s_t)
    grad_I_a = tf.matmul(env.downstream, tf.linalg.diag(s_t), transpose_a=True)
    f_u_expected = - grad_F_a + grad_I_a
    assert f_u.shape == f_u_expected.shape
    assert tf.reduce_all(tf.abs(f_u - f_u_expected) < 1e-3)


def test_linear_transition_model_batch(env):
    batch_size = 5
    state = sample_state(env, batch_size=batch_size)
    action = sample_action(env, batch_size=batch_size)
    next_state = env.transition(state, action, batch=True)

    model = env.get_linear_transition(state, action, batch=True)
    f = model.f
    f_x = model.f_x
    f_u = model.f_u

    assert f.shape == [batch_size, env.state_size, 1]
    assert f_x.shape == [batch_size, env.state_size, env.state_size]
    assert f_u.shape == [batch_size, env.state_size, env.action_size]

    assert tf.reduce_all(tf.abs(f - next_state) < 1e-3)

    a_t = tf.squeeze(action)
    s_t = tf.squeeze(state)
    C = env.BIGGESTMAXCAP
    grad_v_s = tf.linalg.diag(1/2 * (tf.cos(s_t/C) * s_t/C + tf.sin(s_t/C)))
    grad_F_s = tf.linalg.diag(a_t)
    grad_I_s = tf.matmul(env.downstream, tf.linalg.diag(a_t), transpose_a=True)
    f_x_expected = tf.eye(env.state_size) - grad_v_s - grad_F_s + grad_I_s
    assert f_x.shape == f_x_expected.shape
    assert tf.reduce_all(tf.abs(f_x - f_x_expected) < 1e-3)

    grad_F_a = tf.linalg.diag(s_t)
    grad_I_a = tf.matmul(env.downstream, tf.linalg.diag(s_t), transpose_a=True)
    f_u_expected = - grad_F_a + grad_I_a
    assert f_u.shape == f_u_expected.shape
    assert tf.reduce_all(tf.abs(f_u - f_u_expected) < 1e-3)


def test_cost(env):
    state = sample_state(env)
    action = sample_action(env)

    cost = env.cost(state, action)
    assert cost.shape == []


def test_cost_batch(env):
    batch_size = 10
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)

    cost = env.cost(state, action, batch=True)
    assert cost.shape == [batch_size,]


def test_quadratic_cost_model(env):
    state = sample_state(env)
    action = sample_action(env)
    cost = env.cost(state, action)

    model = env.get_quadratic_cost(state, action, batch=False)
    l = model.l
    l_x = model.l_x
    l_u = model.l_u
    l_xx = model.l_xx
    l_uu = model.l_uu
    l_ux = model.l_ux
    l_xu = model.l_xu

    assert l == cost

    s_t = tf.squeeze(state)
    LB = tf.squeeze(env.lower_bound)
    UB = tf.squeeze(env.upper_bound)
    HP = env.HIGH_PENALTY
    LP = env.LOW_PENALTY
    SPP = env.SET_POINT_PENALTY

    e1 = HP * tf.cast(((s_t - UB) > 0.0), tf.float32)
    e2 = -LP * tf.cast(((LB - s_t) > 0.0), tf.float32)
    e3 = -SPP * tf.sign((UB + LB) / 2 - s_t)
    l_x_expected = tf.expand_dims(e1 + e2 + e3, -1)

    assert l_x.shape == state.shape
    assert tf.reduce_all(tf.abs(l_x - l_x_expected) < 1e-3)

    assert l_u.shape == action.shape
    assert tf.reduce_all(l_u == tf.zeros_like(action))

    assert l_xx.shape == [env.state_size, env.state_size]
    assert tf.reduce_all(l_xx == tf.zeros([env.state_size, env.state_size]))

    assert l_uu.shape == [env.action_size, env.action_size]
    assert tf.reduce_all(l_uu == tf.zeros([env.action_size, env.action_size]))

    assert l_ux.shape == [env.action_size, env.state_size]
    assert tf.reduce_all(l_ux == tf.zeros([env.action_size, env.state_size]))

    assert l_xu.shape == [env.state_size, env.action_size]
    assert tf.reduce_all(l_xu == tf.zeros([env.state_size, env.action_size]))


def test_quadratic_cost_model_batch(env):
    batch_size = 10
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)
    cost = env.cost(state, action, batch=True)

    model = env.get_quadratic_cost(state, action, batch=True)
    l = model.l
    l_x = model.l_x
    l_u = model.l_u
    l_xx = model.l_xx
    l_uu = model.l_uu
    l_ux = model.l_ux
    l_xu = model.l_xu

    assert l.shape == [batch_size,]
    assert l_x.shape == [batch_size, env.state_size, 1]
    assert l_u.shape == [batch_size, env.action_size, 1]
    assert l_xx.shape == [batch_size, env.state_size, env.state_size]
    assert l_uu.shape == [batch_size, env.action_size, env.action_size]
    assert l_xu.shape == [batch_size, env.state_size, env.action_size]
    assert l_ux.shape == [batch_size, env.action_size, env.state_size]

    assert tf.reduce_all(tf.abs(l - cost) < 1e-3)

    s_t = tf.squeeze(state)
    LB = tf.squeeze(env.lower_bound)
    UB = tf.squeeze(env.upper_bound)
    HP = env.HIGH_PENALTY
    LP = env.LOW_PENALTY
    SPP = env.SET_POINT_PENALTY

    e1 = HP * tf.cast(((s_t - UB) > 0.0), tf.float32)
    e2 = -LP * tf.cast(((LB - s_t) > 0.0), tf.float32)
    e3 = -SPP * tf.sign((UB + LB) / 2 - s_t)
    l_x_expected = tf.expand_dims(e1 + e2 + e3, -1)

    assert tf.reduce_all(tf.abs(l_x - l_x_expected) < 1e-3)

    assert tf.reduce_all(l_u == tf.zeros_like(action))
    assert tf.reduce_all(l_xx == tf.zeros([batch_size, env.state_size, env.state_size]))
    assert tf.reduce_all(l_uu == tf.zeros([batch_size, env.action_size, env.action_size]))
    assert tf.reduce_all(l_ux == tf.zeros([batch_size, env.action_size, env.state_size]))
    assert tf.reduce_all(l_xu == tf.zeros([batch_size, env.state_size, env.action_size]))
