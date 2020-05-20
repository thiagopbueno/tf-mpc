import numpy as np
import pytest
import tensorflow as tf

from tfmpc.envs import navigation


@pytest.fixture(params=[1, 2], ids=["1-zone", "2-zones"])
def env(request):
    n_zones = request.param
    goal = tf.constant([[8.0], [9.0]])
    deceleration = {
        "center": tf.random.normal(shape=[n_zones, 2, 1]),
        "decay": tf.random.uniform(shape=[n_zones,], maxval=3.0)
    }
    low = [-1.0, -1.0]
    high = [1.0, 1.0]
    return navigation.Navigation(goal, deceleration, low, high)


def test_deceleration(env):
    center = env.deceleration["center"]
    decay = env.deceleration["decay"]

    for _ in range(10):
        state = tf.random.normal(shape=[], mean=center[0], stddev=2.0)
        lambda_ = env._deceleration(state, batch=False)

        expected = 1.0
        for c, d in zip(center, decay):
            delta = np.squeeze(state - c)
            distance = np.sqrt(np.sum(delta ** 2, axis=-1))
            expected *= 2 / (1.0 + np.exp(-d * distance)) - 1.0

        assert isinstance(lambda_, tf.Tensor)
        assert lambda_.shape == []
        assert lambda_.dtype == tf.float32
        assert tf.abs(lambda_ - expected) < 1e-4


def test_deceleration_batch(env):
    center = env.deceleration["center"]
    decay = env.deceleration["decay"]

    batch_size = 5
    for _ in range(10):
        state = tf.random.normal(shape=[batch_size,],
                                 mean=center[0], stddev=2.0)
        state = tf.reshape(state, (batch_size, -1, 1))
        lambda_ = env._deceleration(state, batch=True)

        expected = 1.0
        for c, d in zip(center, decay):
            delta = np.squeeze(state - c)
            distance = np.sqrt(np.sum(delta ** 2, axis=-1))
            expected *= 2 / (1.0 + np.exp(-d * distance)) - 1.0

        assert isinstance(lambda_, tf.Tensor)
        assert lambda_.shape == [batch_size,]
        assert lambda_.dtype == tf.float32
        assert tf.reduce_all(tf.abs(lambda_ - expected) < 1e-4)


def test_transition(env):
    for _ in range(10):
        x0 = tf.random.normal((2, 1))
        u0 = tf.random.uniform((2, 1))

        lambda_ = env._deceleration(x0, batch=False)
        x1 = env.transition(x0, u0, batch=False)

        assert isinstance(x1, tf.Tensor)
        assert x1.shape == x0.shape
        assert all(x1 == x0 + lambda_ * u0)


def test_transition_batch(env):
    batch_size = 5

    for _ in range(10):
        x0 = tf.random.normal((batch_size, 2, 1))
        u0 = tf.random.uniform((batch_size, 2, 1))

        lambda_ = env._deceleration(x0, batch=True)
        x1 = env.transition(x0, u0, batch=True)

        assert isinstance(x1, tf.Tensor)
        assert x1.shape == x0.shape
        lambda_ = np.reshape(lambda_, (lambda_.shape[0], 1, 1))
        assert np.allclose(x1, x0 + lambda_ * u0)


def test_linear_transition(env):
    goal = env.goal
    decay = env.deceleration["decay"]
    center = env.deceleration["center"]

    for _ in range(10):
        x = tf.random.normal(shape=(2, 1), mean=center[0], stddev=2.0)
        u = tf.random.uniform((2, 1))

        lambda_ = env._deceleration(x)

        model = env.get_linear_transition(x, u, batch=False)
        f = model.f
        f_x = model.f_x
        f_u = model.f_u

        assert isinstance(f, tf.Tensor)
        assert f.shape == x.shape
        assert tf.reduce_all(f == env.transition(x, u))

        assert isinstance(f_u, tf.Tensor)
        assert f_u.shape == [u.shape[0]] * 2
        assert tf.reduce_all(f_u == lambda_ * tf.eye(2))

        lambda_x = tf.zeros_like(x)
        for c, d in zip(center, decay):
            dist = tf.norm(x - c)
            l = 2 / (1.0 + tf.exp(-d * dist)) - 1.0

            h = 2 * d * tf.exp(-d * dist) / ((1 + tf.exp(-d * dist)) ** 2)
            lambda_x += h * (x - c) / dist * lambda_ / l

        expected_f_x = (tf.eye(tf.shape(x)[0])
                        + tf.tensordot(tf.squeeze(u),
                                       tf.squeeze(lambda_x), axes=0))
        assert isinstance(f_x, tf.Tensor)
        assert f_x.shape == [x.shape[0]] * 2
        assert tf.reduce_all(tf.abs(f_x - expected_f_x) < 1e-4)


def test_linear_transition_batch(env):
    goal = env.goal
    decay = env.deceleration["decay"]
    center = env.deceleration["center"]

    batch_size = 5
    for _ in range(10):
        x = tf.random.normal(shape=(batch_size, 2, 1),
                             mean=center[0], stddev=2.0)
        u = tf.random.uniform((batch_size, 2, 1))

        lambda_ = env._deceleration(x, batch=True)

        model = env.get_linear_transition(x, u, batch=True)
        f = model.f
        f_x = model.f_x
        f_u = model.f_u

        assert isinstance(f, tf.Tensor)
        assert f.shape == x.shape
        assert tf.reduce_all(f == env.transition(x, u, batch=True))

        assert isinstance(f_u, tf.Tensor)
        assert f_u.shape == [batch_size, 2, 2]
        assert tf.reduce_all(f_u == tf.reshape(lambda_, [5, 1, 1]) * tf.eye(2))

        lambda_x = tf.zeros_like(x)
        for c, d in zip(center, decay):
            dist = tf.norm(x - c, axis=1)
            l = 2 / (1.0 + tf.exp(-d * dist)) - 1.0
            h = 2 * d * tf.exp(-d * dist) / ((1 + tf.exp(-d * dist)) ** 2)
            h *= tf.reshape(lambda_, (batch_size, 1)) / (dist * l)
            h = tf.expand_dims(h, -1)
            lambda_x += h * (x - c)

        I = tf.constant(1.0, shape=[5, 1, 1]) * tf.eye(2)
        expected_f_x = I + tf.einsum("bn,bm->bnm",
                                     tf.squeeze(u), tf.squeeze(lambda_x))
        assert isinstance(f_x, tf.Tensor)
        assert f_x.shape == [batch_size, 2, 2]

        assert tf.reduce_all(tf.abs(f_x - expected_f_x) < 1e-4)


def test_cost(env):
    goal = env.goal

    for _ in range(5):
        x = tf.random.normal((2, 1))
        u = tf.random.uniform((2, 1))

        c = env.cost(x, u)

        assert isinstance(c, tf.Tensor)
        assert c.shape == []
        assert tf.reduce_all(c == tf.reduce_sum((x - goal) ** 2))


def test_cost_batch(env):
    batch_size = 5

    goal = env.goal
    for _ in range(5):
        x = tf.random.normal((batch_size, 2, 1))
        u = tf.random.uniform((batch_size, 2, 1))

        c = env.cost(x, u)

        assert isinstance(c, tf.Tensor)
        assert c.shape == [batch_size,]


def test_quadratic_cost(env):
    goal = env.goal

    for _ in range(5):
        x = tf.random.normal((2, 1))
        u = tf.random.uniform((2, 1))

        model = env.get_quadratic_cost(x, u, batch=False)
        l = model.l
        l_x = model.l_x
        l_u = model.l_u
        l_xx = model.l_xx
        l_uu = model.l_uu
        l_xu = model.l_xu
        l_ux = model.l_ux

        assert isinstance(l, tf.Tensor)
        assert l.shape == []
        assert tf.reduce_all(l == env.cost(x, u))

        assert isinstance(l_x, tf.Tensor)
        assert tf.reduce_all(l_x == 2 * (x - goal))

        assert isinstance(l_u, tf.Tensor)
        assert tf.reduce_all(l_u == tf.zeros_like(u))

        assert isinstance(l_xx, tf.Tensor)
        assert tf.reduce_all(l_xx == 2 * tf.eye(2))

        assert isinstance(l_uu, tf.Tensor)
        assert tf.reduce_all(l_uu == tf.zeros([u.shape[0], u.shape[0]]))

        assert isinstance(l_ux, tf.Tensor)
        assert tf.reduce_all(l_ux == tf.zeros((x.shape[0], u.shape[0])))

        assert isinstance(l_xu, tf.Tensor)
        assert tf.reduce_all(l_xu == tf.zeros((u.shape[0], x.shape[0])))


def test_quadratic_cost_batch(env):
    goal = env.goal

    batch_size = 5
    for _ in range(10):
        x = tf.random.normal((batch_size, 2, 1))
        u = tf.random.uniform((batch_size, 2, 1))

        model = env.get_quadratic_cost(x, u, batch=True)
        l = model.l
        l_x = model.l_x
        l_u = model.l_u
        l_xx = model.l_xx
        l_uu = model.l_uu
        l_xu = model.l_xu
        l_ux = model.l_ux

        assert isinstance(l, tf.Tensor)
        assert l.shape == [batch_size,]
        assert tf.reduce_all(l == env.cost(x, u, batch=True))

        assert isinstance(l_x, tf.Tensor)
        assert tf.reduce_all(l_x == 2 * (x - goal))

        assert isinstance(l_u, tf.Tensor)
        assert tf.reduce_all(l_u == tf.zeros_like(u))

        assert isinstance(l_xx, tf.Tensor)
        assert tf.reduce_all(l_xx == (
            tf.constant(2.0, shape=(batch_size, 1, 1)) * tf.eye(2)))

        assert isinstance(l_uu, tf.Tensor)
        assert tf.reduce_all(l_uu == tf.zeros(
            (batch_size, u.shape[1], u.shape[1])))

        assert isinstance(l_ux, tf.Tensor)
        assert tf.reduce_all(l_ux == tf.zeros(
            (batch_size, x.shape[1], u.shape[1])))

        assert isinstance(l_xu, tf.Tensor)
        assert tf.reduce_all(l_xu == tf.zeros(
            (batch_size, u.shape[1], x.shape[1])))


def test_quadratic_final_cost(env):
    goal = env.goal

    for _ in range(5):
        x = tf.random.normal((2, 1))

        model = env.get_quadratic_final_cost(x)
        l = model.l
        l_x = model.l_x
        l_xx = model.l_xx

        assert isinstance(l, tf.Tensor)
        assert l.shape == []
        assert tf.reduce_all(l == env.final_cost(x))

        assert isinstance(l_x, tf.Tensor)
        assert l_x.shape == x.shape
        assert tf.reduce_all(l_x == 2 * (x - goal))

        assert isinstance(l_xx, tf.Tensor)
        assert l_xx.shape == [x.shape[0]] * 2
        assert tf.reduce_all(l_xx == 2 * tf.eye(2))
