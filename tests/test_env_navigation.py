import numpy as np
import pytest
import tensorflow as tf


def test_deceleration(navigation):
    center = navigation.deceleration["center"]
    decay = navigation.deceleration["decay"]

    for _ in range(10):
        state = tf.random.normal(shape=[], mean=center[0], stddev=2.0)
        lambda_ = navigation._deceleration(state, batch=False)

        expected = 1.0
        for c, d in zip(center, decay):
            delta = np.squeeze(state - c)
            distance = np.sqrt(np.sum(delta ** 2, axis=-1))
            expected *= 2 / (1.0 + np.exp(-d * distance)) - 1.0

        assert isinstance(lambda_, tf.Tensor)
        assert lambda_.shape == []
        assert lambda_.dtype == tf.float32
        assert tf.abs(lambda_ - expected) < 1e-4


def test_deceleration_batch(navigation):
    center = navigation.deceleration["center"]
    decay = navigation.deceleration["decay"]

    batch_size = 5
    for _ in range(10):
        state = tf.random.normal(shape=[batch_size,],
                                 mean=center[0], stddev=2.0)
        state = tf.reshape(state, (batch_size, -1, 1))
        lambda_ = navigation._deceleration(state, batch=True)

        expected = 1.0
        for c, d in zip(center, decay):
            delta = np.squeeze(state - c)
            distance = np.sqrt(np.sum(delta ** 2, axis=-1))
            expected *= 2 / (1.0 + np.exp(-d * distance)) - 1.0

        assert isinstance(lambda_, tf.Tensor)
        assert lambda_.shape == [batch_size,]
        assert lambda_.dtype == tf.float32
        assert tf.reduce_all(tf.abs(lambda_ - expected) < 1e-4)


def test_transition(navigation):
    for _ in range(10):
        x0 = tf.random.normal((2, 1))
        u0 = tf.random.uniform((2, 1))

        lambda_ = navigation._deceleration(x0, batch=False)
        x1 = navigation.transition(x0, u0, batch=False)

        assert isinstance(x1, tf.Tensor)
        assert x1.shape == x0.shape
        assert all(x1 == x0 + lambda_ * u0)


def test_linear_transition(navigation):
    goal = navigation.goal
    decay = navigation.deceleration["decay"]
    center = navigation.deceleration["center"]

    for _ in range(10):
        x = tf.random.normal(shape=(2, 1), mean=center[0], stddev=2.0)
        u = tf.random.uniform((2, 1))

        lambda_ = navigation._deceleration(x)

        model = navigation.get_linear_transition(x, u, batch=False)
        f = model.f
        f_x = model.f_x
        f_u = model.f_u

        assert isinstance(f, tf.Tensor)
        assert f.shape == x.shape
        assert tf.reduce_all(f == navigation.transition(x, u))

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


def test_cost(navigation):
    goal = navigation.goal

    for _ in range(5):
        x = tf.random.normal((2, 1))
        u = tf.random.uniform((2, 1))

        c = navigation.cost(x, u)

        assert isinstance(c, tf.Tensor)
        assert c.shape == []
        assert tf.reduce_all(c == tf.reduce_sum((x - goal) ** 2))


def test_quadratic_cost(navigation):
    goal = navigation.goal

    for _ in range(5):
        x = tf.random.normal((2, 1))
        u = tf.random.uniform((2, 1))

        model = navigation.get_quadratic_cost(x, u, batch=False)
        l = model.l
        l_x = model.l_x
        l_u = model.l_u
        l_xx = model.l_xx
        l_uu = model.l_uu
        l_xu = model.l_xu
        l_ux = model.l_ux

        assert isinstance(l, tf.Tensor)
        assert l.shape == []
        assert tf.reduce_all(l == navigation.cost(x, u))

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


def test_quadratic_final_cost(navigation):
    goal = navigation.goal

    for _ in range(5):
        x = tf.random.normal((2, 1))

        model = navigation.get_quadratic_final_cost(x)
        l = model.l
        l_x = model.l_x
        l_xx = model.l_xx

        assert isinstance(l, tf.Tensor)
        assert l.shape == []
        assert tf.reduce_all(l == navigation.final_cost(x))

        assert isinstance(l_x, tf.Tensor)
        assert l_x.shape == x.shape
        assert tf.reduce_all(l_x == 2 * (x - goal))

        assert isinstance(l_xx, tf.Tensor)
        assert l_xx.shape == [x.shape[0]] * 2
        assert tf.reduce_all(l_xx == 2 * tf.eye(2))
