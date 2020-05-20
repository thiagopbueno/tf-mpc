import numpy as np
import pytest
import tensorflow as tf

from tfmpc.envs import navigation_lqr


@pytest.fixture
def env():
    goal = tf.constant([[8.0], [-9.0]])
    beta = 5.0
    return navigation_lqr.NavigationLQR(goal, beta)


def test_transition(env):
    for _ in range(5):
        x0 = tf.random.normal((2, 1))
        u0 = tf.random.uniform((2, 1))

        x1 = env.transition(x0, u0)

        assert isinstance(x1, tf.Tensor)
        assert x1.shape == x0.shape
        assert all(x1 == x0 + u0)


def test_linear_transition(env):
    for _ in range(5):
        x = tf.random.normal((2, 1))
        u = tf.random.uniform((2, 1))

        model = env.get_linear_transition(x, u, batch=False)
        f = model.f
        f_x = model.f_x
        f_u = model.f_u

        assert isinstance(f, tf.Tensor)
        assert f.shape == x.shape
        assert tf.reduce_all(f == env.transition(x, u))

        assert isinstance(f_x, tf.Tensor)
        assert f_x.shape == [x.shape[0]] * 2
        assert tf.reduce_all(f_x == tf.eye(2))

        assert isinstance(f_u, tf.Tensor)
        assert f_u.shape == [u.shape[0]] * 2
        assert tf.reduce_all(f_u == tf.eye(2))


def test_cost(env):
    goal = env.goal
    beta = env.beta

    for _ in range(5):
        x = tf.random.normal((2, 1))
        u = tf.random.uniform((2, 1))

        c = env.cost(x, u)

        assert isinstance(c, tf.Tensor)
        assert c.shape == []
        assert tf.reduce_all(
            c == tf.reduce_sum((x - goal) ** 2) + beta * tf.reduce_sum(u ** 2)
        )


def test_quadratic_cost(env):
    goal = env.goal
    beta = env.beta

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
        assert l_x.shape == x.shape
        assert tf.reduce_all(l_x == 2 * (x - goal))

        assert isinstance(l_u, tf.Tensor)
        assert l_u.shape == u.shape
        assert tf.reduce_all(l_u == 2 * beta * u)

        assert isinstance(l_xx, tf.Tensor)
        assert l_xx.shape == [x.shape[0]] * 2
        assert tf.reduce_all(l_xx == 2 * tf.eye(2))

        assert isinstance(l_uu, tf.Tensor)
        assert l_uu.shape == [u.shape[0]] * 2
        assert tf.reduce_all(l_uu == 2 * beta * tf.eye(2))

        assert isinstance(l_ux, tf.Tensor)
        assert l_ux.shape == (x.shape[0], u.shape[0])
        assert tf.reduce_all(l_ux == tf.zeros((x.shape[0], u.shape[0])))

        assert isinstance(l_xu, tf.Tensor)
        assert l_xu.shape == (u.shape[0], x.shape[0])
        assert tf.reduce_all(l_xu == tf.zeros((u.shape[0], x.shape[0])))


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
