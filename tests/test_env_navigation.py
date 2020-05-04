import pytest
import tensorflow as tf

from tfmpc.envs import navigation


@pytest.fixture
def navlin():
    goal = tf.constant([[8.0], [-9.0]])
    beta = 5.0
    return navigation.Navigation(goal, beta)


def test_transition(navlin):
    for _ in range(5):
        x0 = tf.random.normal((2, 1))
        u0 = tf.random.uniform((2, 1))

        x1 = navlin.transition(x0, u0)

        assert isinstance(x1, tf.Tensor)
        assert x1.shape == x0.shape
        assert all(x1 == x0 + u0)


def test_linear_transition(navlin):
    for _ in range(5):
        x = tf.random.normal((2, 1))
        u = tf.random.uniform((2, 1))

        model = navlin.get_linear_transition(x, u)
        f = model.f
        f_x = model.f_x
        f_u = model.f_u

        assert isinstance(f, tf.Tensor)
        assert f.shape == x.shape
        assert tf.reduce_all(f == navlin.transition(x, u))

        assert isinstance(f_x, tf.Tensor)
        assert f_x.shape == [x.shape[0]] * 2
        assert tf.reduce_all(f_x == tf.eye(2))

        assert isinstance(f_u, tf.Tensor)
        assert f_u.shape == [u.shape[0]] * 2
        assert tf.reduce_all(f_u == tf.eye(2))


def test_cost(navlin):
    goal = navlin.goal
    beta = navlin.beta

    for _ in range(5):
        x = tf.random.normal((2, 1))
        u = tf.random.uniform((2, 1))

        c = navlin.cost(x, u)

        assert isinstance(c, tf.Tensor)
        assert c.shape == []
        assert tf.reduce_all(
            c == tf.reduce_sum((x - goal) ** 2) + beta * tf.reduce_sum(u ** 2)
        )


def test_quadratic_cost(navlin):
    goal = navlin.goal
    beta = navlin.beta

    for _ in range(5):
        x = tf.random.normal((2, 1))
        u = tf.random.uniform((2, 1))

        model = navlin.get_quadratic_cost(x, u)
        l = model.l
        l_x = model.l_x
        l_u = model.l_u
        l_xx = model.l_xx
        l_uu = model.l_uu
        l_xu = model.l_xu
        l_ux = model.l_ux

        assert isinstance(l, tf.Tensor)
        assert l.shape == []
        assert tf.reduce_all(l == navlin.cost(x, u))

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
        assert l_xx.shape == (x.shape[0], u.shape[0])
        assert tf.reduce_all(l_ux == tf.zeros((x.shape[0], u.shape[0])))

        assert isinstance(l_xu, tf.Tensor)
        assert l_xx.shape == (u.shape[0], x.shape[0])
        assert tf.reduce_all(l_xu == tf.zeros((u.shape[0], x.shape[0])))
