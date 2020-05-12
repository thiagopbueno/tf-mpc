import pytest
import tensorflow as tf

from tfmpc.utils import optimization


@pytest.fixture(params=[
    ([0.0, 0.0], [-1.0, 0.5], [1.0, 1.0], [0.0, 0.5]),
    ([0.0, 0.0], [0.5, -1.0], [1.0, 1.0], [0.5, 0.0]),
    ([1.0, 1.0], [0.0, 1.5], [2.0, 2.0], [1.0, 1.5]),
    ([1.0, 1.0], [1.5, 0.0], [2.0, 2.0], [1.5, 1.0]),

    ([0.0, 0.0, 0.0], [-1.0, 0.5, -1.0], [1.0, 1.0, 1.0], [0.0, 0.5, 0.0]),
    ([0.0, 0.0, 0.0], [-1.0, 0.5, 0.30], [1.0, 1.0, 1.0], [0.0, 0.5, 0.30]),
])
def qp(request):
    goal, low, high, x_star = request.param

    dim = len(goal)

    H = 2 * tf.eye(dim)
    q = -2 * tf.constant(goal, shape=(dim, 1))
    low = tf.constant(low, shape=(dim, 1))
    high = tf.constant(high, shape=(dim, 1))
    x_star = tf.constant(x_star, shape=(dim, 1))

    return H, q, low, high, x_star


def test_get_qp_indices(qp):
    H, q, low, high, x_star = qp

    for _ in range(10):
        x = tf.random.uniform(
            tf.shape(x_star), minval=low + 1e-4, maxval=high - 1e-4)

        g = q + tf.matmul(H, x)
        f, c = optimization._get_qp_indices(g, low, high, x)

        assert tf.reduce_all(f == tf.constant([[True]] * len(x)))
        assert tf.reduce_all(c == tf.constant([[False]] * len(x)))


# def test_get_newton_step(qp):
#     H, q, low, high, x_star = qp

#     for _ in range(10):
#         x = tf.random.uniform(tf.shape(x_star), minval=low, maxval=high)

#         g = q + tf.matmul(H, x)
#         f, c = optimization._get_qp_indices(g, low, high, x)

#         delta_x = tf.Variable(tf.zeros_like(x))
#         H_ff, g_f = optimization._get_newton_step(H, q, x, f, c, delta_x)


def test_projected_newton_qp(qp):
    H, q, low, high, x_star = qp

    x, *_ = optimization.projected_newton_qp(H, q, low, high, x_star)
    assert tf.reduce_all(tf.abs(x - x_star) < 1e-4)

    for _ in range(10):
        x_0 = tf.random.uniform(tf.shape(x_star), minval=low, maxval=high)

        x, *_ = optimization.projected_newton_qp(H, q, low, high, x_0)
        assert tf.reduce_all(tf.abs(x - x_star) < 1e-4)

        for i in range(x_0.shape[0]):
            x = tf.Variable(x_0, trainable=False)
            x[i].assign(low[i])
            x, *_ = optimization.projected_newton_qp(H, q, low, high, x)
            assert tf.reduce_all(tf.abs(x - x_star) < 1e-4)

        for i in range(x_0.shape[0]):
            x = tf.Variable(x_0, trainable=False)
            x[i].assign(high[i])
            x, *_ = optimization.projected_newton_qp(H, q, low, high, x)
            assert tf.reduce_all(tf.abs(x - x_star) < 1e-4)
