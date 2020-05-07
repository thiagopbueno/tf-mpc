import tensorflow as tf


def projected_newton_qp(H, q, low, high, x, eps=1e-6, alpha_0=1.0, rho=0.5, c=1e-4):

    def f(x):
        quadratic = tf.matmul(x, tf.matmul(H, x), transpose_a=True)
        linear = tf.matmul(q, x, transpose_a=True)
        return tf.squeeze(1 / 2 * quadratic + linear)

    while True:
        g = q + tf.matmul(H, x)
        free, clamped = _get_qp_indices(g, low, high, x)

        delta_x = tf.Variable(tf.zeros_like(x))
        H_ff, g_f = _get_newton_step(H, q, x, free, clamped, delta_x)

        if tf.norm(g_f) < eps:
            break

        alpha = _armijo_line_search(f, x, g, delta_x, alpha_0, rho, c)
        x = tf.clip_by_value(x + alpha * delta_x, low, high)

    return x, H_ff, free, clamped


def _armijo_line_search(f, x, g, delta_x, alpha_0=1.0, rho=0.5, c=1e-4):
    grad = tf.squeeze(tf.matmul(g, delta_x, transpose_a=True))

    alpha = alpha_0

    while True:
        lhs = f(x + alpha * delta_x)
        rhs = f(x) + c * alpha * grad

        if lhs <= rhs:
            break

        alpha *= rho

    return alpha


@tf.function
def _get_qp_indices(g, low, high, x, eps=1e-6):
    c = tf.logical_or(
            tf.logical_and(tf.abs(x - low) < eps, g > 0),
            tf.logical_and(tf.abs(high - x) < eps, g < 0))
    f = tf.logical_not(c)
    return f, c


@tf.function
def _get_newton_step(H, q, x, f, c, delta_x):
    x_f = tf.expand_dims(x[f], -1)
    x_c = tf.expand_dims(x[c], -1)

    q_f = tf.expand_dims(q[f], -1)

    f = tf.cast(f, tf.int32)
    c = tf.cast(c, tf.int32)
    dim_f = tf.math.count_nonzero(f)
    dim_c = tf.math.count_nonzero(c)

    ff = tf.cast(tf.matmul(f, f, transpose_b=True), tf.bool)
    H_ff = tf.reshape(H[ff], [dim_f, dim_f])

    fc = tf.cast(tf.matmul(f, c, transpose_b=True), tf.bool)
    H_fc = tf.reshape(H[fc], [dim_f, dim_c])

    g_f = q_f
    if dim_f > 0:
        g_f += tf.matmul(H_ff, x_f)
    if dim_c > 0:
        g_f += tf.matmul(H_fc, x_c)

    H_ff_inv = tf.linalg.inv(H_ff)
    delta_x_f = -tf.matmul(H_ff_inv, g_f)
    delta_x_f = tf.reshape(delta_x_f, [-1])

    indices = tf.where(tf.cast(f, tf.bool))
    delta_x.scatter_nd_update(indices, delta_x_f)

    return H_ff_inv, g_f
