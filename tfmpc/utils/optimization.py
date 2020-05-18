import tensorflow as tf

import tensorflow.compat.v1.logging as tf_logging


def projected_newton_qp(H, q, low, high, x, eps=1e-6, alpha_0=1.0, rho=0.5, c=1e-4):

    def f(x):
        quadratic = tf.matmul(x, tf.matmul(H, x), transpose_a=True)
        linear = tf.matmul(q, x, transpose_a=True)
        return tf.squeeze(1 / 2 * quadratic + linear)

    max_iterations = 100
    rtol = 1e-8
    step_dec = 0.6
    min_step = 1e-22
    armijo = 0.1

    clamped = tf.zeros_like(x)

    # x = tf.clip_by_value(x, low, high)
    value = f(x)

    for iteration in range(max_iterations):
        tf_logging.debug(f"[boxQP] x = {x.numpy().tolist()}")

        if iteration > 0 and (old_value - value) < rtol * tf.abs(old_value):
            tf_logging.debug(f"[boxQP] Improvement smaller than tolerance.")
            break

        old_value = value
        old_clamped = clamped

        g = q + tf.matmul(H, x)
        free, clamped = _get_qp_indices(g, low, high, x)

        factorize = (iteration == 0 or tf.reduce_any(old_clamped != clamped))
        tf_logging.debug(f"[boxQP] factorize = {factorize}")

        if factorize:
            free_int = tf.cast(free, tf.int32)
            dim_f = tf.math.count_nonzero(free_int)

            ff = tf.cast(tf.matmul(free_int, free_int, transpose_b=True), tf.bool)
            H_ff = tf.reshape(H[ff], [dim_f, dim_f])

            try:
                Hfree = tf.linalg.cholesky(H_ff)
            except tf.errors.InvalidArgumentError as e:
                tf_logging.error(f"[boxQP] Hessian is not positive definite.")
                break

        if tf.reduce_all(clamped):
            tf_logging.debug("[boxQP] All dimensions are clamped.")
            break

        # check convergence
        grad_norm = tf.norm(g[free])
        tf_logging.debug(f"[boxQP] grad_norm = {grad_norm}")
        if grad_norm < eps:
            tf_logging.debug("[boxQP] Gradient norm smaller than tolerance.")
            break

        # get descent direction
        grad_clamped = q + tf.matmul(H, x * tf.cast(clamped, tf.float32))
        search = tf.Variable(tf.zeros_like(x))
        indices = tf.where(free)
        x_free = tf.expand_dims(x[free], axis=-1)
        grad_clamped_free = tf.expand_dims(grad_clamped[free], axis=-1)
        values = -tf.linalg.cholesky_solve(Hfree, grad_clamped_free) - x_free
        values = tf.reshape(values, [-1])
        search.scatter_nd_update(indices, values)

        # check for descent direction
        sdotg = tf.squeeze(tf.matmul(search, g, transpose_a=True))
        tf_logging.debug(f"[boxQP] sdotg = {sdotg}")
        if sdotg >= 0:
            tf_logging.debug(f"[boxQP] not a descent direction.")
            break

        # armijo linesearch
        step = 1
        nstep = 0
        xc = tf.clip_by_value(x + step * search, low, high)
        vc = f(xc)
        while (vc - old_value) / (step * sdotg) < armijo:
            tf_logging.debug(f"[boxQP] xc = {xc.numpy().tolist()}, vc = {vc}")
            tf_logging.debug(f"[boxQP] nstep = {nstep}, step = {step}, step_dec = {step_dec}, min_step = {min_step}")
            step *= step_dec
            nstep += 1
            xc = tf.clip_by_value(x + step * search, low, high)
            vc = f(xc)
            if step < min_step:
                tf_logging.debug("[boxQP] Maximum line-search iterations exceeded.")
                break

        # accept candidate
        x = xc
        value = vc

    return x, Hfree, free, clamped


# def _armijo_line_search(f, x, g, delta_x, alpha_0=1.0, rho=0.5, c=1e-4):
#     grad = tf.squeeze(tf.matmul(g, delta_x, transpose_a=True))

#     alpha = alpha_0

#     while True:
#         lhs = f(x + alpha * delta_x)
#         rhs = f(x) + c * alpha * grad

#         if lhs <= rhs:
#             break

#         alpha *= rho

#     return alpha


@tf.function
def _get_qp_indices(g, low, high, x, eps=1e-6):
    c = tf.logical_or(
            tf.logical_and(tf.abs(x - low) < eps, g > 0),
            tf.logical_and(tf.abs(high - x) < eps, g < 0))
    f = tf.logical_not(c)
    return f, c


# @tf.function
# def _get_newton_step(H, q, x, f, c, delta_x):
#     x_f = tf.expand_dims(x[f], -1)
#     x_c = tf.expand_dims(x[c], -1)

#     q_f = tf.expand_dims(q[f], -1)

#     f = tf.cast(f, tf.int32)
#     c = tf.cast(c, tf.int32)
#     dim_f = tf.math.count_nonzero(f)
#     dim_c = tf.math.count_nonzero(c)

#     ff = tf.cast(tf.matmul(f, f, transpose_b=True), tf.bool)
#     H_ff = tf.reshape(H[ff], [dim_f, dim_f])

#     fc = tf.cast(tf.matmul(f, c, transpose_b=True), tf.bool)
#     H_fc = tf.reshape(H[fc], [dim_f, dim_c])

#     g_f = q_f
#     if dim_f > 0:
#         g_f += tf.matmul(H_ff, x_f)
#     if dim_c > 0:
#         g_f += tf.matmul(H_fc, x_c)

#     H_ff_inv = tf.linalg.inv(H_ff)
#     delta_x_f = -tf.matmul(H_ff_inv, g_f)
#     delta_x_f = tf.reshape(delta_x_f, [-1])

#     indices = tf.where(tf.cast(f, tf.bool))
#     delta_x.scatter_nd_update(indices, delta_x_f)

#     return H_ff_inv, g_f
