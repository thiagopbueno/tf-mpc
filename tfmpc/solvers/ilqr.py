"""
Iterative Linear Quadratic Regulator (iLQR)

Please see 'Synthesis and stabilization of complex behaviors
through online trajectory optimization' (IROS, 2012)
for algorithmic details and notation.
"""

import numpy as np
import pprint
import time
import tensorflow as tf
import tensorflow.compat.v1.logging as tf_logging

from tfmpc.utils import optimization
from tfmpc.utils import trajectory


class iLQR:

    def __init__(self, env, atol=1e-4, c1=0.0, alpha_min=1e-3, max_iterations=100):
        self.env = env

        self.atol = atol

        self.c1 = c1
        self.alpha_min = alpha_min

        self.max_iterations = max_iterations

    @tf.function
    def start(self, x0, T):
        states = tf.TensorArray(dtype=tf.float32, size=T+1)
        actions = tf.TensorArray(dtype=tf.float32, size=T)

        low = tf.constant(self.env.action_space.low)
        minval = tf.where(tf.math.is_inf(low), -tf.ones_like(low), low)

        high = tf.constant(self.env.action_space.high)
        maxval = tf.where(tf.math.is_inf(high), tf.ones_like(high), high)

        states = states.write(0, x0)

        for t in tf.range(T):
            state = states.read(t)
            action = tf.random.uniform([], minval=minval, maxval=maxval)
            state = self.env.transition(state, action)

            actions = actions.write(t, action)
            states = states.write(t+1, state)

        return states.stack(), actions.stack()

    def derivatives(self, states, actions):
        start = time.time()
        transition_model = self.env.get_linear_transition(states[:-1], actions, batch=True)
        cost_model = self.env.get_quadratic_cost(states[:-1], actions, batch=True)
        final_cost_model = self.env.get_quadratic_final_cost(states[-1])
        uptime = time.time() - start
        tf_logging.info(f"[DERIVATIVES] uptime = {uptime:.6f}")

        return transition_model, cost_model, final_cost_model

    @tf.function
    def backward(self, T, actions, transition_model, cost_model, final_cost_model, mu=1.0):
        state_size = self.env.state_size

        K = tf.TensorArray(dtype=tf.float32, size=T)
        k = tf.TensorArray(dtype=tf.float32, size=T)

        V_x = final_cost_model.l_x
        V_xx = final_cost_model.l_xx

        J = final_cost_model.l
        dV1 = 0.0
        dV2 = 0.0

        for t in tf.range(T - 1, -1, -1):
            f_x = transition_model.f_x[t]
            f_u = transition_model.f_u[t]

            l = cost_model.l[t]
            l_x = cost_model.l_x[t]
            l_u = cost_model.l_u[t]
            l_xx = cost_model.l_xx[t]
            l_uu = cost_model.l_uu[t]
            l_xu = cost_model.l_xu[t]

            f_x_trans = tf.transpose(f_x)
            f_u_trans = tf.transpose(f_u)

            Q_x = l_x + tf.matmul(f_x_trans, V_x)
            Q_u = l_u + tf.matmul(f_u_trans, V_x)

            f_x_trans_V_xx = tf.matmul(f_x_trans, V_xx)
            f_u_trans_V_xx = tf.matmul(f_u_trans, V_xx)
            f_u_trans_V_xx_reg = tf.matmul(f_u_trans, V_xx + mu * tf.eye(state_size))

            Q_xx = l_xx + tf.matmul(f_x_trans_V_xx, f_x)
            Q_uu = l_uu + tf.matmul(f_u_trans_V_xx, f_u)
            Q_ux = tf.transpose(l_xu) + tf.matmul(f_u_trans_V_xx, f_x)

            Q_uu_reg = l_uu + tf.matmul(f_u_trans_V_xx_reg, f_u)
            Q_ux_reg = tf.transpose(l_xu) + tf.matmul(f_u_trans_V_xx_reg, f_x)

            if self.env.action_space.is_bounded():
                K_t, k_t = tf.py_function(self._get_constrained_controller, inp=[actions[t], Q_uu_reg, Q_ux_reg, Q_u], Tout=[tf.float32, tf.float32])
            else:
                K_t, k_t = self._get_unconstrained_controller(Q_uu_reg, Q_ux_reg, Q_u)

            K_t_trans = tf.transpose(K_t)
            k_t_trans = tf.transpose(k_t)
            K_t_trans_Q_uu = tf.matmul(K_t_trans, Q_uu)

            V_x = tf.reshape(
                (Q_x
                 + tf.matmul(Q_ux, k_t, transpose_a=True)
                 + tf.matmul(K_t_trans, Q_u)
                 + tf.matmul(K_t_trans_Q_uu, k_t)),
                [self.env.state_size, 1])

            V_xx = tf.reshape(
                (Q_xx
                 + tf.matmul(Q_ux, K_t, transpose_a=True)
                 + tf.matmul(K_t_trans, Q_ux)
                 + tf.matmul(K_t_trans_Q_uu, K_t)),
                [self.env.state_size, self.env.state_size])
            V_xx = 1 / 2 * (V_xx + tf.transpose(V_xx))

            J += l

            dV1 += tf.reshape(tf.squeeze(tf.matmul(k_t_trans, Q_u)), [])
            dV2 += tf.reshape(1 / 2 * tf.squeeze(tf.matmul(tf.matmul(k_t_trans, Q_uu), k_t)), [])

            K = K.write(t, K_t)
            k = k.write(t, k_t)

        return K.stack(), k.stack(), J, dV1, dV2

    @tf.function
    def forward(self, x, u, K, k, alpha=1.0):
        T = x.shape[0] - 1

        states = tf.TensorArray(size=T+1, dtype=tf.float32)
        costs = tf.TensorArray(size=T+1, dtype=tf.float32)
        actions = tf.TensorArray(size=T, dtype=tf.float32)

        states = states.write(0, x[0])

        J = 0.0

        state = x[0]
        residual = tf.constant(0.0)

        for t in tf.range(T):
            delta_x = state - x[t]
            delta_u = alpha * k[t] + tf.matmul(K[t], delta_x)

            action = u[t] + delta_u
            cost = self.env.cost(state, action)
            state = self.env.transition(state, action)

            actions = actions.write(t, action)
            costs = costs.write(t, cost)
            states = states.write(t + 1, state)

            J += cost
            residual = tf.math.maximum(residual, tf.reduce_max(tf.abs(delta_u)))

        final_cost = self.env.final_cost(state)
        costs = costs.write(T, final_cost)
        J += final_cost

        return states.stack(), actions.stack(), costs.stack(), J, residual

    def solve(self, x0, T):
        x_hat, u_hat = self.start(x0, T)

        for iteration in range(self.max_iterations):
            tf_logging.info(f"[SOLVE] Iteration = {iteration}")

            # model approximation
            transition_model, cost_model, final_cost_model = self.derivatives(x_hat, u_hat)

            # backward pass
            K, k, J_hat, dV1, dV2 = self._backward(T, u_hat, transition_model, cost_model, final_cost_model)

            # forward pass
            x_hat, u_hat, c_hat, residual = self._forward(x_hat, u_hat, J_hat, K, k, dV1, dV2)

            # convergence test
            if residual < self.atol:
                break

        traj = trajectory.Trajectory(x_hat, u_hat, c_hat)

        return traj, iteration

    def _backward(self, T, u_hat, transition_model, cost_model, final_cost_model):
        mu = 0.0
        mu_min = 1e-6
        mu_old = None
        delta = 1.0
        delta_0 = 2.0

        optimally_regularized = False

        while not optimally_regularized:
            tf_logging.debug(f"[BACKWARD] mu = {mu}, delta = {delta}")

            try:
                start = time.time()
                K, k, J_hat, dV1, dV2 = self.backward(T, u_hat, transition_model, cost_model, final_cost_model, mu)
                uptime = time.time() - start

                tf_logging.info(f"[BACKWARD] uptime = {uptime:.4f} sec")
                tf_logging.info(f"[BACKWARD] J_hat = {J_hat:.4f}")

                if mu == 0.0:
                    optimally_regularized = True
                elif not optimally_regularized:
                    mu_old = mu
                    delta = min(1 / delta_0, delta / delta_0)
                    mu = mu * delta * (mu * delta > mu_min)

            except tf.errors.InvalidArgumentError as e:
                tf_logging.warn(f"[BACKWARD] could not run iLQR.backward : {e}")

                if mu_old:
                    mu = mu_old
                    optimally_regularized = True
                    continue

                delta = max(delta_0, delta * delta_0)
                mu = max(mu_min, mu * delta)

            except Exception as e:
                tf_logging.fatal(f"[BACKWARD] Error: {e}")
                exit(-1)

        return K, k, J_hat, dV1, dV2

    def _forward(self, x_hat, u_hat, J_hat, K, k, dV1, dV2):

        for alpha in np.geomspace(1.0, self.alpha_min, 11):
            tf_logging.debug(f"[FORWARD] alpha = {alpha}")

            start = time.time()
            x, u, c, J, residual = self.forward(x_hat, u_hat, K, k, alpha)
            uptime = time.time() - start

            tf_logging.info(f"[FORWARD] uptime = {uptime:.4f} sec")
            tf_logging.info(f"[FORWARD] J = {J:.4f}")
            tf_logging.debug(f"[FORWARD] residual = {residual}")

            if residual < self.atol:
                break

            delta_J = - alpha * (dV1 + alpha * dV2)
            dcost = J_hat - J

            if delta_J > 0:
                z = dcost / delta_J
            else:
                z = tf.sign(dcost)
                tf_logging.warn(f"[FORWARD] Non-positive expected reduction: delta_J = {delta_J:.4f}")

            tf_logging.debug(f"[FORWARD] dcost = {dcost}, delta_J = {delta_J:.4f}")
            tf_logging.debug(f"[FORWARD] z = {z}, c1 = {self.c1}")

            if z >= self.c1:
                accept = True
                break

        return x, u, c, residual

    def _get_unconstrained_controller(self, Q_uu, Q_ux, Q_u):
        R = tf.linalg.cholesky(Q_uu)
        kK = -tf.linalg.cholesky_solve(R, tf.concat([Q_u, Q_ux], axis=1))
        k = kK[:,:1]
        K = kK[:,1:]
        return K, k

    def _get_constrained_controller(self, u, Q_uu, Q_ux, Q_u):
        low = self.env.action_space.low - u
        high = self.env.action_space.high - u

        k_0 = tf.zeros_like(u)
        k, Hfree, free, clamped = optimization.projected_newton_qp(Q_uu, Q_u, low, high, k_0)

        action_dim = self.env.action_size
        state_dim = self.env.state_size
        K = tf.Variable(tf.zeros([action_dim, state_dim]))

        n_free = tf.math.count_nonzero(tf.squeeze(free))
        if n_free > 0:
            free = tf.squeeze(free)
            Q_ux_f = Q_ux[free]
            indices = tf.where(free)
            values = - tf.linalg.cholesky_solve(Hfree, Q_ux_f)
            K.scatter_nd_update(indices, values)

        K = tf.constant(K.numpy())

        return K, k
