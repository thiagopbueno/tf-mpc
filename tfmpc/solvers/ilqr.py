"""
Iterative Linear Quadratic Regulator (iLQR)

Please see 'Synthesis and stabilization of complex behaviors
through online trajectory optimization' (IROS, 2012)
for algorithmic details and notation.
"""

import tensorflow as tf

from tfmpc.utils import optimization
from tfmpc.utils import trajectory


class iLQR:

    def __init__(self, env, atol=1e-4):
        self.env = env
        self.atol = atol

        self.V_x = tf.Variable(tf.zeros([env.state_size, 1]), trainable=False)
        self.V_xx = tf.Variable(tf.zeros([env.state_size, env.state_size]), trainable=False)

        self.J = tf.Variable(tf.zeros([]), trainable=False)
        self.delta_J = tf.Variable(tf.zeros([]), trainable=False)

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
            action = tf.random.uniform(x0.shape, minval=minval, maxval=maxval)
            state = self.env.transition(state, action)

            actions = actions.write(t, action)
            states = states.write(t+1, state)

        return states.stack(), actions.stack()

    @tf.function
    def backward(self, states, actions, alpha=1.0):
        T = states.shape[0] - 1

        K = tf.TensorArray(dtype=tf.float32, size=T)
        k = tf.TensorArray(dtype=tf.float32, size=T)

        final_cost_model = self.env.get_quadratic_final_cost(states[-1])
        l_x = final_cost_model.l_x
        l_xx = final_cost_model.l_xx
        self.V_x.assign(l_x)
        self.V_xx.assign(l_xx)

        self.J.assign(tf.zeros(shape=[]))
        self.delta_J.assign(tf.zeros(shape=[]))

        for t in tf.range(T - 1, -1, -1):
            state, action = states[t], actions[t]

            transition_model = self.env.get_linear_transition(state, action)
            f_x = transition_model.f_x
            f_u = transition_model.f_u

            cost_model = self.env.get_quadratic_cost(state, action)
            l = cost_model.l
            l_x = cost_model.l_x
            l_u = cost_model.l_u
            l_xx = cost_model.l_xx
            l_uu = cost_model.l_uu
            l_ux = cost_model.l_ux
            l_xu = cost_model.l_xu

            f_x_trans = tf.transpose(f_x)
            f_u_trans = tf.transpose(f_u)

            Q_x = l_x + tf.matmul(f_x_trans, self.V_x)
            Q_u = l_u + tf.matmul(f_u_trans, self.V_x)

            f_x_trans_V_xx = tf.matmul(f_x_trans, self.V_xx)
            f_u_trans_V_xx = tf.matmul(f_u_trans, self.V_xx)

            Q_xx = l_xx + tf.matmul(f_x_trans_V_xx, f_x)
            Q_uu = l_uu + tf.matmul(f_u_trans_V_xx, f_u)
            Q_ux = l_ux + tf.matmul(f_u_trans_V_xx, f_x)
            Q_xu = l_xu + tf.matmul(f_x_trans_V_xx, f_u)

            Q_uu, mu = tf.py_function(self._regularization_scheduler, inp=[Q_uu, f_u, f_u], Tout=[tf.float32, tf.float32])
            Q_ux, = tf.py_function(self._regularize, inp=[mu, Q_ux, f_u, f_x], Tout=[tf.float32])
            Q_xu, = tf.py_function(self._regularize, inp=[mu, Q_xu, f_x, f_u], Tout=[tf.float32])

            K_t, k_t = tf.py_function(self._get_controller, inp=[action, Q_uu, Q_u, Q_ux], Tout=[tf.float32, tf.float32])

            K_t_trans_Q_uu = tf.matmul(tf.transpose(K_t), Q_uu)

            self.V_x.assign(Q_x
                            + tf.matmul(Q_xu, k_t)
                            + tf.matmul(tf.transpose(K_t), Q_u)
                            + tf.matmul(K_t_trans_Q_uu, k_t))
            self.V_xx.assign(Q_xx
                             + tf.matmul(Q_xu, K_t)
                             + tf.matmul(tf.transpose(K_t), Q_ux)
                             + tf.matmul(K_t_trans_Q_uu, K_t))

            self.J.assign_add(l)

            d1 = alpha * tf.squeeze(tf.matmul(k_t, Q_u, transpose_a=True))
            self.delta_J.assign_add(d1)

            d2 = (alpha ** 2) / 2 * tf.squeeze(tf.matmul(tf.matmul(k_t, Q_uu, transpose_a=True), k_t))
            self.delta_J.assign_add(d2)

            K = K.write(t, K_t)
            k = k.write(t, k_t)

        return K.stack(), k.stack(), self.J, self.delta_J

    @tf.function
    def forward(self, x, u, K, k, alpha=1.0):
        T = x.shape[0] - 1

        states = tf.TensorArray(dtype=tf.float32, size=T + 1)
        actions = tf.TensorArray(dtype=tf.float32, size=T)
        costs = tf.TensorArray(dtype=tf.float32, size=T)

        states = states.write(0, x[0])

        self.J.assign(tf.zeros([]))

        state = x[0]
        residual = tf.constant(0.0)

        for t in tf.range(T):
            delta_x = state - x[t]

            residual = tf.math.maximum(residual, tf.reduce_max(tf.abs(delta_x)))

            action = u[t] + alpha * k[t] + tf.matmul(K[t], delta_x)
            cost = self.env.cost(state, action)
            state = self.env.transition(state, action)

            actions = actions.write(t, action)
            costs = costs.write(t, cost)
            states = states.write(t + 1, state)

            self.J.assign_add(cost)

        return states.stack(), actions.stack(), costs.stack(), self.J, residual

    def solve(self, x0, T):
        x_hat, u_hat = self.start(x0, T)

        iterations = 0
        converged = False

        alpha = 1.0
        rho = 0.5
        c1 = 0.1

        while not converged:
            K, k, J, delta_J = self.backward(x_hat, u_hat)

            # improved line search
            accept = False
            while not accept:
                x, u, c, J_hat, residual = self.forward(x_hat, u_hat, K, k, alpha)

                if residual < self.atol:
                    converged = True
                    break

                if delta_J < 0:
                    z = (J_hat - J) / delta_J
                    if z >= c1:
                        accept = True
                    else:
                        alpha *= rho
                else:
                    tf.assert_equal(delta_J, 0.0)
                    accept = True

            x_hat, u_hat = x, u
            iterations += 1

        traj = trajectory.Trajectory(x_hat, u_hat, c)

        return traj, iterations

    def _get_controller(self, u, Q_uu, Q_u, Q_ux):
        if not self.env.action_space.is_bounded():
            K = -tf.matmul(tf.linalg.inv(Q_uu), Q_ux)
            k = -tf.matmul(tf.linalg.inv(Q_uu), Q_u)
        else:
            low = self.env.action_space.low - u
            high = self.env.action_space.high - u

            k_0 = tf.zeros_like(u)
            k, Q_uu_f_inv, f, c = optimization.projected_newton_qp(
                Q_uu, Q_u, low, high, k_0)

            action_dim = self.env.action_size
            state_dim = self.env.state_size
            K = tf.Variable(tf.zeros([action_dim, state_dim]))

            n_free = tf.math.count_nonzero(tf.squeeze(f))
            if n_free > 0:
                Q_ux_f = Q_ux[tf.squeeze(f)]

                indices = tf.where(tf.squeeze(f))
                values = -tf.matmul(Q_uu_f_inv, Q_ux_f)
                K.scatter_nd_update(indices, values)

            K = tf.constant(K.numpy())

        return K, k

    def _regularization_scheduler(self, Q, f1, f2):
        mu, mu_min = 0.0, 1e-6
        delta, delta_0 = 1.0, 2.0

        Q_reg = Q

        mu_list = []
        is_optimally_regularized = False
        decreased = False

        while not is_optimally_regularized:
            mu_list.append(mu)

            try:
                tf.linalg.cholesky(Q_reg)
                # print(">> Q is positive definite.")

                # positive definite
                if mu > 0.0: # decrease mu
                    delta = min(1 / delta_0, delta / delta_0)
                    mu = mu * delta if mu * delta >= mu_min else 0.0
                    decreased = True
                else:
                    # print(">> Q is optimally regularized.")
                    is_optimally_regularized = True

            except Exception as e: # not positive definite
                if decreased:
                    is_optimally_regularized = True
                    mu = mu_list[-1]
                else:
                    # increase
                    delta = max(delta_0, delta * delta_0)
                    mu = max(mu_min, mu * delta)

            if mu > 0.0:
                Q_reg = self._regularize(mu, Q, f1, f2)

        return Q_reg, mu

    def _regularize(self, mu, Q, f1, f2):
        return Q + mu * tf.matmul(f1, f2, transpose_a=True)
