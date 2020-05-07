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

        self.Q_x = tf.Variable(tf.zeros([env.state_size, 1]), trainable=False)
        self.Q_u = tf.Variable(tf.zeros([env.action_size, 1]), trainable=False)

        self.Q_xx = tf.Variable(tf.zeros([env.state_size, env.state_size]), trainable=False)
        self.Q_uu = tf.Variable(tf.zeros([env.action_size, env.action_size]), trainable=False)
        self.Q_ux = tf.Variable(tf.zeros([env.action_size, env.state_size]), trainable=False)
        self.Q_xu = tf.Variable(tf.zeros([env.state_size, env.action_size]), trainable=False)

        self.Q_uu_reg = tf.Variable(tf.zeros([env.action_size, env.action_size]), trainable=False)
        self.Q_xu_reg = tf.Variable(tf.zeros([env.state_size, env.action_size]), trainable=False)

        self.V_x = tf.Variable(tf.zeros([env.state_size, 1]), trainable=False)
        self.V_xx = tf.Variable(tf.zeros([env.state_size, env.state_size]), trainable=False)

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
    def backward(self, states, actions):
        T = states.shape[0] - 1

        K = tf.TensorArray(dtype=tf.float32, size=T)
        k = tf.TensorArray(dtype=tf.float32, size=T)

        self.V_x.assign(tf.zeros([self.env.state_size, 1]))
        self.V_xx.assign(tf.zeros([self.env.state_size, self.env.state_size]))

        for t in tf.range(T - 1, -1, -1):
            state, action = states[t], actions[t]

            transition_model = self.env.get_linear_transition(state, action)
            f_x = transition_model.f_x
            f_u = transition_model.f_u

            cost_model = self.env.get_quadratic_cost(state, action)
            l_x = cost_model.l_x
            l_u = cost_model.l_u
            l_xx = cost_model.l_xx
            l_uu = cost_model.l_uu
            l_ux = cost_model.l_ux
            l_xu = cost_model.l_xu

            f_x_trans = tf.transpose(f_x)
            f_u_trans = tf.transpose(f_u)

            self.Q_x.assign(l_x + tf.matmul(f_x_trans, self.V_x))
            self.Q_u.assign(l_u + tf.matmul(f_u_trans, self.V_x))

            f_x_trans_V_xx = tf.matmul(f_x_trans, self.V_xx)
            f_u_trans_V_xx = tf.matmul(f_u_trans, self.V_xx)

            self.Q_xx.assign(l_xx + tf.matmul(f_x_trans_V_xx, f_x))
            self.Q_uu.assign(l_uu + tf.matmul(f_u_trans_V_xx, f_u))
            self.Q_ux.assign(l_ux + tf.matmul(f_u_trans_V_xx, f_x))
            self.Q_xu.assign(l_xu + tf.matmul(f_x_trans_V_xx, f_u))

            Q_reg, mu = tf.py_function(self._regularization_scheduler, inp=[self.Q_uu, f_u, f_u], Tout=[tf.float32, tf.float32])
            self.Q_uu.assign(Q_reg)

            Q_reg, = tf.py_function(self._regularize, inp=[mu, self.Q_ux, f_u, f_x], Tout=[tf.float32])
            self.Q_ux.assign(Q_reg)

            Q_reg, = tf.py_function(self._regularize, inp=[mu, self.Q_xu, f_x, f_u], Tout=[tf.float32])
            self.Q_xu.assign(Q_reg)

            K_t, k_t = tf.py_function(self._get_controller, inp=[action, self.Q_uu, self.Q_u, self.Q_ux], Tout=[tf.float32, tf.float32])

            K_t_trans_Q_uu = tf.matmul(tf.transpose(K_t), self.Q_uu)

            self.V_x.assign(self.Q_x
                            + tf.matmul(self.Q_xu, k_t)
                            + tf.matmul(tf.transpose(K_t), self.Q_u)
                            + tf.matmul(K_t_trans_Q_uu, k_t))
            self.V_xx.assign(self.Q_xx
                             + tf.matmul(self.Q_xu, K_t)
                             + tf.matmul(tf.transpose(K_t), self.Q_ux)
                             + tf.matmul(K_t_trans_Q_uu, K_t))

            K = K.write(t, K_t)
            k = k.write(t, k_t)

        return K.stack(), k.stack()

    @tf.function
    def forward(self, x, u, K, k):
        T = x.shape[0] - 1

        states = tf.TensorArray(dtype=tf.float32, size=T + 1)
        actions = tf.TensorArray(dtype=tf.float32, size=T)
        costs = tf.TensorArray(dtype=tf.float32, size=T)

        states = states.write(0, x[0])

        state = x[0]
        for t in tf.range(T):
            action = u[t] + k[t] + tf.matmul(K[t], state - x[t])
            cost = self.env.cost(state, action)
            state = self.env.transition(state, action)

            actions = actions.write(t, action)
            costs = costs.write(t, cost)
            states = states.write(t + 1, state)

        return states.stack(), actions.stack(), costs.stack()

    def solve(self, x0, T):
        x_hat, u_hat = self.start(x0, T)

        iterations = 0
        converged = False

        while not converged:
            K, k = self.backward(x_hat, u_hat)
            x, u, c = self.forward(x_hat, u_hat, K, k)

            if tf.reduce_max(tf.abs(x - x_hat)) < self.atol:
                converged = True

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
                # print(f">> Q is updated with mu={mu}")
                Q_reg = self._regularize(mu, Q, f1, f2)

        return Q_reg, mu

    def _regularize(self, mu, Q, f1, f2):
        return Q + mu * tf.matmul(f1, f2, transpose_a=True)
