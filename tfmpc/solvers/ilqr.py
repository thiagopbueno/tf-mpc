"""
Iterative Linear Quadratic Regulator (iLQR)

Please see 'Synthesis and stabilization of complex behaviors
through online trajectory optimization' (IROS, 2012)
for algorithmic details and notation.
"""

import tensorflow as tf


class iLQR:

    def __init__(self, env):
        self.env = env

        self.Q_x = tf.Variable(tf.zeros([env.state_size, 1]), trainable=False)
        self.Q_u = tf.Variable(tf.zeros([env.action_size, 1]), trainable=False)

        self.Q_xx = tf.Variable(tf.zeros([env.state_size, env.state_size]), trainable=False)
        self.Q_uu = tf.Variable(tf.zeros([env.action_size, env.action_size]), trainable=False)
        self.Q_ux = tf.Variable(tf.zeros([env.action_size, env.state_size]), trainable=False)
        self.Q_xu = tf.Variable(tf.zeros([env.state_size, env.action_size]), trainable=False)

        self.V_x = tf.Variable(tf.zeros([env.state_size, 1]), trainable=False)
        self.V_xx = tf.Variable(tf.zeros([env.state_size, env.state_size]), trainable=False)

    @tf.function
    def start(self, x0, T):
        states = tf.TensorArray(dtype=tf.float32, size=T+1)
        actions = tf.TensorArray(dtype=tf.float32, size=T)

        states = states.write(0, x0)

        for t in tf.range(T):
            state = states.read(t)
            action = tf.random.uniform(x0.shape, minval=-1.0, maxval=1.0)
            state = self.env.transition(state, action)

            actions = actions.write(t, action)
            states = states.write(t+1, state)

        return states.stack(), actions.stack()

    @tf.function
    def backward(self, states, actions):
        T = states.shape[0] - 1

        K = tf.TensorArray(dtype=tf.float32, size=T)
        k = tf.TensorArray(dtype=tf.float32, size=T)

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

            K_t = -tf.matmul(tf.linalg.inv(self.Q_uu), self.Q_ux)
            k_t = -tf.matmul(tf.linalg.inv(self.Q_uu), self.Q_u)

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

        states = states.write(0, x[0])

        state = x[0]
        for t in tf.range(T):
            action = u[t] + k[t] + tf.matmul(K[t], state - x[t])
            state = self.env.transition(state, action)

            actions = actions.write(t, action)
            states = states.write(t + 1, state)

        return states.stack(), actions.stack()
