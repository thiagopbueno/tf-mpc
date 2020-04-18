import tensorflow as tf


class LQR:

    def __init__(self, F, f, C, c):
        self.F = tf.constant(F, dtype=tf.float32)
        self.f = tf.constant(f, dtype=tf.float32)
        self.C = tf.constant(C, dtype=tf.float32)
        self.c = tf.constant(c, dtype=tf.float32)

    @property
    def n_dim(self):
        return self.F.shape[1]

    @property
    def state_size(self):
        return self.F.shape[0]

    @property
    def action_size(self):
        return self.n_dim - self.state_size

    @tf.function
    def transition(self, x, u):
        inputs = tf.concat([x, u], axis=0)
        return tf.matmul(self.F, inputs) + self.f

    @tf.function
    def cost(self, x, u):
        inputs = tf.concat([x, u], axis=0)
        inputs_transposed = tf.transpose(inputs)
        return 1 / 2 * tf.matmul(tf.matmul(inputs_transposed, self.C), inputs) + \
               tf.matmul(inputs_transposed, self.c)

    def backward(self, T):
        policy, value_fn = [], []

        state_size = self.state_size

        F, f, C, c = self.F, self.f, self.C, self.c

        V = tf.zeros((state_size, state_size))
        v = tf.zeros((state_size, 1))

        for t in reversed(range(T)):
            F_trans_V = tf.matmul(tf.transpose(F), V)
            Q = C + tf.matmul(F_trans_V, F)
            q = c + \
                tf.matmul(F_trans_V, f) + \
                tf.matmul(tf.transpose(F), v)

            Q_uu = Q[state_size:, state_size:]
            Q_ux = Q[state_size:, :state_size]
            q_u = q[state_size:]
            inv_Q_uu = tf.linalg.inv(Q_uu)

            K = -tf.matmul(inv_Q_uu, Q_ux)
            k = -tf.matmul(inv_Q_uu, q_u)

            Q_xx = Q[:state_size, :state_size]
            Q_xu = Q[:state_size, state_size:]
            q_x = q[:state_size]
            K_Q_uu = tf.matmul(tf.transpose(K), Q_uu)

            V = Q_xx + \
                tf.matmul(Q_xu, K) + \
                tf.matmul(tf.transpose(K), Q_ux) + \
                tf.matmul(K_Q_uu, K)

            v = q_x + \
                tf.matmul(Q_xu, k) + \
                tf.matmul(tf.transpose(K), q_u) + \
                tf.matmul(K_Q_uu, k)

            policy.append((K, k))
            value_fn.append((V, v))

        policy = list(reversed(policy))
        value_fn = list(reversed(value_fn))

        return policy, value_fn

    def forward(self, policy, x0, T):
        states = [x0]
        actions = []
        costs = []

        F, f, C, c = self.F, self.f, self.C, self.c

        state = tf.constant(x0, dtype=tf.float32)

        for t in range(T):
            K, k = policy[t]
            action = tf.matmul(K, state) + k

            next_state = self.transition(state, action)
            cost = self.cost(state, action)

            state = next_state

            states.append(next_state.numpy())
            actions.append(action.numpy())
            costs.append(cost.numpy())

        return states, actions, costs
