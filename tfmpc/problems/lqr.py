import json

import numpy as np
import tensorflow as tf


class LQR:

    def __init__(self, F, f, C, c):
        self.F = tf.Variable(F, trainable=False, dtype=tf.float32, name="F")
        self.f = tf.Variable(f, trainable=False, dtype=tf.float32, name="f")
        self.C = tf.Variable(C, trainable=False, dtype=tf.float32, name="C")
        self.c = tf.Variable(c, trainable=False, dtype=tf.float32, name="c")

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

    @tf.function
    def backward(self, T):
        policy, value_fn = [], []

        state_size = self.state_size

        F, f, C, c = self.F, self.f, self.C, self.c

        V = tf.zeros((state_size, state_size))
        v = tf.zeros((state_size, 1))
        const = 0.0

        value_fn.append((V, v, const))

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

            V_, v_, _ = value_fn[-1]

            k_trans = tf.transpose(k)
            F_trans = tf.transpose(F)
            f_trans = tf.transpose(f)
            V_f = tf.matmul(V_, f)

            W = C + tf.matmul(F_trans, tf.matmul(V_, F))
            w = c + tf.matmul(F_trans, V_f) + tf.matmul(F_trans, v_)
            W_uu = W[state_size:, state_size:]
            w_u = w[state_size:]

            const1 = 1 / 2 * tf.matmul(k_trans, tf.matmul(W_uu, k))
            const2 = tf.matmul(k_trans, w_u)
            const3 = 1/ 2 * tf.matmul(f_trans, V_f) + tf.matmul(f_trans, v_)
            const += (const1 + const2 + const3)

            policy.append((K, k))
            value_fn.append((V, v, const))

        policy = list(reversed(policy))
        value_fn = list(reversed(value_fn[1:]))

        return policy, value_fn

    @tf.function
    def forward(self, policy, x0, T):
        states = [x0]
        actions = []
        costs = []

        F, f, C, c = self.F, self.f, self.C, self.c

        state = x0

        for t in range(T):
            K, k = policy[t]
            action = tf.matmul(K, state) + k

            next_state = self.transition(state, action)
            cost = self.cost(state, action)

            state = next_state

            states.append(next_state)
            actions.append(action)
            costs.append(cost)

        states = tf.stack(states, axis=0)
        actions = tf.stack(actions, axis=0)
        costs = tf.stack(costs, axis=0)

        return states, actions, costs

    def dump(self, file):
        config = {
            "F": self.F.numpy().tolist(),
            "f": self.f.numpy().tolist(),
            "C": self.C.numpy().tolist(),
            "c": self.c.numpy().tolist()
        }
        json.dump(config, file)

    @classmethod
    def load(cls, file):
        config = json.load(file)
        config = {k: np.array(v).astype("f") for k, v in config.items()}
        return cls(**config)
