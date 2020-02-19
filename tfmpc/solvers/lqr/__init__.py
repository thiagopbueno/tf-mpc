import tensorflow as tf

import numpy as np


def backward(lqr, T):
    policy, value_fn = [], []

    n_dim = lqr.n_dim
    state_size = lqr.state_size

    F = tf.constant(lqr.F, dtype=tf.float32)
    f = tf.constant(lqr.f, dtype=tf.float32)
    C = tf.constant(lqr.C, dtype=tf.float32)
    c = tf.constant(lqr.c, dtype=tf.float32)

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


def forward(lqr, x0, T, policy):
    states = [x0]
    actions = []
    costs = []

    F = tf.constant(lqr.F, dtype=tf.float32)
    f = tf.constant(lqr.f, dtype=tf.float32)
    C = tf.constant(lqr.C, dtype=tf.float32)
    c = tf.constant(lqr.c, dtype=tf.float32)

    state = tf.constant(x0, dtype=tf.float32)
    for t in range(T):
        K, k = policy[t]
        action = tf.matmul(K, state) + k
        assert action.shape == (lqr.action_size, 1)

        inputs = tf.concat([state, action], axis=0)

        next_state = tf.matmul(F, inputs) + f

        cost = 1 / 2 * tf.matmul(tf.matmul(tf.transpose(inputs), C), inputs) + \
               tf.matmul(tf.transpose(inputs), c)

        state = next_state
        assert state.shape == (lqr.state_size, 1)

        states.append(next_state.numpy())
        actions.append(action.numpy())
        costs.append(cost.numpy())

    return states, actions, costs
