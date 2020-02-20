"""
Linear Quadratic Regulator (LQR):

Please see http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf
for notation and more details on LQR.
"""

import tensorflow as tf


def solve(lqr, x0, T):
    policy, value_fn = backward(lqr, T)
    states, actions, costs = forward(lqr, x0, T, policy)
    return states, actions, costs


def backward(lqr, T):
    policy, value_fn = [], []

    state_size = lqr.state_size

    F, f, C, c = lqr.F, lqr.f, lqr.C, lqr.c

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

    F, f, C, c = lqr.F, lqr.f, lqr.C, lqr.c

    state = tf.constant(x0, dtype=tf.float32)

    for t in range(T):
        K, k = policy[t]
        action = tf.matmul(K, state) + k

        next_state = lqr.transition(state, action)
        cost = lqr.cost(state, action)

        state = next_state

        states.append(next_state.numpy())
        actions.append(action.numpy())
        costs.append(cost.numpy())

    return states, actions, costs
