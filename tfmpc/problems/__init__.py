import numpy as np
from sklearn.datasets import make_spd_matrix


from tfmpc.problems.lqr import LQR


def make_lqr(state_size, action_size):
    n_dim = state_size + action_size

    F = np.random.normal(size=(state_size, n_dim))
    f = np.random.normal(size=(state_size, 1))

    C = make_spd_matrix(n_dim)
    c = np.random.normal(size=(n_dim, 1))

    return LQR(F, f, C, c)


def make_lqr_linear_navigation(goal, beta):
    state_size = action_size = goal.shape[0]

    F = np.concatenate([np.identity(state_size)] * action_size, axis=1).astype("f")
    f = np.zeros((state_size, 1)).astype("f")

    C = np.diag([2.0] * state_size + [2.0 * beta] * action_size).astype("f")
    c = np.concatenate([-2.0 * goal, np.zeros((action_size, 1))], axis=0).astype("f")

    return LQR(F, f, C, c)
