# pylint: disable=missing-docstring

import numpy as np
import pytest
import tensorflow as tf

from tfmpc.problems import make_lqr_linear_navigation
from tfmpc.solvers.lqr import backward, forward


@pytest.fixture
def lqr():
    goal = np.array([[8.32], [-5.5]])
    beta = 10.0
    return make_lqr_linear_navigation(goal, beta)


def test_backward(lqr):
    T = 10
    policy, value_fn = backward(lqr, T)
    assert len(policy) == len(value_fn)


def test_forward(lqr):
    T = 10
    x0 = np.random.normal(size=(lqr.state_size, 1))

    policy, _ = backward(lqr, T)

    x, u, c = forward(lqr, x0, T, policy)
    assert len(x) == len(u) + 1 == len(c) + 1
    assert np.allclose(x[0], x0, atol=1e-4)

    F_t = lqr.F.numpy()
    f_t = lqr.f.numpy()
    C_t = lqr.C.numpy()
    c_t = lqr.c.numpy()

    for t in range(T):
        K, k = policy[t]
        action = np.dot(K, x[t]) + k
        assert np.allclose(action, u[t], atol=1e-4)

        inputs = np.concatenate([x[t], u[t]], axis=0)
        next_state = np.dot(F_t, inputs) + f_t
        assert np.allclose(next_state, x[t + 1], atol=1e-4)

        cost = 1 / 2 * np.dot(np.dot(inputs.T, C_t), inputs) + np.dot(c_t.T, inputs)
        assert np.allclose(cost, c[t], atol=1e-4)
