# pylint: disable=missing-docstring

import numpy as np
import pytest
import tensorflow as tf

from tfmpc.problems import make_lqr
from tfmpc.problems.lqr import LQR


@pytest.fixture
def lqr():
    state_size = np.random.randint(2, 10)
    action_size = np.random.randint(2, 10)
    return make_lqr(state_size, action_size)


def test_make_lqr(lqr):
    state_size = lqr.state_size
    action_size = lqr.action_size
    n_dim = state_size + action_size

    F = lqr.F
    assert isinstance(F, tf.Variable)
    assert F.shape == (state_size, n_dim)

    f = lqr.f
    assert isinstance(f, tf.Variable)
    assert f.shape == (state_size, 1)

    C = lqr.C
    assert isinstance(C, tf.Variable)
    assert C.shape == (n_dim, n_dim)

    C = C.numpy()
    assert np.allclose(C.T, C, atol=1e-2)
    np.linalg.cholesky(C)
    assert np.all(np.linalg.eigvals(C) > 0)

    c = lqr.c
    assert isinstance(c, tf.Variable)
    assert c.shape == (n_dim, 1)


def test_backward(lqr):
    T = 10
    policy, value_fn = lqr.backward(T)
    assert len(policy) == len(value_fn) == T


def test_forward(lqr):
    T = 10
    x0 = np.random.normal(size=(lqr.state_size, 1)).astype("f")

    policy, value_fn = lqr.backward(T)

    x, u, c = lqr.forward(policy, x0, T)
    assert len(x) == len(u) + 1 == len(c) + 1
    assert np.allclose(x[0], x0, atol=1e-2)

    F_t = lqr.F.numpy()
    f_t = lqr.f.numpy()
    C_t = lqr.C.numpy()
    c_t = lqr.c.numpy()

    for t in range(T):
        K, k = policy[t]
        action = np.dot(K, x[t]) + k
        assert np.allclose(action, u[t], atol=1e-2)

        inputs = np.concatenate([x[t], u[t]], axis=0)
        next_state = np.dot(F_t, inputs) + f_t
        assert np.allclose(next_state, x[t + 1], atol=1e-2)

        cost = 1 / 2 * np.dot(np.dot(inputs.T, C_t), inputs) + np.dot(c_t.T, inputs)
        assert np.allclose(cost, c[t], atol=1e-2)

        V, v, const = value_fn[t]
        V = V.numpy()
        v = v.numpy()
        const = const.numpy()

        x_t = x[t].numpy()
        value = const + 1 / 2 * np.dot(np.dot(x_t.T, V), x_t) + np.dot(v.T, x_t)
        cost_to_go = np.sum(c[t:])
        assert np.allclose(value, cost_to_go, atol=1e-2)


def test_dump_and_load(lqr):
    tmpfile = "/tmp/config.json"

    with open(tmpfile, "w") as file:
        lqr.dump(file)

    with open(tmpfile, "r") as file:
        lqr2 = LQR.load(file)

    for k in lqr.__dict__:
        assert tf.reduce_all(getattr(lqr, k) == getattr(lqr2, k))
