# pylint: disable=missing-docstring

import numpy as np
import pytest
import tensorflow as tf

from tfmpc.problems import make_lqr, make_lqr_linear_navigation


@pytest.fixture
def lqr():
    state_size = np.random.randint(2, 10)
    action_size = np.random.randint(2, 10)
    return make_lqr(state_size, action_size)


@pytest.fixture
def nav():
    goal = np.array([[8.0], [9.0]])
    beta = 3.0
    return make_lqr_linear_navigation(goal, beta)


def test_make_lqr(lqr):
    state_size = lqr.state_size
    action_size = lqr.action_size
    n_dim = state_size + action_size

    F = lqr.F
    assert isinstance(F, tf.Tensor)
    assert F.shape == (state_size, n_dim)

    f = lqr.f
    assert isinstance(f, tf.Tensor)
    assert f.shape == (state_size, 1)


    C = lqr.C
    assert isinstance(C, tf.Tensor)
    assert C.shape == (n_dim, n_dim)

    C = C.numpy()
    assert np.allclose(C.T, C, atol=1e-4)
    np.linalg.cholesky(C)
    assert np.all(np.linalg.eigvals(C) > 0)

    c = lqr.c
    assert isinstance(c, tf.Tensor)
    assert c.shape == (n_dim, 1)


def test_make_lqr_linear_navigation(nav):
    state_size = nav.state_size
    action_size = nav.action_size
    n_dim = state_size + action_size

    F = nav.F
    assert isinstance(F, tf.Tensor)
    assert F.shape == (state_size, n_dim)

    f = nav.f
    assert isinstance(f, tf.Tensor)
    assert f.shape == (state_size, 1)


    C = nav.C
    assert isinstance(C, tf.Tensor)
    assert C.shape == (n_dim, n_dim)

    C = C.numpy()
    assert np.allclose(C.T, C, atol=1e-4)
    np.linalg.cholesky(C)
    assert np.all(np.linalg.eigvals(C) > 0)

    c = nav.c
    assert isinstance(c, tf.Tensor)
    assert c.shape == (n_dim, 1)
