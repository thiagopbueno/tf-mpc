# pylint: disable=missing-docstring

import numpy as np
import pytest

from tfmpc.problems import make_lqr
from tfmpc.solvers.lqr import solve


@pytest.fixture
def lqr():
    state_size = np.random.randint(2, 4)
    action_size = np.random.randint(2, 4)
    return make_lqr(state_size, action_size)


def test_solve(lqr):
    x0 = np.random.normal(size=(lqr.state_size, 1))
    T = 10
    x, u, c = solve(lqr, x0, T)
    assert len(x) == len(u) + 1 == len(c) + 1


