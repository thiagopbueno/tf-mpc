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
    x0 = np.random.normal(size=(lqr.state_size, 1)).astype("f")
    T = 10
    trajectory = solve(lqr, x0, T)
    assert len(trajectory.states) == len(trajectory.actions) + 1
    assert len(trajectory.states) == len(trajectory.costs) + 1
