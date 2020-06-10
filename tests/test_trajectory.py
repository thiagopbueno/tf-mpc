import numpy as np
import pytest
import tensorflow as tf

from tfmpc.utils.trajectory import Trajectory


@pytest.fixture
def trajectory():
    T = 10
    state_size = action_size = 3

    states = tf.random.normal(shape=(T + 1, state_size, 1))
    actions = tf.random.truncated_normal(shape=(T, action_size, 1))
    costs = tf.random.uniform(shape=(T,), minval=-1.0, maxval=1.0)

    return Trajectory(states, actions, costs)


def test_initial_state(trajectory):
    assert np.allclose(trajectory.initial_state, trajectory.states[0], atol=1e-3)


def test_final_state(trajectory):
    assert np.allclose(trajectory.final_state, trajectory.states[-1], atol=1e-3)


def test_total_cost(trajectory):
    assert np.allclose(trajectory.total_cost, sum(trajectory.costs), atol=1e-3)


def test_cumulative_cost(trajectory):
    costs = trajectory.costs
    expected = [np.sum(costs[: t + 1]) for t in range(len(costs))]
    assert np.allclose(trajectory.cumulative_cost, expected, atol=1e-3)


def test_cost_to_go(trajectory):
    costs = trajectory.costs
    expected = [np.sum(costs[t:]) for t in range(len(costs))]
    assert np.allclose(trajectory.cost_to_go, expected, atol=1e-3)


def test_len(trajectory):
    assert len(trajectory) == len(trajectory.actions)
    assert len(trajectory) == len(trajectory.costs)
    assert len(trajectory) == len(trajectory.states) - 1


def test_getitem(trajectory):
    for t, (x, u, c) in enumerate(trajectory):
        assert np.allclose(x, trajectory.states[t + 1], atol=1e-3)
        assert np.allclose(u, trajectory.actions[t], atol=1e-3)
        assert c == trajectory.costs[t]


def test_repr(trajectory):
    print()
    print(repr(trajectory))


def test_str(trajectory):
    print()
    print(str(trajectory))
