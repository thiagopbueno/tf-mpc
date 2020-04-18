"""
Linear Quadratic Regulator (LQR):

Please see http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf
for notation and more details on LQR.
"""

import tensorflow as tf

from tfmpc.utils import trajectory


def solve(lqr, x0, T):
    policy, value_fn = lqr.backward(T)
    states, actions, costs = lqr.forward(policy, x0, T)
    return trajectory.Trajectory(states, actions, costs)
