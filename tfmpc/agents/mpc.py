import numpy as np
import tensorflow as tf

class MPC:

    def __init__(self, solver, horizon):
        self.solver = solver
        self.horizon = horizon

    def __call__(self, state, timestep):
        steps_to_go = self.horizon - timestep

        trajectory, iterations = self.solver.solve(state, steps_to_go)
        self.solver.logger.log(timestep, {"iterations": iterations})

        action = trajectory[0].action
        action = tf.constant(action[..., np.newaxis], dtype=tf.float32)
        return action
