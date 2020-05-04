import tensorflow as tf

from tfmpc.envs import DiffEnv


class Navigation(DiffEnv):

    def __init__(self, goal, beta):
        self.goal = goal
        self.beta = beta

    @property
    def action_size(self):
        return self.state_size

    @property
    def state_size(self):
        return self.goal.shape[0]

    @tf.function
    def transition(self, state, action):
        return state + action

    @tf.function
    def cost(self, state, action):
        c1 = tf.reduce_sum((state - self.goal) ** 2)
        c2 = tf.reduce_sum(action ** 2)
        return c1 + self.beta * c2
