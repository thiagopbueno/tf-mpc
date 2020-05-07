import gym
import numpy as np
import tensorflow as tf

from tfmpc.envs import DiffEnv


class Navigation(DiffEnv):

    def __init__(self, goal, beta, low=None, high=None):
        self.goal = goal
        self.beta = beta

        if low is None:
            low = -np.inf
        if high is None:
            high = np.inf

        self.action_space = gym.spaces.Box(low, high, shape=tf.shape(goal))

        self.obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=tf.shape(goal))

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
