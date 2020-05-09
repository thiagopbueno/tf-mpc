import gym
import json
import numpy as np
import tensorflow as tf

from tfmpc.envs.diffenv import DiffEnv


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

    @tf.function
    def final_cost(self, state):
        return tf.reduce_sum((state - self.goal) ** 2)

    @classmethod
    def load(cls, path):
        with open(path, "r") as file:
            config = json.load(file)

        goal = config["goal"]
        state_dim = len(goal)
        goal = tf.constant(goal, shape=(state_dim, 1), dtype=tf.float32)
        beta = tf.constant(config["beta"], dtype=tf.float32)
        low = config.get("low")
        high = config.get("high")

        return cls(goal, beta, low, high)

    def __repr__(self):
        goal = self.goal.numpy().squeeze().tolist()
        beta = self.beta
        bounds = ""
        if self.action_space.is_bounded():
            low = self.action_space.low.squeeze().tolist()
            high = self.action_space.high.squeeze().tolist()
            bounds = f"bounds=[{low}, {high}]"
        return f"Navigation(goal={goal}, beta={beta}, {bounds})"
