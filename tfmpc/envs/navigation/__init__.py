import gym
import numpy as np
import tensorflow as tf

from tfmpc.envs.diffenv import DiffEnv
from tfmpc.envs.gymenv import GymEnv


class Navigation(DiffEnv, GymEnv):

    def __init__(self, goal, deceleration, low, high):
        super().__init__()

        self.goal = goal
        self.deceleration = deceleration

        self.obs_space = gym.spaces.Box(
            shape=goal.shape,
            low=-np.inf,
            high=np.inf
        )

        self.action_space = gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
        )

    @property
    def action_size(self):
        return self.state_size

    @property
    def state_size(self):
        return self.goal.shape[0]

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool)
    ])
    def transition(self, state, action, cec):
        lambda_ = self._deceleration(state)
        lambda_ = tf.reshape(lambda_, [-1, 1, 1])

        p = state + lambda_ * action

        if cec:
            next_state = p
        else:
            noise = tf.random.truncated_normal(tf.shape(p), mean=0.0, stddev=0.2)
            next_state = p + noise

        return next_state

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32)
    ])
    def cost(self, state, action):
        state = tf.squeeze(state, axis=-1)
        goal = tf.squeeze(self.goal)
        return tf.reduce_sum((state - goal) ** 2, axis=-1)

    @tf.function
    def final_cost(self, state):
        state = tf.squeeze(state, axis=-1)
        goal = tf.squeeze(self.goal)
        return tf.reduce_sum((state - goal) ** 2, axis=-1)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32)
    ])
    def _deceleration(self, state):
        center = self.deceleration["center"]
        decay = self.deceleration["decay"]

        center = tf.expand_dims(center, axis=0)
        decay = tf.expand_dims(decay, axis=0)
        state = tf.expand_dims(state, axis=1)

        delta = state - center
        delta = tf.squeeze(delta, axis=-1)
        distance = tf.norm(delta, axis=-1)

        lambdas = 2 / (1.0 + tf.exp(-decay * distance)) - 1.0
        return tf.reduce_prod(lambdas,  axis=-1)

    def __repr__(self):
        goal = self.goal.numpy().squeeze().tolist()

        low = self.action_space.low.squeeze().tolist()
        high = self.action_space.high.squeeze().tolist()
        bounds = f"[{low}, {high}]"

        center = self.deceleration["center"].numpy().tolist()
        decay = ", ".join([f"{d:.4f}" for d in self.deceleration["decay"].numpy().tolist()])
        deceleration = f"{{center={center}, decay=[{decay}]}}"

        return f"Navigation(goal={goal}, deceleration={deceleration}, bounds={bounds})"

    @classmethod
    def load(cls, config):
        goal = tf.constant(config["goal"])
        low = config["low"]
        high = config["high"]
        deceleration = {
            key: tf.constant(val)
            for key, val in config["deceleration"].items()
        }
        return cls(goal, deceleration, low, high)
