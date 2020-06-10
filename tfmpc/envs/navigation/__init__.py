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
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
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

    @tf.function
    def transition(self, state, action, batch=False, cec=True):
        lambda_ = self._deceleration(state, batch)
        if batch:
            lambda_ = tf.reshape(lambda_, [tf.shape(lambda_)[0], 1, 1])

        p = state + lambda_ * action

        if cec:
            next_state = p
        else:
            next_state = p + tf.random.truncated_normal(tf.shape(p), mean=0.0, stddev=0.2)

        return next_state

    @tf.function
    def cost(self, state, action, batch=False):
        state = tf.squeeze(state, axis=-1)
        goal = tf.squeeze(self.goal)
        return tf.reduce_sum((state - goal) ** 2, axis=-1)

    @tf.function
    def final_cost(self, state, batch=False):
        state = tf.squeeze(state, axis=-1)
        goal = tf.squeeze(self.goal)
        return tf.reduce_sum((state - goal) ** 2, axis=-1)

    @tf.function
    def _deceleration(self, state, batch=False):
        center = self.deceleration["center"]
        decay = self.deceleration["decay"]

        if batch:
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
