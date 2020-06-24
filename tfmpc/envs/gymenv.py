import gym.core
import tensorflow as tf


class GymEnv(gym.core.Env):

    def __init__(self):
        self._t = None
        self._state = None
        self._info = {}

    def setup(self, initial_state, horizon):
        self.initial_state = initial_state
        self.horizon = horizon

    def step(self, action):
        self._t += 1

        cec = tf.constant(False)
        state = tf.expand_dims(self._state, axis=0)
        action = tf.expand_dims(action, axis=0)

        next_state = tf.squeeze(self.transition(state, action, cec), axis=0)
        cost = tf.squeeze(self.cost(state, action))
        done = (self._t == self.horizon)

        self._info["total_cost"] += cost
        info = self._info

        self._state = next_state

        return (next_state, cost, done, info)

    def reset(self):
        self._t = 0
        self._state = self.initial_state
        self._info = {
            "total_cost": 0.0
        }

        return self._state

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass
