import gym.core


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

        next_state = self.transition(self._state, action, cec=False)
        cost = self.cost(self._state, action)
        done = (self._t == self.horizon)
        info = self._info

        self._state = next_state

        return (next_state, cost, done, info)

    def reset(self):
        self._t = 0
        self._state = self.initial_state
        self._info = {}

        return self._state

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass
