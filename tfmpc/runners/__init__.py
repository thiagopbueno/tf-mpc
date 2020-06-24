import contextlib
import time

import tensorflow as tf

from tfmpc.utils.trajectory import Trajectory


class Runner:

    def __init__(self, env, agent, **kwargs):
        self.env = env
        self.agent = agent

        self._on_step_hook = kwargs.get("on_step")
        self._on_episode_end_hook = kwargs.get("on_episode_end")

    def _on_step(self, timestep, state, action, cost, next_state, done, info):
        if self._on_step_hook:
            self._on_step_hook(timestep, state, action, cost, next_state, done, info)

    def _on_episode_end(self, trajectory, uptime):
        if self._on_episode_end:
            self._on_episode_end_hook(trajectory, uptime)

    def run(self, mode=None):
        start_time = time.perf_counter()

        state = self.env.reset()
        timestep = 0
        done = False

        states = [state]
        actions = []
        costs = []

        total_cost = 0.0

        while not done:
            action = self.agent(state, timestep)
            next_state, cost, done, info = self.env.step(action)

            total_cost += cost

            self._on_step(timestep, state, action, cost, next_state, done, info)

            if mode is not None:
                self.env.render(mode)

            state = next_state
            timestep = self.env._t

            states.append(state)
            actions.append(action)
            costs.append(cost)

        costs.append(self.env.final_cost(state))

        states = tf.stack(states)
        actions = tf.stack(actions)
        costs = tf.stack(costs)

        trajectory = Trajectory(states, actions, costs)
        uptime = time.perf_counter() - start_time

        self._on_episode_end(trajectory, uptime)

        return trajectory

    @contextlib.contextmanager
    def __call__(self, initial_state, horizon):
        self.env.setup(initial_state, horizon)
        yield self
        self.env.close()
