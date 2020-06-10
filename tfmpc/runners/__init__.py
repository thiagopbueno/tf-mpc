import contextlib

import tensorflow as tf

from tfmpc.utils import trajectory


class Runner:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self, mode=None):
        state = self.env.reset()
        timestep = 0
        done = False

        states = [state]
        actions = []
        costs = []

        while not done:
            action = self.agent(state, timestep)
            next_state, cost, done, info = self.env.step(action)

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

        return trajectory.Trajectory(states, actions, costs)

    @contextlib.contextmanager
    def __call__(self, initial_state, horizon):
        self.env.setup(initial_state, horizon)
        yield self
        self.env.close()
