from collections import namedtuple
import multiprocessing as mp

import tensorflow as tf


TransitionApprox = namedtuple("TransitionApprox", "f f_x f_u")
CostApprox = namedtuple("CostApprox", "l l_x l_u l_xx l_uu l_ux l_xu")
FinalCostApprox = namedtuple("CostApprox", "l l_x l_xx")


class DiffEnv:

    @tf.function
    def get_linear_transition(self, state, action):
        with tf.GradientTape() as tape:
            tape.watch(state)
            tape.watch(action)
            f = self.transition(state, action)

        f_x, f_u = tape.jacobian(f, [state, action])
        f_x = tf.squeeze(f_x)
        f_u = tf.squeeze(f_u)

        return TransitionApprox(f, f_x, f_u)

    @tf.function
    def get_quadratic_cost(self, state, action):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(state)
            tape.watch(action)
            l = self.cost(state, action)
            l_x, l_u = tape.gradient(
                l, [state, action],
                unconnected_gradients=tf.UnconnectedGradients.ZERO)

        l_xx, l_xu = tape.jacobian(
            l_x, [state, action],
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        l_xx = tf.squeeze(l_xx)
        l_xu = tf.squeeze(l_xu)

        l_ux, l_uu = tape.jacobian(
            l_u, [state, action],
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        l_ux = tf.squeeze(l_ux)
        l_uu = tf.squeeze(l_uu)

        del tape

        return CostApprox(l, l_x, l_u, l_xx, l_uu, l_ux, l_xu)

    @tf.function
    def get_quadratic_final_cost(self, state):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(state)
            l = self.final_cost(state)
            l_x = tape.gradient(l, state)

        l_xx = tape.jacobian(l_x, state)
        l_xx = tf.squeeze(l_xx)

        del tape

        return FinalCostApprox(l, l_x, l_xx)

    def _get_lq_model(self, inp):
        state, action = inp
        transition_model = self.get_linear_transition(state, action)
        cost_model = self.get_quadratic_cost(state, action)
        return transition_model, cost_model

    #@tf.function
    def get_linear_quadratic_model(self, states, actions, processes=2):
        states = tf.unstack(states[:-1])
        actions = tf.unstack(actions)
        with mp.get_context("spawn").Pool(processes) as pool:
            models = pool.map(self._get_lq_model, zip(states, actions))
            return models
