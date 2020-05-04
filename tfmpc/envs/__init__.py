from collections import namedtuple

import tensorflow as tf


TransitionApprox = namedtuple("TransitionApprox", "f f_x f_u")
CostApprox = namedtuple("CostApprox", "l l_x l_u l_xx l_uu l_ux l_xu")


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
            l_x, l_u = tape.gradient(l, [state, action])

        l_xx, l_xu = tape.jacobian(l_x, [state, action])
        l_xx = tf.squeeze(l_xx)
        l_xu = tf.squeeze(l_xu)

        l_ux, l_uu = tape.jacobian(l_u, [state, action])
        l_ux = tf.squeeze(l_ux)
        l_uu = tf.squeeze(l_uu)

        del tape

        return CostApprox(l, l_x, l_u, l_xx, l_uu, l_ux, l_xu)
