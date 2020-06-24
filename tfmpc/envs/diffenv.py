from collections import namedtuple

import tensorflow as tf


TransitionApprox = namedtuple("TransitionApprox", "f f_x f_u")
CostApprox = namedtuple("CostApprox", "l l_x l_u l_xx l_uu l_ux l_xu")
FinalCostApprox = namedtuple("CostApprox", "l l_x l_xx")


class DiffEnv:

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32)
    ])
    def get_linear_transition(self, state, action):
        cec =  tf.constant(True)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(state)
            tape.watch(action)
            f = self.transition(state, action, cec)

        f_x = tape.batch_jacobian(f, state)
        f_u = tape.batch_jacobian(f, action)
        f_x = tf.squeeze(f_x, axis=[2, 4])
        f_u = tf.squeeze(f_u, axis=[2, 4])

        del tape

        return TransitionApprox(f, f_x, f_u)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),
    ])
    def get_quadratic_cost(self, state, action):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(state)
            tape.watch(action)
            l = self.cost(state, action)
            l_x, l_u = tape.gradient(
                l, [state, action],
                unconnected_gradients=tf.UnconnectedGradients.ZERO)

        l_xx = tape.batch_jacobian(
            l_x, state,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        l_xu = tape.batch_jacobian(
            l_x, action,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        l_xx = tf.squeeze(l_xx, axis=[2, 4])
        l_xu = tf.squeeze(l_xu, axis=[2, 4])

        l_ux = tape.batch_jacobian(
            l_u, state,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        l_uu = tape.batch_jacobian(
            l_u, action,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        l_ux = tf.squeeze(l_ux, axis=[2, 4])
        l_uu = tf.squeeze(l_uu, axis=[2, 4])

        del tape

        return CostApprox(l, l_x, l_u, l_xx, l_uu, l_ux, l_xu)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
    ])
    def get_quadratic_final_cost(self, state):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(state)
            l = self.final_cost(state)
            l_x = tape.gradient(
                l, state,
                unconnected_gradients=tf.UnconnectedGradients.ZERO)

        l_xx = tape.jacobian(
            l_x, state,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        l_xx = tf.squeeze(l_xx)

        del tape

        return FinalCostApprox(l, l_x, l_xx)
