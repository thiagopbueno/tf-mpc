import tensorflow as tf


class LQR:

    def __init__(self, F, f, C, c):
        self.F = tf.constant(F, dtype=tf.float32)
        self.f = tf.constant(f, dtype=tf.float32)
        self.C = tf.constant(C, dtype=tf.float32)
        self.c = tf.constant(c, dtype=tf.float32)

    @property
    def n_dim(self):
        return self.F.shape[1]

    @property
    def state_size(self):
        return self.F.shape[0]

    @property
    def action_size(self):
        return self.n_dim - self.state_size

    @tf.function
    def transition(self, x, u):
        inputs = tf.concat([x, u], axis=0)
        return tf.matmul(self.F, inputs) + self.f

    @tf.function
    def cost(self, x, u):
        inputs = tf.concat([x, u], axis=0)
        inputs_transposed = tf.transpose(inputs)
        return 1 / 2 * tf.matmul(tf.matmul(inputs_transposed, self.C), inputs) + \
               tf.matmul(inputs_transposed, self.c)

