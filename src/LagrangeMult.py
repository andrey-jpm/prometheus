from tensorflow.keras.layers import Layer
import tensorflow as tf

class LagrangeMult(Layer):
    def __init__(self, flav_dim, **kwargs):
        super(LagrangeMult, self).__init__(**kwargs)
        w_init = tf.keras.initializers.Constant(value=1)
        self.w = tf.Variable(
            initial_value=w_init(shape=(flav_dim), dtype="float32"),
            trainable=False,
        )

    def call(self, inputs):
        return inputs*self.w