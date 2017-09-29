import tensorflow as tf


class Encoder(object):
    def __init__(self, input, scale):
        self.input = input
        self.scale = scale

    def setup_model(self):
        self.input = self.input + self.scale * tf.random_normal()
        self.h1 = tf.contrib.layers.conv2d(self.input, num_outputs=32,
                                           kernel_size=4, strde=2,
                                           activation_fn=tf.nn.elu)
        self.h2 = tf.contrib.layers.conv2d(self.h1, num_outputs=32,
                                           kernel_size=4, strde=2,
                                           activation_fn=tf.nn.elu)
        self.h3 = tf.contrib.layers.conv2d(self.h2, num_outputs=64,
                                           kernel_size=4, strde=2,
                                           activation_fn=tf.nn.elu)
        self.h4 = tf.contrib.layers.conv2d(self.h3, num_outputs=64,
                                           kernel_size=4, strde=2,
                                           activation_fn=tf.nn.elu)


class Decoder(object):
    def __init__(self, input):
        self.input = input

    def setup_model(self):
        self.h1 = tf.contrib.layers.conv2d_transpose(self.input,
                                                     num_outputs=64,
                                                     kernel_size=4, stride=2,
                                                     activation_fn=tf.nn.elu)
        self.h2 = tf.contrib.layers.conv2d_transpose(self.h1, num_outputs=64,
                                                     kernel_size=4, stride=2,
                                                     activation_fn=tf.nn.elu)
        self.h3 = tf.contrib.layers.conv2d_transpose(self.h2, num_outputs=32,
                                                     kernel_size=4, stride=2,
                                                     activation_fn=tf.nn.elu)
        self.h4 = tf.contrib.layers.conv2d_transpose(self.h3, num_outputs=32,
                                                     kernel_size=4, stride=2,
                                                     activation_fn=tf.nn.elu)


if __name__ == "__main__":
    pass
