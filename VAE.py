import tensorflow as tf


class Decoder(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, input):
        with tf.varible_scope("decoder"):
            h1 = tf.contrib.layers.conv2d_transpose(input, num_outputs=64,
                                                    kernel_size=4, stride=2)
            h2 = tf.contrib.layers.conv2d_transpose(h1, num_outputs=64,
                                                    kernel_size=4, stride=2)
            h3 = tf.contrib.layers.conv2d_transpose(h2, num_outputs=32,
                                                    kernel_size=4, stride=2)
            h4 = tf.contrib.layers.conv2d_transpose(h3, num_outputs=32,
                                                    kernel_size=4, stride=2)

            return(h4)


class Encoder(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, input):
        """
        Args:
            input: image Tensors.

        Returns:
            out: output.
        """
        with tf.variable_scope("encoder"):
            h1 = tf.contrib.layers.conv2d(input, num_outputs=32,
                                          kernel_size=4, stride=2)
            h2 = tf.contrib.layers.conv2d(h1, num_outputs=32,
                                          kernel_size=4, stride=2)
            h3 = tf.contrib.layers.conv2d(h2, num_outputs=64,
                                          kernel_size=4, stride=2)
            h4 = tf.contrib.layers.conv2d(h3, num_outputs=64,
                                          kernel_size=4, stride=2)
            h5 = tf.contrib.layers.fully_connected(h4, num_outputs=256)

            return(h5)
