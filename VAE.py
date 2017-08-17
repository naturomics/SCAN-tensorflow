import tensorflow as tf


class Decoder(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, input):
        with tf.varible_scope("decoder"):
            out = input

            return(out)


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
            out = input

            return(out)
