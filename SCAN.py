import tensorflow as tf

from VAE import Encoder, Decoder


class SCAN(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.img_encoder = Encoder()
        self.img_decoder = Decoder()
        self.sym_encoder = Encoder()
        self.sym_decoder = Decoder()

    def train(self):
        img, sym = self.read_data_sets()

        with tf.variable_scope("beta_VAE"):
            img_q_mu, img_q_sigma = self.img_encoder(img)
            sym_q_mu, sym_q_sigma = self.sym_encoder(sym)

            img_z = tf.exp(img_q_sigma) + img_q_mu
            sym_z = tf.exp(sym_q_sigma) + sym_q_mu

            self.img_decoder(img_z)
            self.sym_decoder(sym_z)

        with tf.variable_scope("SCAN"):
            pass

    def inference(self):
        pass

    def read_data_sets(self):
        """
        Returns:
            data queues of image and symbol.
        """
        return()
