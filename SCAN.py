import tensorflow as tf

from VAE import Encoder, Decoder


# TF 1.3 release the statistical distribution library tf.distributions,
# support for versions of TF before 1.3
try:
    distributions = tf.distributions
    kl_divergence = tf.distributions.kl_divergence
except:
    distributions = tf.contrib.distributions
    kl_divergence = tf.contrib.distributions.kl_divergence


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
            img_z = distributions.Normal(img_q_mu, img_q_sigma)
            img_gen = self.img_decoder(img_z)

            img_reconstruct_error = tf.reduce_mean(img_gen)

            img_z_prior = distributions.Normal()
            KL_divergence = kl_divergence(img_z, img_z_prior)
            KL_divergence = self.cfg.beta_vae * KL_divergence

            loss = img_reconstruct_error - KL_divergence

        # train beta VAE
        optimizer = tf.train.AdamOptimizer(self.cfg.learning_rate)
        train_op = optimizer.minimize(loss)

        for step in range(self.cfg.epoch):
            self.sess.run(train_op)

        with tf.variable_scope("SCAN"):
            sym_q_mu, sym_q_sigma = self.sym_encoder(sym)
            sym_z = distributions.Normal(sym_q_mu, sym_q_sigma)
            self.sym_decoder(sym_z)

            sym_reconstruct_error = tf.reduce_mean()

            sym_z_prior = distributions.Normal()
            beta_KL_divergence = kl_divergence(sym_z, sym_z_prior)
            beta_KL_divergence = self.cfg.beta_scan * beta_KL_divergence
            lambda_KL_divergence = kl_divergence(img_z, sym_z)

            loss = sym_reconstruct_error - beta_KL_divergence
            loss -= self.cfg.lambda_scan * lambda_KL_divergence

        # train SCAN
        optimizer = tf.train.AdamOptimizer(self.cfg.learning_rate)
        train_op = optimizer.minimize(loss)

        for step in range(self.cfg.epoch):
            self.sess.run(train_op)

    def inference(self):
        pass

    def read_data_sets(self):
        """
        Returns:
            data queues of image and symbol.
        """
        img, sym = [], []

        return(img, sym)
