#!/usr/bin/env python

import tensorflow as tf
from tensorflow import logging

import conf.config
from SCAN import SCAN

FLAGS = tf.app.flags.FLAGS
logging.set_verbosity(tf.logging.INFO)


def main(_):
    with tf.Session() as sess:
        cfg = FLAGS
        scan = SCAN(sess, cfg)
        if FLAGS.is_train:
            logging.info("start training...")
            scan.train()
            logging.info("train finished.")
        else:
            scan.inference()


if __name__ == "__main__":
    tf.app.run()
