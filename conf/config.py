from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


flags = tf.app.flags


###########################
#     model structure     #
###########################


###########################
#     hyper parameter     #
###########################
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_integer('epoch', 1000, 'epoch for training')

# for DAE model
flags.DEFINE_string("DAENoiseType", 'mask', "the method of adding noise in \
                    DAE model, including additive gaussian(gaussian) noise \
                    and masking noise(mask), default to masking noise")
if flags.FLAGS.DAENoiseType == "gaussian":
    flags.DEFINE_float("scale", 0.5, "scale of gaussian noise")
elif flags.FLAGS.DAENoiseType == "mask":
    flags.DEFINE_float("keep_prob", 0.8, "the keeping probability of dropout")
else:
    tf.logging.error("check for the DAE noise method in the config file, \
it should be one of gaussian and mask")
    exit(1)

# for VAE model
flags.DEFINE_float('beta_vae', 50, 'hyperparameter beta for VAE objective')

# for SCAN model
flags.DEFINE_float('beta_scan', 1, 'hyperparameter beta for SCAN objective')
flags.DEFINE_float('lambda_scan', 10, 'hyperparameter lambda \
                   for SCAN objective')


###################################
#   running environment setting   #
###################################
flags.DEFINE_integer('seed', 1178, 'seed for random number generation')
flags.DEFINE_boolean('is_train', True, 'train or inference')
