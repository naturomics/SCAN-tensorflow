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
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('beta_vae', 50, 'hyperparameter beta for VAE objective')
flags.DEFINE_float('beta_scan', 1, 'hyperparameter beta for SCAN objective')
flags.DEFINE_float('lambda_scan', 10, 'hyperparameter lambda for SCAN objective')
flags.DEFINE_integer('epoch', 1000, 'epoch for training')

###########################
#   environment setting   #
###########################
flags.DEFINE_integer('seed', 1178, 'seed for random number generation')
flags.DEFINE_boolean('is_train', True, 'train or inference')
