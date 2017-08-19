import tensorflow as tf


def preBlock(x):
    out = tf.layers.conv3d(x, filters=24, kernel_size=3,
                           padding="same", name="preBlk.conv3d1")
    out = tf.layers.batch_normalization(out, axis=4, name="preBlk.batchNorm1")
    out = tf.nn.relu(out, name="preBlk.relu1")
    out = tf.layers.conv3d(out, filters=24, kernel_size=3,
                           padding="same", name="preBlk.conv3d2")
    out = tf.layers.batch_normalization(out, axis=4, name="preBlk.batchNorm2")
    out = tf.nn.relu(out, name="preBlk.relu2")

    return(out)


def block(x, n_filters, scope, stride=1):
    shortcut = x
    with tf.variable_scope(scope):
        out = tf.layers.conv3d(x, n_filters, kernel_size=3,
                               strides=(stride, stride, stride),
                               padding="same", name="blk.conv3d1")
        out = tf.layers.batch_normalization(out, axis=4, name="blk.batchNorm1")
        out = tf.nn.relu(out, name="blk.relu1")
        out = tf.layers.conv3d(x, n_filters, kernel_size=3,
                               padding="same", name="blk.conv3d2")
        out = tf.layers.batch_normalization(out, axis=4, name="blk.batchNorm2")

        if stride != 1 or n_filters != x.get_shape().as_list()[-1]:
            with tf.variable_scope("shortcut"):
                shortcut = tf.layers.conv3d(shortcut, n_filters, kernel_size=1,
                                            strides=(stride, stride, stride),
                                            name="shortcut.conv3d")
                shortcut = tf.layers.batch_normalization(shortcut,
                                                         axis=4,
                                                         name="shortcut.bn")
        out = out + shortcut
        out = tf.nn.relu(out, name="blk.relu2")

    return(out)


def conv3d(inputs, filters=16):
    feature = tf.layers.conv3d(inputs, filters, padding="same")
    return(feature)


def fc(inputs, num_output_units):
    output = tf.contrib.layers.legacy_fully_connected(inputs, num_output_units)
    return(output)


def batchNorm3d(inputs):
    output = tf.layers.batch_normalization(inputs)
    return(output)


def activation(inputs):
    return(tf.nn.relu(inputs))


def max_pool():
    pass
