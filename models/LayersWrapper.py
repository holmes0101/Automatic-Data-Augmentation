import tensorflow as tf 
import keras.layers as layers

def KerasConvlayer(x, n_filters, k_size, strides = (1, 1), batch_norm = True, name = "ConvLayer"):
    """
    Wraps the three steps needed to compute the Convlayer
    with BatchNorm and LearkyRelu activation
    """
    with tf.name_scope(name):
        pre_activations = layers.Conv2D(filters = n_filters, 
                                              kernel_size = k_size, 
                                              strides = strides, 
                                              padding = 'same', activation = "linear")(x)
        if batch_norm:
            pre_activations = layers.BatchNormalization(scale = False)(pre_activations)

        return layers.LeakyReLU()(pre_activations)

def KerasTranspConvlayer(x, n_filters, k_size, strides = (1, 1), batch_norm = True, name = "T_ConvLayer"):
    """
    Wraps the three steps needed to compute the TranspConvlayer
    with BatchNorm and LearkyRelu activation
    """
    with tf.name_scope(name):
        pre_activations = layers.Conv2DTranspose(filters = n_filters, 
                                                       kernel_size = k_size, 
                                                       strides = strides, 
                                                       padding = 'same', activation = "linear")(x)
        if batch_norm:
            pre_activations = layers.BatchNormalization(scale = False)(pre_activations)

        return layers.LeakyReLU()(pre_activations)

def KerasPooling(x, pool_size = (2, 2), strides = None, name = "PoolingLayer"):
    with tf.name_scope(name):
        return layers.MaxPooling2D(pool_size = pool_size, strides = strides, padding = "same")(x)