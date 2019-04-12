import tensorflow as tf

import numpy as np

# from keras.layers import Conv2D, Flatten, Reshape, Dense, Concatenate, Input
# from keras.layers import Lambda as keras_lambda
# from keras.layers.wrappers import TimeDistributed
# from keras import regularizers as rgl
from softlearning.utils.keras import PicklableKerasModel


def feedforward_model(input_shapes,
                      output_size,
                      hidden_layer_sizes,
                      activation='relu',
                      output_activation='linear',
                      preprocessors=None,
                      name='feedforward_model',
                      *args,
                      **kwargs):
    inputs = [
        tf.keras.layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]

    if preprocessors is None:
        preprocessors = (None, ) * len(inputs)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_ in zip(preprocessors, inputs)
    ]

    concatenated = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(preprocessed_inputs)

    out = concatenated
    for units in hidden_layer_sizes:
        out = tf.keras.layers.Dense(
            units, *args, activation=activation, **kwargs
        )(out)

    out = tf.keras.layers.Dense(
        output_size, *args, activation=output_activation, **kwargs
    )(out)

    model = PicklableKerasModel(inputs, out, name=name)

    return model


def cnn_model(input_shapes,
              output_size,
              activation='relu',
              output_activation='linear',
              preprocessors=None,
              name='cnn_model',
              img_dim=(256,256),
              *args,
              **kwargs):
    l2_reg = 0
    ob = Input(shape=input_shapes[0])


    # if preprocessors is None:
    #     preprocessors = (None,) * len(inputs)
    #
    # preprocessed_inputs = [
    #     preprocessor(input_) if preprocessor is not None else input_
    #     for preprocessor, input_ in zip(preprocessors, inputs)
    # ]
    #
    # ob = tf.keras.layers.Lambda(
    #     lambda x: tf.concat(x, axis=-1)
    # )(preprocessed_inputs)
    #ob_image = ob[:, :(4 * np.product(img_dim))]
    ob_image = keras_lambda(lambda x: x[:, :(4 * np.product(img_dim))])(ob)
    #ob_image = tf.reshape(ob_image, [tf.shape(ob_image)[0], 4] + list(img_dim) + [1])
    ob_image = Reshape(tuple([4] + list(img_dim) + [1]))(ob_image)
    #ob_scalar = ob[:, (4 * np.product(img_dim)):]
    ob_scalar = keras_lambda(lambda x: x[:, (4 * np.product(img_dim)):])(ob)
    x_image = ob_image
    x_image = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4),
                                     activation="relu",
                                     kernel_regularizer=rgl.l2(l2_reg)))(x_image)

    x_image = TimeDistributed(Conv2D(64, (5, 5), strides=(2, 2),
                                     activation="relu",
                                     kernel_regularizer=rgl.l2(l2_reg)))(x_image)

    x_image = TimeDistributed(Conv2D(128, (5, 5), strides=(1, 1),
                                     activation="relu",
                                     kernel_regularizer=rgl.l2(l2_reg)))(x_image)

    x_image = TimeDistributed(Conv2D(128, (5, 5), strides=(1, 1),
                                     activation="relu",
                                     kernel_regularizer=rgl.l2(l2_reg)))(x_image)

    x_image = TimeDistributed(Conv2D(128, (3, 3), strides=(1, 1),
                                     activation="relu",
                                     kernel_regularizer=rgl.l2(l2_reg)))(x_image)

    #x_image = TimeDistributed(keras_lambda(tf.contrib.layers.spatial_softmax))(x_image)
    x_image = Reshape((4*20*20*128,))(x_image)
    x_scalar = ob_scalar
    x_image = Dense(256, activation='relu', name='lin1')(x_image)
    x = Concatenate()([x_image, x_scalar])
    # x = x_image
    x = Dense(128, activation='relu', name='lin2')(x)
    x = Dense(128, activation='relu', name='lin3')(x)
    out = Dense(output_size, name='final')(x)
    model = PicklableKerasModel(ob, out, name=name)

    return model

