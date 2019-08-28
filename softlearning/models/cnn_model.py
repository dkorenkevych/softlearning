import tensorflow as tf

import numpy as np


from tensorflow.keras.layers import *

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


def cnn_core(input_shapes,
              preprocessors=None,
              name='cnn_model',
              img_dim=(256,256),
              ob_rms=None,
              *args,
              **kwargs):

    l2_reg = 0
    inputs = [Input(shape=input_shape) for input_shape in input_shapes]
    if preprocessors is None:
        preprocessors = (None, ) * len(inputs)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_ in zip(preprocessors, inputs)
    ]

    preprocessed_inputs[0] = Lambda(lambda x: tf.clip_by_value((x - ob_rms.mean) / ob_rms.std, -5.0, 5.0))(preprocessed_inputs[0])

    if len(preprocessed_inputs) > 1:
        concatenated = Concatenate(axis=-1)(preprocessed_inputs)
    else:
        concatenated = preprocessed_inputs[0]


    ob = concatenated


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



    ob_image = Lambda(lambda x: x[:, :(4 * np.product(img_dim))])(ob)
    #ob_image = tf.reshape(ob_image, [tf.shape(ob_image)[0], 4] + list(img_dim) + [1])
    ob_image = Reshape(tuple([4] + list(img_dim) + [1]))(ob_image)
    #ob_scalar = ob[:, (4 * np.product(img_dim)):]
    ob_scalar = Lambda(lambda x: x[:, (4 * np.product(img_dim)):])(ob)



    x_image = Lambda(lambda x: x - tf.reduce_mean(x, axis=[-3, -2, -1], keep_dims=True))(ob_image)
    cnn_layers = [
                     TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4),
                               activation="relu",
                               padding='same'), name="shared1"),
                     TimeDistributed(Conv2D(64, (5, 5), strides=(2, 2),
                                            activation="relu",
                                            padding='same'), name="shared2"),
                     TimeDistributed(Conv2D(64, (5, 5), strides=(2, 2),
                                            activation="relu",
                                            padding='same'), name="shared3")

                 ]
    count = 4
    for i in range(3):
        cnn_layers.append(TimeDistributed(Conv2D(64, (5, 5), strides=(1, 1),
                               activation="relu",
                               padding='same'), name="shared{}".format(count)))
        count += 1
    for i in range(5):
        TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1),
                               activation="relu",
                               padding='same'), name="shared{}".format(count))
        count += 1

    for layer in cnn_layers:
        x_image = layer(x_image)


    #x_image = TimeDistributed(keras_lambda(tf.contrib.layers.spatial_softmax))(x_image)
    #pool = TimeDistributed(GlobalAveragePooling2D())(x_image)
    #x_image = TimeDistributed(Lambda(tf.contrib.layers.spatial_softmax))(x_image)
    #x_image = Concatenate(axis=-1)([x_image, pool])
    x_image = TimeDistributed(Flatten())(x_image)
    x_image = TimeDistributed(Dense(512, activation='relu'), name="shared{}".format(count))(x_image)
    x_image = Flatten()(x_image)
    #x_image = Reshape((4*20*20*128,))(x_image)
    x_scalar = ob_scalar
    #x_image = Dense(256, activation='relu', name='lin1')(x_image)
    x = Concatenate()([x_image, x_scalar])
    # x = x_image


    # x = Dense(256, activation='relu', name='lin2')(x)
    # x = Dense(256, activation='relu', name='lin3')(x)
    # out = Dense(output_size, name='final')(x)
    # model = PicklableKerasModel(inputs, out, name=name)

    return inputs, [x]


