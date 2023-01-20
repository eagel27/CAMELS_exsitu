import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras.regularizers import l2


def build_model(input_size):
    """
    Build a simple dense model.
    :param input_size: An integer corresponding to the size of the input
    :return: The compiled model
    """
    dense_init = tf.keras.initializers.RandomNormal(mean=0.0,
                                                    stddev=0.2,
                                                    seed=None)

    input_layer = Input(shape=(input_size, ))
    x = Dense(32, activation='relu', kernel_initializer=dense_init)(input_layer)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu', kernel_initializer=dense_init)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu', kernel_initializer=dense_init)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(tfp.layers.IndependentNormal.params_size(1), activation=None,
              kernel_initializer=dense_init)(x)
    y = tfp.layers.IndependentNormal(1, tfd.Normal.sample)(x)
    model = Model(inputs=input_layer, outputs=y)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss=lambda a, rv_x: -rv_x.log_prob(a),
                  metrics=['mse'])
    return model


def build_cnn_model(input_shape):
    """
    Build a simple CNN.
    :param input_shape: A tuple corresponding to the shape of the input
    :return: The compiled model
    """

    dense_init = tf.keras.initializers.TruncatedNormal()
    conv_init = tf.keras.initializers.GlorotUniform(seed=None)

    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), activation='relu',
               padding="same", kernel_initializer=conv_init)(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu',
               padding="same", kernel_initializer=conv_init)(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu',
               padding="same", kernel_initializer=conv_init)(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001),
              kernel_initializer=dense_init)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dense(10, activation='relu', kernel_regularizer=l2(0.001),
              kernel_initializer=dense_init)(x)
    # model.add(Dropout(0.2))

    x = Flatten()(x)
    x = Dense(tfp.layers.IndependentNormal.params_size(1), activation=None,
              kernel_initializer=dense_init)(x)
    y = tfp.layers.IndependentNormal(1, tfd.Normal.sample)(x)

    model = Model(inputs=input_layer, outputs=y)
    model.summary()

    # compile CNN
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-6, clipnorm=1.)
    model.compile(optimizer=opt,
                  loss=lambda a, rv_x: -rv_x.log_prob(a),
                  metrics=['mse'])

    return model
