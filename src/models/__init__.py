import tensorflow as tf
from .blocks import ReflectionPad2d, ResidualBlock


def Generator(n_res_blocks=9):
    model = tf.keras.Sequential()

    # Encoding
    model.add(
        ReflectionPad2d(
            3, input_shape=(256, 256, 3)
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            64, (7, 7), strides=(1, 1),
            padding='valid', use_bias=False
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(
        tf.keras.layers.Conv2D(
            128, (3, 3), strides=(2, 2),
            padding='same', use_bias=False
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(
        tf.keras.layers.Conv2D(
            256, (3, 3), strides=(2, 2),
            padding='same', use_bias=False
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Transformation
    for i in range(n_res_blocks):
        model.add(ResidualBlock(256))

    # Decoding
    model.add(
        tf.keras.layers.Conv2DTranspose(
            128, (3, 3), strides=(2, 2),
            padding='same', use_bias=False
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            64, (3, 3), strides=(2, 2),
            padding='same', use_bias=False
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(ReflectionPad2d(3))
    model.add(
        tf.keras.layers.Conv2D(
            3, (7, 7), strides=(1, 1),
            padding='valid', activation='tanh'
        )
    )

    return model
