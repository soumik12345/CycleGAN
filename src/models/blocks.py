import tensorflow as tf


class ReflectionPad2d(tf.keras.layers.Layer):

    def __init__(self, padding, **kwargs):
        super(ReflectionPad2d, self).__init__(**kwargs)

        self.padding = [
            [0, 0], [padding, padding],
            [padding, padding], [0, 0]
        ]

    def call(self, inputs, **kwargs):
        return tf.pad(inputs, self.padding, 'REFLECT')


class ResidualBlock(tf.keras.Model):

    def __init__(self, dim):
        super(ResidualBlock, self).__init__()

        self.padding1 = ReflectionPad2d(1)
        self.conv1 = tf.keras.layers.Conv2D(dim, (3, 3), padding='valid', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.padding2 = ReflectionPad2d(1)
        self.conv2 = tf.keras.layers.Conv2D(dim, (3, 3), padding='valid', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.padding1(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.padding2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        outputs = inputs + x
        return outputs
