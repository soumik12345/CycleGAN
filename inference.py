import os
from PIL import Image
import tensorflow as tf
from src.models import Generator


class Inferer:

    def __init__(self, configs):
        self.configs = configs
        self.generator_a2b = Generator(
            input_size=self.configs['input_size'],
            n_res_blocks=self.configs['residual_blocks']
        )
        self.generator_b2a = Generator(
            input_size=self.configs['input_size'],
            n_res_blocks=self.configs['residual_blocks']
        )

    def restore_checkpoint(self):
        checkpoint_dir = os.path.join(
            self.configs['checkpoint_dir'],
            './checkpoints-{}'.format(
                self.configs['dataset_configs']['dataset_name']
            )
        )
        checkpoint = tf.train.Checkpoint(
            generator_a2b=self.generator_a2b,
            generator_b2a=self.generator_b2a
        )
        manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_dir, max_to_keep=3
        )
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    @staticmethod
    def read_file(file_name):
        with open(file_name, 'rb') as f:
            content = f.read()
        original = tf.image.decode_jpeg(content)
        resized_original = tf.image.resize(original, (256, 256))
        float_original = tf.cast(resized_original, tf.float32)
        inputs = float_original / 127.5 - 1
        inputs = tf.expand_dims(inputs, 0)
        return original, inputs

    def infer_a2b(self, file_name):
        original, inputs = Inferer.read_file(file_name)
        outputs = self.generator_a2b(inputs)
        generated = outputs[0]
        generated = (generated + 1) * 127.5
        generated = tf.cast(generated, tf.uint8)
        original = Image.fromarray(original.numpy())
        generated = Image.fromarray(generated.numpy())
        return original, generated

    def infer_b2a(self, file_name):
        original, inputs = Inferer.read_file(file_name)
        outputs = self.generator_b2a(inputs)
        generated = outputs[0]
        generated = (generated + 1) * 127.5
        generated = tf.cast(generated, tf.uint8)
        original = Image.fromarray(original.numpy())
        generated = Image.fromarray(generated.numpy())
        return original, generated
