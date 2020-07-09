import os
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
