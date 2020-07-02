import tensorflow as tf
from src.dataset import DataLoader
from src.scheduler import LinearDecay
from src.dataset.utils import count_batches
from src.models import Generator, Discriminator


class Trainer:

    def __init__(self, configs):
        self.configs = configs

        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.mae_loss = tf.keras.losses.MeanAbsoluteError()

        self.dataset = self.get_dataset()

        self.generator_a2b = Generator(
            input_size=self.configs['input_size'],
            n_res_blocks=self.configs['residual_blocks']
        )
        self.generator_b2a = Generator(
            input_size=self.configs['input_size'],
            n_res_blocks=self.configs['residual_blocks']
        )
        self.discriminator_a = Discriminator(
            input_size=self.configs['input_size']
        )
        self.discriminator_b = Discriminator(
            input_size=self.configs['input_size']
        )

        total_batches = count_batches(self.dataset)
        self.generator_lr_scheduler = LinearDecay(
            initial_learning_rate=self.configs['lr'],
            total_steps=self.configs['epochs'] * total_batches,
            step_decay=self.configs['decay_epochs'] * total_batches
        )
        self.discriminator_lr_scheduler = LinearDecay(
            initial_learning_rate=self.configs['lr'],
            total_steps=self.configs['epochs'] * total_batches,
            step_decay=self.configs['decay_epochs'] * total_batches
        )

        self.generator_optimizer = tf.keras.optimizers.Adam(
            self.generator_lr_scheduler,
            self.configs['adam_beta_1']
        )
        self.dicriminator_optimizer = tf.keras.optimizers.Adam(
            self.discriminator_lr_scheduler,
            self.configs['adam_beta_1']
        )

        self.checkpoint, self.checkpoint_manager = self.make_checkpoints()

    def calculate_gan_loss(self, prediction, is_real):
        if is_real:
            return self.mse_loss(prediction, tf.ones_like(prediction))
        else:
            return self.mse_loss(prediction, tf.zeros_like(prediction))

    def calc_cycle_loss(self, reconstructed_images, real_images):
        return self.mae_loss(reconstructed_images, real_images)

    def calc_identity_loss(self, identity_images, real_images):
        return self.mae_loss(identity_images, real_images)

    def get_dataset(self):
        data_loader = DataLoader(self.configs['dataset_configs'])
        dataset = data_loader.make_datasets()
        return dataset

    def make_checkpoints(self):
        checkpoint_dir = './checkpoints-{}'.format(
            self.configs['dataset_configs']['dataset_name']
        )
        checkpoint = tf.train.Checkpoint(
            generator_a2b=self.generator_a2b,
            generator_b2a=self.generator_b2a,
            discriminator_b=self.discriminator_b,
            discriminator_a=self.discriminator_a,
            optimizer_gen=self.generator_optimizer,
            optimizer_dis=self.dicriminator_optimizer,
            epoch=tf.Variable(0)
        )
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_dir, max_to_keep=None
        )
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        if checkpoint_manager.latest_checkpoint:
            print("Restored from {}".format(
                checkpoint_manager.latest_checkpoint)
            )
        else:
            print("Initializing from scratch.")
        return checkpoint, checkpoint_manager
