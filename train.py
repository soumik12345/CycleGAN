import os
import wandb
from tqdm import tqdm
import tensorflow as tf
from src.utils import ImagePool
from src.dataset import DataLoader
from src.scheduler import LinearDecay
from src.dataset.utils import count_batches
from src.models import Generator, Discriminator


class Trainer:

    def __init__(self, configs):
        self.configs = configs

        wandb.init(
            project=self.configs['project_name'],
            name=self.configs['experiment_name'],
            sync_tensorboard=True
        )

        self.fake_pool_b2a = ImagePool(self.configs['pool_size'])
        self.fake_pool_a2b = ImagePool(self.configs['pool_size'])

        self.loss_gen_total_metrics = tf.keras.metrics.Mean(
            'loss_gen_total_metrics', dtype=tf.float32
        )
        self.loss_dis_total_metrics = tf.keras.metrics.Mean(
            'loss_dis_total_metrics', dtype=tf.float32
        )
        self.loss_cycle_a2b2a_metrics = tf.keras.metrics.Mean(
            'loss_cycle_a2b2a_metrics', dtype=tf.float32
        )
        self.loss_cycle_b2a2b_metrics = tf.keras.metrics.Mean(
            'loss_cycle_b2a2b_metrics', dtype=tf.float32
        )
        self.loss_gen_a2b_metrics = tf.keras.metrics.Mean(
            'loss_gen_a2b_metrics', dtype=tf.float32
        )
        self.loss_gen_b2a_metrics = tf.keras.metrics.Mean(
            'loss_gen_b2a_metrics', dtype=tf.float32
        )
        self.loss_dis_b_metrics = tf.keras.metrics.Mean(
            'loss_dis_b_metrics', dtype=tf.float32
        )
        self.loss_dis_a_metrics = tf.keras.metrics.Mean(
            'loss_dis_a_metrics', dtype=tf.float32
        )
        self.loss_id_b2a_metrics = tf.keras.metrics.Mean(
            'loss_id_b2a_metrics', dtype=tf.float32
        )
        self.loss_id_a2b_metrics = tf.keras.metrics.Mean(
            'loss_id_a2b_metrics', dtype=tf.float32
        )

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
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            self.discriminator_lr_scheduler,
            self.configs['adam_beta_1']
        )

        self.checkpoint, self.checkpoint_manager = self.make_checkpoints()

    def calculate_gan_loss(self, prediction, is_real):
        if is_real:
            return self.mse_loss(prediction, tf.ones_like(prediction))
        else:
            return self.mse_loss(prediction, tf.zeros_like(prediction))

    def calculate_cycle_loss(self, reconstructed_images, real_images):
        return self.mae_loss(reconstructed_images, real_images)

    def calculate_identity_loss(self, identity_images, real_images):
        return self.mae_loss(identity_images, real_images)

    def get_dataset(self):
        data_loader = DataLoader(self.configs['dataset_configs'])
        dataset = data_loader.make_datasets()
        return dataset

    def reset_metrics(self):
        self.loss_gen_a2b_metrics.reset_states()
        self.loss_gen_b2a_metrics.reset_states()
        self.loss_dis_b_metrics.reset_states()
        self.loss_dis_a_metrics.reset_states()
        self.loss_id_a2b_metrics.reset_states()
        self.loss_id_b2a_metrics.reset_states()

    def make_checkpoints(self):
        checkpoint_dir = os.path.join(
            wandb.run.dir,
            './checkpoints-{}'.format(
                self.configs['dataset_configs']['dataset_name']
            )
        )
        checkpoint = tf.train.Checkpoint(
            generator_a2b=self.generator_a2b,
            generator_b2a=self.generator_b2a,
            discriminator_b=self.discriminator_b,
            discriminator_a=self.discriminator_a,
            optimizer_gen=self.generator_optimizer,
            optimizer_dis=self.discriminator_optimizer,
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

    @tf.function
    def train_generator(self, images_a, images_b):
        real_a = images_a
        real_b = images_b

        with tf.GradientTape() as tape:
            # Cycle A -> B -> A
            fake_a2b = self.generator_a2b(real_a, training=True)
            recon_b2a = self.generator_b2a(fake_a2b, training=True)
            # Cycle B -> A -> B
            fake_b2a = self.generator_b2a(real_b, training=True)
            recon_a2b = self.generator_a2b(fake_b2a, training=True)

            # Use real B to generate B should be identical
            identity_a2b = self.generator_a2b(real_b, training=True)
            identity_b2a = self.generator_b2a(real_a, training=True)
            loss_identity_a2b = self.calculate_identity_loss(identity_a2b, real_b)
            loss_identity_b2a = self.calculate_identity_loss(identity_b2a, real_a)

            # Generator A2B tries to trick Discriminator B that the generated image is B
            loss_gan_gen_a2b = self.calculate_gan_loss(
                self.discriminator_b(fake_a2b, training=True), True
            )
            # Generator B2A tries to trick Discriminator A that the generated image is A
            loss_gan_gen_b2a = self.calculate_gan_loss(
                self.discriminator_a(fake_b2a, training=True), True
            )
            loss_cycle_a2b2a = self.calculate_cycle_loss(recon_b2a, real_a)
            loss_cycle_b2a2b = self.calculate_cycle_loss(recon_a2b, real_b)

            # Total generator loss
            loss_gen_total = loss_gan_gen_a2b + loss_gan_gen_b2a \
                             + (loss_cycle_a2b2a + loss_cycle_b2a2b) * self.configs['lambda_cycle'] \
                             + (loss_identity_a2b + loss_identity_b2a) * self.configs['lambda_id']

        trainable_variables = self.generator_a2b.trainable_variables + self.generator_b2a.trainable_variables
        gradient_gen = tape.gradient(loss_gen_total, trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradient_gen, trainable_variables))

        # Metrics
        self.loss_gen_a2b_metrics(loss_gan_gen_a2b)
        self.loss_gen_b2a_metrics(loss_gan_gen_b2a)
        self.loss_id_b2a_metrics(loss_identity_b2a)
        self.loss_id_a2b_metrics(loss_identity_a2b)
        self.loss_cycle_a2b2a_metrics(loss_cycle_a2b2a)
        self.loss_cycle_b2a2b_metrics(loss_cycle_b2a2b)
        self.loss_gen_total_metrics(loss_gen_total)

        loss_dict = {
            'loss_gen_a2b': loss_gan_gen_a2b,
            'loss_gen_b2a': loss_gan_gen_b2a,
            'loss_id_a2b': loss_identity_a2b,
            'loss_id_b2a': loss_identity_b2a,
            'loss_cycle_a2b2a': loss_cycle_a2b2a,
            'loss_cycle_b2a2b': loss_cycle_b2a2b,
            'loss_gen_total': loss_gen_total,
        }
        return fake_a2b, fake_b2a, loss_dict

    @tf.function
    def train_discriminator(self, images_a, images_b, fake_a2b, fake_b2a):
        real_a = images_a
        real_b = images_b

        with tf.GradientTape() as tape:
            # Discriminator A should classify real_a as A
            loss_gan_dis_a_real = self.calculate_gan_loss(
                self.discriminator_a(real_a, training=True), True
            )
            # Discriminator A should classify generated fake_b2a as not A
            loss_gan_dis_a_fake = self.calculate_gan_loss(
                self.discriminator_a(fake_b2a, training=True), False
            )
            # Discriminator B should classify real_b as B
            loss_gan_dis_b_real = self.calculate_gan_loss(
                self.discriminator_b(real_b, training=True), True)
            # Discriminator B should classify generated fake_a2b as not B
            loss_gan_dis_b_fake = self.calculate_gan_loss(
                self.discriminator_b(fake_a2b, training=True), False
            )

            # Total discriminator loss
            loss_dis_a = (loss_gan_dis_a_real + loss_gan_dis_a_fake) * 0.5
            loss_dis_b = (loss_gan_dis_b_real + loss_gan_dis_b_fake) * 0.5
            loss_dis_total = loss_dis_a + loss_dis_b

        trainable_variables = self.discriminator_a.trainable_variables + self.discriminator_b.trainable_variables
        gradient_dis = tape.gradient(loss_dis_total, trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradient_dis, trainable_variables))

        # Metrics
        self.loss_dis_a_metrics(loss_dis_a)
        self.loss_dis_b_metrics(loss_dis_b)
        self.loss_dis_total_metrics(loss_dis_total)

        loss_dict = {
            'loss_dis_b': loss_dis_b,
            'loss_dis_a': loss_dis_a,
            'loss_dis_total': loss_dis_total
        }

        return loss_dict

    def train_step(self, images_a, images_b):
        fake_a2b, fake_b2a, gen_loss_dict = self.train_generator(images_a, images_b)
        fake_b2a_from_pool = self.fake_pool_b2a.query(fake_b2a)
        fake_a2b_from_pool = self.fake_pool_a2b.query(fake_a2b)
        dis_loss_dict = self.train_discriminator(
            images_a, images_b,
            fake_a2b_from_pool, fake_b2a_from_pool
        )
        return gen_loss_dict, dis_loss_dict

    def log_metrics(self):
        wandb.log({
            'loss_gen_a2b': self.loss_gen_a2b_metrics.result(),
            'loss_gen_b2a': self.loss_gen_b2a_metrics.result(),
            'loss_dis_b': self.loss_dis_b_metrics.result(),
            'loss_dis_a': self.loss_dis_a_metrics.result(),
            'loss_id_a2b': self.loss_id_a2b_metrics.result(),
            'loss_id_b2a': self.loss_id_b2a_metrics.result(),
            'loss_gen_total': self.loss_gen_total_metrics.result(),
            'loss_dis_total': self.loss_dis_total_metrics.result(),
            'loss_cycle_a2b2a': self.loss_cycle_a2b2a_metrics.result(),
            'loss_cycle_b2a2b': self.loss_cycle_b2a2b_metrics.result(),
            'gen_learning_rate': self.gen_lr_scheduler.current_learning_rate,
            'dis_learning_rate': self.dis_lr_scheduler.current_learning_rate
        })
        self.reset_metrics()

    def train(self):
        for epoch in range(self.checkpoint.epoch + 1, self.configs['epochs'] + 1):
            print('Epoch:', epoch)
            for step, batch in tqdm(enumerate(self.dataset)):
                self.train_step(batch[0], batch[1])
            self.log_metrics()
            self.checkpoint.epoch.assign_add(1)
            if epoch % 2 == 0:
                save_path = self.checkpoint_manager.save()
                print(
                    "Saved checkpoint for epoch {}: {}".format(
                        int(self.checkpoint.epoch), save_path
                    )
                )


if __name__ == '__main__':
    configurations = {
        'project_name': 'cyclegan',
        'experiment_name': 'horse2zebra',
        'pool_size': 50,
        'input_size': 256,
        'residual_blocks': 9,
        'lr': 2e-4,
        'epochs': 200,
        'decay_epochs': 100,
        'adam_beta_1': 0.5,
        'dataset_configs': {
            'dataset_name': 'horse2zebra',
            'resize_size': 286,
            'crop_size': 256,
            'shuffle_size': 1000,
            'batch_size': 1
        },
        'lambda_cycle': 10.0,
        'lambda_id': 5.0,
    }
    trainer = Trainer(configurations)
    trainer.train()
