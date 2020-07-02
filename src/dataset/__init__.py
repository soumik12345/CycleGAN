import tensorflow as tf
from .tfrecord import create_tfrecords


class DataLoader:

    def __init__(self, configs):
        self.configs = configs
        self.train_a_output, self.train_b_output, \
            self.test_a_output, self.test_b_output \
            = create_tfrecords(self.configs['dataset_name'])
        self.image_feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
        }

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize(
            image, [
                self.configs['resize_size'],
                self.configs['resize_size']
            ]
        )
        image = tf.image.random_crop(
            image, [
                self.configs['crop_size'],
                self.configs['crop_size'],
                tf.shape(image)[-1]
            ]
        )
        image = image / 127.5 - 1
        return image

    def parse_image_function(self, example_proto):
        features = tf.io.parse_single_example(
            example_proto, self.image_feature_description
        )
        encoded_image = features['image/encoded']
        image = self.preprocess_image(encoded_image)
        return image

    def get_parsed_dataset(self, tfrecord_file_path):
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file_path)
        parsed_dataset = raw_dataset.map(self.parse_image_function)
        return parsed_dataset

    def make_datasets(self):
        train_a_dataset = self.get_parsed_dataset(self.train_a_output)
        train_b_dataset = self.get_parsed_dataset(self.train_b_output)
        dataset = tf.data.Dataset.zip(
            (train_a_dataset, train_b_dataset)
        ).shuffle(
            self.configs['shuffle_size']
        ).batch(
            self.configs['batch_size']
        )
        return dataset
