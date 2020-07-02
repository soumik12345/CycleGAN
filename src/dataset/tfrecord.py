import os
from PIL import Image
from glob import glob
from tqdm import tqdm
import tensorflow as tf


def make_example(file_path):
    try:
        with open(file_path, 'rb') as file:
            image_string = file.read()
        with Image.open(file_path) as image:
            image.load()
            assert (image.format == 'JPEG')
            filename = os.path.basename(file_path)
            return tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/encoded': tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[image_string]
                            )
                        ),
                        'image/format': tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=['JPEG'.encode()]
                            )
                        ),
                        'image/width': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[image.width]
                            )
                        ),
                        'image/height': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[image.height]
                            )
                        ),
                        'image/filename': tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[filename.encode()]
                            )
                        )
                    }
                )
            )
    except Exception as e:
        print(e)
        return None


def write_to_tfrecord(image_list, file_name):
    with tf.io.TFRecordWriter(file_name) as writer:
        for file in tqdm(image_list):
            example = make_example(file)
            writer.write(example.SerializeToString())


def create_tfrecords(dataset_name):
    train_a_files = sorted(glob(os.path.join('data', dataset_name, 'trainA/*')))
    train_b_files = sorted(glob(os.path.join('data', dataset_name, 'trainB/*')))
    test_a_files = sorted(glob(os.path.join('data', dataset_name, 'testA/*')))
    test_b_files = sorted(glob(os.path.join('data', dataset_name, 'testB/*')))

    try:
        os.mkdir('./tfrecords')
    except Exception as e:
        print('Directory Already Exists')

    try:
        os.mkdir(os.path.join('./tfrecords/', dataset_name))
    except Exception as e:
        print(str(os.path.join('./tfrecords/', dataset_name)), 'Directory Already Exists')

    train_a_output = os.path.join('./tfrecords', dataset_name, 'trainA.tfrecord')
    train_b_output = os.path.join('./tfrecords', dataset_name, 'trainB.tfrecord')
    test_a_output = os.path.join('./tfrecords', dataset_name, 'testA.tfrecord')
    test_b_output = os.path.join('./tfrecords', dataset_name, 'testB.tfrecord')

    print('Generating TFRecord for TrainA Images...')
    write_to_tfrecord(train_a_files, train_a_output)

    print('Generating TFRecord for TrainB Images...')
    write_to_tfrecord(train_b_files, train_b_output)

    print('Generating TFRecord for TestA Images...')
    write_to_tfrecord(test_a_files, test_a_output)

    print('Generating TFRecord for TestB Images...')
    write_to_tfrecord(test_b_files, test_b_output)
