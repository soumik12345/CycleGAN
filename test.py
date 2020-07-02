import tensorflow as tf
from src.dataset import DataLoader
from matplotlib import pyplot as plt
from src.dataset.utils import count_batches, denormalize


def execute_dataset_test():

    configurations = {
        'dataset_name': 'horse2zebra',
        'resize_size': 286,
        'crop_size': 256,
        'shuffle_size': 1000,
        'batch_size': 1
    }

    dataloader = DataLoader(configurations)
    dataset = dataloader.make_datasets()
    print(dataset)

    print('Number of Batches:', count_batches(dataset))

    image_a, image_b = next(iter(dataset))
    print(image_a.shape, image_b.shape)

    plt.imshow(
        tf.cast(
            denormalize(image_a)[0],
            dtype=tf.uint8
        )
    )
    plt.title('Sample Image A')
    plt.show()

    plt.imshow(
        tf.cast(
            denormalize(image_b)[0],
            dtype=tf.uint8
        )
    )
    plt.title('Sample Image B')
    plt.show()


if __name__ == '__main__':
    execute_dataset_test()
