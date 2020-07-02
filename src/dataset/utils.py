def count_batches(dataset):
    size = 0
    for _ in dataset:
        size += 1
    return size


def denormalize(image):
    return (image + 1) * 127.5
