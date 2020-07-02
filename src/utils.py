import tensorflow as tf
from random import uniform, randint


class ImagePool:

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.count = 0
        self.pool = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if self.count < self.pool_size:
                self.count = self.count + 1
                self.pool.append(image)
                return_images.append(image)
            else:
                p = uniform(0, 1)
                if p > 0.5:
                    random_id = randint(0, self.pool_size - 1)
                    tmp = self.pool[random_id]
                    self.pool[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return tf.stack(return_images, axis=0)
