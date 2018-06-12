import tensorflow as tf
import numpy as np
from functools import reduce
import operator

# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def gen_random_noise(batch_size, shape):
    z = np.random.normal(0.0, 1.0, size=(batch_size,)+shape)
    return z

def gen_random_label(batch_size, class_size):
    onehot = np.eye(class_size)
    y = onehot[np.random.randint(low=0, high=class_size,
                                 size=(batch_size,1)).astype(np.int32)].squeeze()
    return y

def make_tile_image(generator, image_shape, input_shape, shape_per_class):
    '''
    Make a tile of images (for demo)
    
    Args:
    'generator'              generator tensor
    'image_shape'            shape of a single image
    'input_shape'            shpae of generator input (z shape, y shape)
    'shape_per_class'        tile shape of each class (2D). concatenated horizontally.
    '''

    class_size = input_shape[1][0]
    size_per_same_class = reduce(operator.mul, shape_per_class)

    batch_output = generator # (batch_size, ) + generator_output_shape
    '''split batch output into size_per_same_class chunks'''
    batch_per_class = tf.split(batch_output, class_size)
    def make_batch_to_tile(batch):
        b = tf.reshape(batch, (size_per_same_class,)+image_shape)
        col=[]
        for row in [b[shape_per_class[1]*i:shape_per_class[1]*(i+1)] for i in range(shape_per_class[0])]:
            row_concat = tf.concat(tf.split(row, shape_per_class[1]), 2)
            col.append(row_concat)
        grid = tf.concat(col, 1)
        return grid
    tile_per_class = list(map(make_batch_to_tile, batch_per_class))
    return tf.concat(tile_per_class, axis=2)

