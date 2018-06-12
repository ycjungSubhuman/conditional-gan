import os, time, itertools
import numpy as np
import tensorflow as tf
import cgan
import sys
import network
from data_interface import mnist

DATA_SET_SIZE = 65000

if __name__=='__main__':
    tf.reset_default_graph()
    # load MNIST
    train_validation_set = mnist.train('../../dataset/MNIST_Dataset/train/')
    test_set = mnist.test('../../dataset/MNIST_Dataset/test/')
    data_set = train_validation_set.concatenate(test_set)

    # run cycle for current parameter settings
    gan = cgan.CGAN(
        generator_fn=network.cgan_generator,
        discriminator_fn=network.cgan_discriminator)
    train_params = {
        'lr': 0.0002,
        'beta1': 0.5,
        'batch_size': 100,
        'max_epoch' : 100,
        }
    g_model_params = {
        'input_shape': ((200,), (10,)),
        'output_shape': (784,),
        'output_image_shape': (28, 28, 1),
        'summ_shape_per_class': (10, 1),
        'z_dense_width': 256,
        'y_dense_width': 256,
        'mid_dense_widths': [512, 1024],
    }
    d_model_params = {
        'input_shape': ((784,), (10,)),
        'output_shape': ((1,), (1,)),
        'x_dense_width': 1024,
        'y_dense_width': 1024,
        'mid_dense_widths': [512, 256],
    }

    cgan.fid_run(gan,
                 run_name=sys.argv[1],
                 train_params=train_params,
                 g_model_params=g_model_params,
                 d_model_params=d_model_params,
                 summ_root="../../summary",
                 chk_root="../../checkpoints",
                 stat_root="./mnist_tmp",
                 data_set=data_set,
                 data_set_size=DATA_SET_SIZE)

