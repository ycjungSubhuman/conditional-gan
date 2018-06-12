import numpy as np
import tensorflow as tf
import network
import cgan
import sys
from data_interface import mnist

DATA_SET_SIZE = 65000
X_SHAPE = (784, 1)
Y_SHAPE = (28, 28, 10)

if __name__=='__main__':
    tf.reset_default_graph()

    # load MNIST
    train_validation_set = mnist.train('../../dataset/MNIST_Dataset/train/')
    test_set = mnist.test('../../dataset/MNIST_Dataset/test/')
    data_set = train_validation_set.concatenate(test_set)

    # run cycle for current parameter settings
    gan = cgan.CGAN(
        generator_fn=network.dcgan_generator,
        discriminator_fn=network.dcgan_discriminator
    )
    train_params = {
        'lr': 0.0002,
        'beta1': 0.5,
        'batch_size': 100,
        'max_epoch' : 90,
        }
    g_model_params = {
        'input_shape': ((200,), (10,)),
        'z_pre_shape': (1, 1, 200),
        'y_pre_shape': (1, 1, 10),
        'z_deconv_param': (256, (7, 7), 1, 'valid'), # (7, 7, -)
        'y_deconv_param': (256, (7, 7), 1, 'valid'), # (7, 7, -)
        'mid_deconv_params': [
            (128, (4, 4), 2, 'same'), # (14, 14, -)
            ],
        'final_deconv_param': (1, (4, 4), 2, 'same'), # (28, 28, -)
        'output_shape': (28, 28, 1),
        'output_image_shape': (28, 28, 1),
        'summ_shape_per_class': (10, 1),
    }
    d_model_params = {
        'input_shape': ((784,), (10,)),
        'x_pre_shape': (28, 28, 1),
        'y_pre_shape': (28, 28, 10),
        'x_conv_param': (64, (4, 4), 2, 'same'), # (14, 14, -)
        'y_conv_param': (64, (4, 4), 2, 'same'), # (14, 14, -)
        'mid_conv_params': [
            (256, (4, 4), 2, 'same'), # (7, 7, -)
            ],
        'final_conv_param': (1, (7, 7), 1, 'valid'),
        'output_shape': ((1,), (1,)),
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
