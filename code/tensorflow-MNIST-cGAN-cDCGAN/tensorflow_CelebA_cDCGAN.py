import numpy as np
import tensorflow as tf
import network
import cgan
import sys
from data_interface import celeba

DATA_SET_SIZE = 202599
X_SHAPE = (64, 64, 3)
Y_SHAPE = (2,)
Z_SHAPE = (200,)

if __name__=='__main__':
    tf.reset_default_graph()

    # load CelebA
    data_set = (celeba.dataset('../../dataset/CelebA/'))

    # run cycle for current parameter settings
    gan = cgan.CGAN(
        generator_fn=network.dcgan_generator,
        discriminator_fn=network.dcgan_discriminator
    )
    train_params = {
        'lr': 0.0002,
        'beta1': 0.5,
        'batch_size': 128,
        'max_epoch' : 20,
        }
    g_model_params = {
        'input_shape': (Z_SHAPE, Y_SHAPE),
        'z_pre_shape': (1, 1,)+Z_SHAPE,
        'y_pre_shape': (1, 1,)+Y_SHAPE,
        'z_deconv_param': (512, (4, 4), 1, 'valid'), # (4, 4, -)
        'y_deconv_param': (512, (4, 4), 1, 'valid'), # (4, 4, -)
        'mid_deconv_params': [
            (512, (4, 4), 2, 'same'), # (8, 8, -)
            (256, (4, 4), 2, 'same'), # (16, 16, -)
            (128, (4, 4), 2, 'same'), # (32, 32, -)
            ],
        'final_deconv_param': (3, (4, 4), 2, 'same'), # (64, 64, -)
        'output_shape': X_SHAPE,
        'output_image_shape': X_SHAPE,
        'summ_shape_per_class': (10, 5),
    }
    d_model_params = {
        'input_shape': (X_SHAPE, Y_SHAPE),
        'x_pre_shape': X_SHAPE,
        'y_pre_shape': X_SHAPE[0:2]+Y_SHAPE,
        'x_conv_param': (128, (4, 4), 2, 'same'), # (32, 32, -)
        'y_conv_param': (128, (4, 4), 2, 'same'), # (32, 32, -)
        'mid_conv_params': [
            (256, (4, 4), 2, 'same'), # (16, 16, -)
            (512, (4, 4), 2, 'same'), # (8, 8, -)
            (1024, (4, 4), 2, 'same'), # (4, 4, -)
            ],
        'final_conv_param': (1, (4, 4), 1, 'valid'),
        'output_shape': ((1,), (1,)),
    }

    cgan.fid_run(gan,
                 run_name=sys.argv[1],
                 train_params=train_params,
                 g_model_params=g_model_params,
                 d_model_params=d_model_params,
                 summ_root="../../summary",
                 chk_root="../../checkpoints",
                 stat_root="./celebA_tmp",
                 data_set=data_set,
                 data_set_size=DATA_SET_SIZE)
