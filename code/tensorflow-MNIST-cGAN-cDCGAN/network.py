import tensorflow as tf
from util import lrelu

def cgan_generator(features, model_params, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        z = tf.reshape(features['z'], (-1,)+model_params['input_shape'][0])
        y = tf.reshape(features['y'], (-1,)+model_params['input_shape'][1])

        z_representation = tf.layers.dense(z, model_params['z_dense_width'], kernel_initializer=w_init)
        y_representation = tf.layers.dense(y, model_params['y_dense_width'], kernel_initializer=w_init)

        cat1 = tf.concat([z_representation, y_representation], 1)

        prev = cat1
        for width in model_params['mid_dense_widths']:
            dense = tf.layers.dense(prev, width, kernel_initializer=w_init)
            dense_batch_normalized = tf.layers.batch_normalization(dense)
            relu = tf.nn.relu(dense_batch_normalized)
            prev = relu

        image = tf.layers.dense(prev, model_params['output_shape'][0], kernel_initializer=w_init)
        o = tf.nn.tanh(image)

        return o
def cgan_discriminator(features, model_params, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        x = tf.reshape(features['x'], (-1,)+model_params['input_shape'][0])
        y = tf.reshape(features['y'], (-1,)+model_params['input_shape'][1])

        x_representation = tf.layers.dense(x, model_params['x_dense_width'], kernel_initializer=w_init)
        y_representation = tf.layers.dense(y, model_params['y_dense_width'], kernel_initializer=w_init)

        cat1 = tf.concat([x_representation, y_representation], 1)

        prev = cat1
        for width in model_params['mid_dense_widths']:
            dense = tf.layers.dense(prev, width, kernel_initializer=w_init)
            dense_batch_normalized = tf.layers.batch_normalization(dense)
            leaky_relu = lrelu(dense_batch_normalized)
            prev = leaky_relu
            
        logit = tf.layers.dense(prev, model_params['output_shape'][0][0], kernel_initializer=w_init)
        o = tf.nn.sigmoid(logit)

        return o, logit

# G(z)
def dcgan_generator(features, model_params, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        z_4d = tf.reshape(features['z'], (-1,)+model_params['z_pre_shape'])
        y_4d = tf.reshape(features['y'], (-1,)+model_params['y_pre_shape'])

        def deconv_fn(prev, param):
            return tf.layers.conv2d_transpose(prev, param[0], param[1], strides=param[2], padding=param[3],
                                              kernel_initializer=w_init, bias_initializer=b_init)

        z_deconv = tf.nn.relu(tf.layers.batch_normalization(deconv_fn(z_4d, model_params['z_deconv_param'])))
        y_deconv = tf.nn.relu(tf.layers.batch_normalization(deconv_fn(y_4d, model_params['y_deconv_param'])))
        
        cat1 = tf.concat([z_deconv, y_deconv], 3)

        prev = cat1
        for param in model_params['mid_deconv_params']:
            deconv = deconv_fn(prev, param)
            deconv_batch_normalized = tf.layers.batch_normalization(deconv)
            relu = tf.nn.relu(deconv_batch_normalized)
            prev = relu

        # output layer
        image = deconv_fn(prev, model_params['final_deconv_param'])
        assert_op = tf.assert_equal(tf.shape(image)[1:], model_params['output_shape'])
        with tf.control_dependencies([assert_op]):
            o = tf.nn.tanh(image)

        return o

# D(x)
def dcgan_discriminator(features, model_params, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        x = tf.reshape(features['x'], (-1,)+model_params['x_pre_shape'])
        y_width, y_height, _ = model_params['y_pre_shape']
        y = tf.reshape(tf.tile(features['y'], [1, y_width*y_height]), (-1,)+model_params['y_pre_shape'])
        summ_y = tf.summary.tensor_summary('y_summ', y)

        def conv_fn(data_in, param):
            return tf.layers.conv2d(data_in, param[0], param[1], strides=param[2], padding=param[3],
                                    kernel_initializer=w_init, bias_initializer=b_init)

        x_conv = lrelu(conv_fn(x, model_params['x_conv_param']))
        y_conv = lrelu(conv_fn(y, model_params['y_conv_param']))

        # concat layer
        cat1 = tf.concat([x_conv, y_conv], 3)

        prev = cat1
        for param in model_params['mid_conv_params']:
            conv = conv_fn(prev, param)
            conv_batch_normalized = tf.layers.batch_normalization(conv)
            relu = lrelu(conv_batch_normalized)
            prev = relu
            
        # output layer
        conv = conv_fn(prev, model_params['final_conv_param'])

        assert_op = tf.assert_equal(tf.shape(conv)[1:], model_params['output_shape'])
        with tf.control_dependencies([assert_op, summ_y]):
            o = tf.nn.sigmoid(conv)

        return o, conv
