import tensorflow as tf
import numpy as np
import util
import os
from functools import reduce
import operator
import fid

class CGAN:
    '''
    A Wrapper for generator-discriminator pair for cGAN
    '''
    def __init__(self, generator_fn, discriminator_fn):
        self._generator_fn = generator_fn
        self._discriminator_fn = discriminator_fn
    
    def generator_fn(self, features, model_params, mode, reuse=False):
        return self._generator_fn(features, model_params, mode, reuse)

    def discriminator_fn(self, features, model_params, mode, reuse=False):
        return self._discriminator_fn(features, model_params, mode, reuse)

def _shape_for_batch(shape):
    return (None,) + shape

def _run_discriminator_train(sess, placeholders, loss, optim, x_data, y_label, batch_size, z_input_shape):
    z = util.gen_random_noise(batch_size, z_input_shape)
    d_loss, _ = sess.run([loss, optim], {
        placeholders['z']:z,
        placeholders['x']:x_data,
        placeholders['y']:y_label,
        placeholders['mode']:True,
    })
    return d_loss

def _run_generator_train(sess, placeholders, loss, optim, x_data, batch_size, z_input_shape, y_input_shape):
    z = util.gen_random_noise(batch_size, z_input_shape)
    y = util.gen_random_label(batch_size, y_input_shape[0])
    g_loss, _ = sess.run([loss, optim], {
        placeholders['z']:z,
        placeholders['x']:x_data,
        placeholders['y']:y,
        placeholders['mode']:True,
    })
    return g_loss

def _run_summary(sess, placeholders, writer, summ, epoch, x_data, y_label, batch_size, z_input_shape):
    z = util.gen_random_noise(batch_size, z_input_shape)
    summary = sess.run(summ, {
        placeholders['z']:z,
        placeholders['x']:x_data,
        placeholders['y']:y_label,
        placeholders['mode']:False
    })
    writer.add_summary(summary, global_step=epoch)

def _run_tile_summary(sess, placeholders, writer, summ, epoch, shape_per_class, z_input_shape, y_input_shape):
    size_per_same_class = reduce(operator.mul, shape_per_class)
    class_size = y_input_shape[0]
    z = util.gen_random_noise(class_size*size_per_same_class, z_input_shape)
    y = np.reshape(np.tile(np.eye(class_size), size_per_same_class), (-1, class_size))
    summary = sess.run(summ, {
        placeholders['z']:z,
        placeholders['y']:y,
        placeholders['mode']:False,
    })
    writer.add_summary(summary, global_step=epoch)

def _log_scalar(name, epoch, writer, value):
    summary = tf.Summary()
    summary.value.add(tag=name, simple_value=value)
    writer.add_summary(summary, global_step=epoch)
    writer.flush()

def _maybe_grayscale_to_rgb(images):
    if images.shape[-1]==3:
        return images
    else:
        return np.repeat(images, 3, axis=3)


def _run_fid_calculation(sess, inception_sess, placeholders, batch_size, iteration, generator, mu, sigma, epoch, image_shape, z_input_shape, y_input_shape):
    f = 0.0
    for _ in range(iteration):
        z = util.gen_random_noise(batch_size, z_input_shape)
        y = util.gen_random_label(batch_size, y_input_shape[0])

        images = sess.run(tf.reshape(generator, (-1,)+image_shape), {
            placeholders['z']:z,
            placeholders['y']:y,
            placeholders['mode']:False,
            })
        images = _maybe_grayscale_to_rgb(images)
        images = (images + 1.0) / 2.0 * 255.0

        mu_gen, sigma_gen = fid.calculate_activation_statistics(images, inception_sess)
        f += fid.calculate_frechet_distance(mu, sigma, mu_gen, sigma_gen)
    return f / iteration

def _get_statistics(stat_root, data, image_shape, inception_sess):
    os.makedirs(stat_root, exist_ok=True)
    mu_path = os.path.join(stat_root, 'ac_mu.npy')
    sigma_path = os.path.join(stat_root, 'ac_sigma.npy')
    if os.path.exists(mu_path) and os.path.exists(sigma_path):
        print('Using cached activation statistics')
        mu = np.load(mu_path)
        sigma = np.load(sigma_path)
    else:
        image = _maybe_grayscale_to_rgb(np.reshape(data, (-1,)+image_shape))
        image = (image + 1.0) / 2.0 * 255.0
        mu, sigma = fid.calculate_activation_statistics(image, inception_sess)
        np.save(mu_path, mu)
        np.save(sigma_path, sigma)
    return mu, sigma

def fid_run(gan, run_name, train_params, g_model_params, d_model_params, summ_root, chk_root, stat_root,
            data_set, data_set_size):
    '''
    cGAN Training cycle

    * Runs training
    * Calculate FID measure
    * Write summaries for tensorboard
    * Save/restore model

    Args:
    'gan'               An instance of CGAN class
    'run_name'          Name for a run
    'train_params'      A dictionary of parameters for training
        ['lr']              Learning rate (float)
        ['beta1']           Beta1 parameter of AdamOptimizer (float)
        ['batch_size']      Mini-batch batch size (int)
        ['max_epoch']       Epoch limit (int)
    'g_model_params'    A dictionary of parameters for generator
        ['input_shape']        input shape of a single data. Usually a pair of shape (z shape, y shape)
        ['output_shape']       output shape of a single data
        ['output_image_shape'] output shape when interpreted as image
        ['summ_shape_per_class'] summary tile shape per class
        other keys depend on your cgan model function
    'd_model_params'    A dictionary of parameters for discriminator
        ['input_shape']        input shape of a single data. Usually a pair of shape (x shape, y shape)
        ['output_shape']       output shape of a single data. Usually a pair of shape (score shape, logit shape)
        other keys depend on your cgan model function
    'summ_root'         Path for summary
    'chk_root'          Path for checkpoints. checkpoints are saved every epoch
    'stat_root'   Path for activation statistics save
    'data_set'         dataset for training (tf.data.Dataset)
    'data_set_size'    data_set size

    Return:
    None
    '''

    '''Create Features'''
    placeholders = {}
    placeholders['z'] = tf.placeholder(tf.float32, shape=_shape_for_batch(g_model_params['input_shape'][0]), name='z')
    placeholders['y'] = tf.placeholder(tf.float32, shape=_shape_for_batch(g_model_params['input_shape'][1]), name='y')
    # y is shared between generator and discriminator
    assert(g_model_params['input_shape'][1] == d_model_params['input_shape'][1])
    placeholders['x'] = tf.placeholder(tf.float32, shape=_shape_for_batch(d_model_params['input_shape'][0]), name='x')
    placeholders['mode'] = tf.placeholder(dtype=tf.bool, name='mode')

    g_features = {'z':placeholders['z'], 'y':placeholders['y']}
    d_real_features = {'x':placeholders['x'], 'y':placeholders['y']}

    '''Create networks'''
    generator = gan.generator_fn(g_features, g_model_params, placeholders['mode'])
    real_discriminator, d_real_logit = gan.discriminator_fn(d_real_features, d_model_params, placeholders['mode'])
    d_fake_features = {'x':generator, 'y':placeholders['y']}
    fake_discriminator, d_fake_logit = gan.discriminator_fn(d_fake_features, d_model_params, placeholders['mode'], reuse=True)

    '''Define loss for optimization'''
    losses = {}
    d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_real_logit,
        labels=tf.ones(tf.shape(d_real_logit))
    ))
    d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logit,
        labels=tf.zeros(tf.shape(d_fake_logit))
    ))
    losses['d_loss'] = d_real_loss + d_fake_loss
    losses['g_loss'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logit,
        labels=tf.ones(tf.shape(d_fake_logit))
    ))

    '''Setup optimizer'''
    trainable_vars = tf.trainable_variables()
    d_vars = [var for var in trainable_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in trainable_vars if var.name.startswith('generator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_optim = (tf.train.AdamOptimizer(train_params['lr'], beta1=train_params['beta1'])
                   .minimize(losses['d_loss'], var_list=d_vars))
        g_optim = (tf.train.AdamOptimizer(train_params['lr'], beta1=train_params['beta1'])
                   .minimize(losses['g_loss'], var_list=g_vars))

    '''Define summaries'''
    summ_g_loss = tf.summary.scalar('g_loss', losses['g_loss'])
    summ_d_loss = tf.summary.scalar('d_loss', losses['d_loss'])
    summ_merged = tf.summary.merge_all()

    tile_image = util.make_tile_image(
        generator=generator,
        image_shape=g_model_params['output_image_shape'],
        input_shape=g_model_params['input_shape'],
        shape_per_class=g_model_params['summ_shape_per_class']
    )
    summ_image = tf.summary.image('generator', tile_image, max_outputs=1)

    '''Setup InceptionNet'''
    inception_graph = tf.Graph()
    inception_path = fid.check_or_download_inception(None)
    with inception_graph.as_default():
        fid.create_inception_graph(inception_path)
    inception_sess = tf.Session(graph=inception_graph)

    epoch_var = tf.get_variable('epoch', shape=(), dtype=tf.int32)
    '''Run training'''
    with tf.Session() as (sess
    ), tf.summary.FileWriter(os.path.join(summ_root, run_name), sess.graph) as train_summ_writer:

        '''Load validation/test data'''
        print('loading data...')
        all_set = (data_set.batch(data_set_size))
        (data_all, all_label) = sess.run(all_set.make_one_shot_iterator().get_next())
        print('data loading done')
        print('calculating InceptionNet activations...')
        mu, sigma = _get_statistics(stat_root, data_all, g_model_params['output_image_shape'], inception_sess)
        print('activation calculation done')
        '''Initialize misc training variables'''
        prefix_tag = run_name
        chk_path = os.path.join(chk_root, run_name, prefix_tag+"model.ckpt")
        chk_saver = tf.train.Saver()
        initial_epoch = 0
        if os.path.exists(chk_path+".index"):
            print('Resuming from previous checkpoint {}'.format(chk_path))
            chk_saver.restore(sess, chk_path)
            initial_epoch = sess.run(epoch_var) + 1
        else:
            tf.global_variables_initializer().run()

        for epoch in range(initial_epoch, train_params['max_epoch']):
            epoch_update_op = epoch_var.assign(epoch)
            sess.run(epoch_update_op)

            it = 0
            batch_size=train_params['batch_size']
            while it < data_all.shape[0]:
                data, label = data_all[it:it+train_params['batch_size']], all_label[it:it+train_params['batch_size']]
                it += train_params['batch_size']
                if it >= data_all.shape[0]:
                    batch_size = data_all.shape[0] % train_params['batch_size']
                    if batch_size==0:
                        batch_size = train_params['batch_size']
                else:
                    batch_size=train_params['batch_size']
                '''Train discriminator'''
                d_loss =_run_discriminator_train(sess=sess, placeholders=placeholders,
                                            loss=losses['d_loss'], optim=d_optim,
                                            x_data=data, y_label=label,
                                            batch_size=batch_size,
                                            z_input_shape=g_model_params['input_shape'][0])
                '''Train generator'''
                g_loss = _run_generator_train(sess=sess, placeholders=placeholders,
                                        loss=losses['g_loss'], optim=g_optim,
                                        x_data=data,
                                        batch_size=batch_size,
                                        z_input_shape=g_model_params['input_shape'][0],
                                        y_input_shape=g_model_params['input_shape'][1])

            '''Training loss summary'''
            _run_summary(sess=sess, placeholders=placeholders,
                        writer=train_summ_writer, summ=summ_merged, epoch=epoch,
                        x_data=data, y_label=label,
                        batch_size=batch_size, z_input_shape=g_model_params['input_shape'][0])

            '''Make image tile'''
            _run_tile_summary(sess=sess, placeholders=placeholders, epoch=epoch,
                            writer=train_summ_writer, summ=summ_image,
                            shape_per_class=g_model_params['summ_shape_per_class'],
                            z_input_shape=g_model_params['input_shape'][0], y_input_shape=g_model_params['input_shape'][1])

            '''Calculate FID of a sample'''
            _log_scalar('fid', epoch, train_summ_writer,
                        _run_fid_calculation(sess=sess, inception_sess=inception_sess,
                                                placeholders=placeholders, batch_size=100, iteration=1,
                                                generator=generator,
                                                mu=mu, sigma=sigma, epoch=epoch,
                                                image_shape=g_model_params['output_image_shape'],
                                                z_input_shape=g_model_params['input_shape'][0],
                                                y_input_shape=g_model_params['input_shape'][1]))

            '''Save checkpoint'''
            chk_saver.save(sess, chk_path)
            print('epoch {} : d_loss : {} / g_loss : {}'.format(epoch, d_loss, g_loss))

        '''Finalize Loop'''
        '''Calculate FID of a sample'''
        final_fid = _run_fid_calculation(sess=sess, inception_sess=inception_sess,
                                            placeholders=placeholders, batch_size=100, iteration=10,
                                            generator=generator,
                                            mu=mu, sigma=sigma, epoch=train_params['max_epoch']-1,
                                            image_shape=g_model_params['output_image_shape'],
                                            z_input_shape=g_model_params['input_shape'][0],
                                            y_input_shape=g_model_params['input_shape'][1])
        print('final_fid: {}'.format(final_fid))
        _log_scalar('final_fid', train_params['max_epoch']-1, train_summ_writer, final_fid)
        inception_sess.close()

