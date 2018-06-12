import os.path as p
import tensorflow as tf

def dataset(directory):
    img_root = p.join(directory, 'Img', 'img_align_celeba')
    attr_file = p.join(directory, 'list_attr_celeba.txt')

    def image_filename_gen():
        for i in range(202599):
            yield p.join(img_root, '{:06d}_re.jpg'.format(i+1))

    def decode_jpeg(path):
        image_contents = tf.read_file(path)
        return tf.cast(tf.image.decode_jpeg(image_contents, channels=3), tf.float32)

    def make_zero_centered(img):
        return (img / tf.constant(255.0)) * tf.constant(2.0) - tf.constant(1.0)

    def make_male_attr(line):
        male_attr_ind = 20
        male = (tf.one_hot(tf.maximum(
                    tf.string_to_number(
                        tf.squeeze(tf.sparse_tensor_to_dense(tf.string_split([line]), ''))[male_attr_ind+1],
                        out_type=tf.int32),
                    tf.constant(0)), depth=2)
               )
        return male

    images = (tf.data.Dataset.from_generator(image_filename_gen, tf.string, tf.TensorShape([]))
              .map(decode_jpeg)
              .map(make_zero_centered))
    labels = (tf.data.TextLineDataset(tf.constant(attr_file))
             .skip(2) # for the first and second line for record count and column name
             .map(make_male_attr).take(202599))
    return tf.data.Dataset.zip((images, labels))
        
