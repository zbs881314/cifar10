import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


# reading the data

def read_cifar10(data_dir, is_train, batch_size, shuffle):    # is_train is true input train data or input test data
    '''
    Read CIFAR10
    Args:
    data_dir: the directory of CIFAR10
    Returns:
        label: 1D tensor, tf.int32
        image: 4D tensor, [batch_size, height, width, 3], tf.float32
    '''
    img_width = 32
    img_height = 32
    img_depth = 3
    label_bytes = 1
    image_bytes = img_width*img_height*img_depth

    with tf.name_scope('input'):
        if is_train:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %ii) # the dataset of file data
                         for ii in np.arange(1, 6)]
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]

        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)
        key, value = reader.read(filename_queue)  # get the value
        record_bytes = tf.decode_raw(value, tf.uint8)   # decode value to uint8

        label = tf.slice(record_bytes, [0], [label_bytes])  # get label
        label = tf.cast(label, tf.int32)
        image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])   # get image
        image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])
        image = tf.transpose(image_raw, (1, 2, 0))  # convert from D/H/W to H/W/D
        image = tf.cast(image, tf.float32)

        # data argumentation
        #image = tf.random_crop(image, [24, 24, 3])
        #image = tf.image.random_flip_left_right(image)
        #image = tf.image.random_brightness(image, max_delta=63)
        #image = tf.image.random_contrast(image, lower=0.2, upper=1.8)


        image = tf.image.per_image_standardization(image)     # normolization


        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size = batch_size,
                num_threads = 16,
                capacity = 2000,
                min_after_dequeue = 1500
            )

        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size = batch_size,
                num_threads = 16,
                capacity = 2000
            )

        return images, tf.reshape(label_batch, [batch_size])

# TEST

# data_dir = '/home/bob/PycharmProjects/CIFAR10/data/'
# BATCH_SIZE = 10
# image_batch, label_batch = read_cifar10(data_dir,
#                                         is_train=True,
#                                         batch_size=BATCH_SIZE,
#                                         shuffle=True)
#
# with tf.Session() as sess:
#     i=0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     try:
#         while not coord.should_stop() and i<1:
#             img, label = sess.run([image_batch, label_batch])
#             for j in np.arange(BATCH_SIZE):
#                 print('label: %d' %label[j])
#                 #plt.imshow(img[j, :, :, :])
#                 #plt.show()
#             i+=1
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(threads)

#



