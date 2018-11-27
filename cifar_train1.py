from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
import tensorflow as tf

import cifar10
import cifar10_input


FLAGS = tf.app.flagS.FLAGS

tf.app.flags.DEFINE_string('log_dir', 'data/train')
tf.app.flags.DEFINE_integer('MAX_STEP', 10000)
tf.app.flags.DEFINE_boolean('log_device_placement', False)

def train():
    with tf.Graph().as_default():

        my_global_step = tf.Variable(0, name='global_step', trainable=False)

        data_dir = '/home/bob/PycharmProjects/CIFAR10/data/'
        log_dir = '/home/bob/PycharmProjects/CIFAR10/logs/'

        images, labels = cifar10_input.read_cifar10(data_dir=data_dir, is_train=True, batch_size=BATCH_SIZE, shuffle=True)

        logits = cifar10.inference(images)
        loss = cifar10.losses(logits, labels)
        train_op = cifar10.train(loss, global_step=my_global_step)

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        #
        # train_op = optimizer.minimize(loss, global_step=my_global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.summary.merge_all()

        init = tf.initializer_all_variables()


        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))

        sess.run(init)


        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph_def=sess.graph_def)

        # try:
        #     for step in np.arange(MAX_STEP):
        #
        #         _, loss_value = sess.run([train_op, loss])
        #         if step % 50 == 0:
        #             print('Step: %d, loss: %.4f'%(step, loss_value))
        #
        #         if step % 100 == 0:
        #             summary_str = sess.run(summary_op)
        #             summary_writer.add_summary(summary_str, step)
        #
        #         if step % 2000 == 0 or (step+1) == MAX_STEP:
        #             checkpoint_path = os.path.join(log_dir, 'model.ckpt')
        #             saver.save(sess, checkpoint_path, global_step=step)
        # except tf.errors.OutOfRangeError:
        #     print('Done training')
        #
        # finally:
        #     coord.request_stop()
        # coord.join(threads)
        # sess.close()
        for step in arange(FLAGS.MAX_STEP):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)


            if step % 1000 == 0 or (step + 1) == FLAGS.MAX_STEP:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    def main(argv=None):
        cifar10.maybe_download_and_extract()
        if gfile.Exists(FLAGS.train_dir):
            gfile.DeleteRecursively(FLAGS.train_dir)
        gfile.MakeDirs(FLAGS.train_dir)

        train()

    if __name__ == '__main__':
    tf.app.run()