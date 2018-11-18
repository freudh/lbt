import numpy as np
import tensorflow as tf

from random import randrange

import dynamic_fixed_point as dfxp


def batch_generator(X, y, shuffle=True, batch_size=32):
    n = X.shape[0]
    n_batch = (n-1)//batch_size + 1

    shuffle_idx = np.arange(n)
    if shuffle:
        shuffle_idx = np.random.permutation(shuffle_idx)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    for batch in range(n_batch):
        start = batch * batch_size
        end = (batch+1) * batch_size
        yield X[start:end], y[start:end], batch


def preprocess_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)
    image = tf.random_crop(image, [32, 32, 3])
    return image, label


class Trainer:
    def __init__(self, model, dataset, logger, logdir,
                 lr=1e-3, lr_decay_factor=0.1, lr_decay_epoch=50,
                 momentum=0.9, n_epoch=50, batch_size=32, m_continue=False):

        self.model = model
        self.model.backward()

        self.n_epoch = n_epoch
        self.batch_size = batch_size

        (self.X_train, self.y_train), (self.X_test, self.y_test) = dataset
        self.train_iterator, self.test_iterator = self.get_dataset_iterators(dataset)

        self.logger = logger
        self.logger.info('Model info:')
        self.logger.info('\n' + model.info())

        self.logger.info('Trainer info:')
        self.logger.info('lr %f decay %f in %f epoch' % (
            lr, lr_decay_factor, lr_decay_epoch))
        self.logger.info('momentum %f' % momentum)
        self.logger.info('training epoch %d' % n_epoch)
        self.logger.info('batch_size %d' % batch_size)
        self.logger.info('m_continue %r' % m_continue)        
        self.logger.info('logdir %s' % logdir)


        self.global_step = tf.train.get_or_create_global_step()
        self.lr = lr
        self.momentum = momentum
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_factor = lr_decay_factor

        self.update_range_op = tf.get_collection('update_range')

        # add summary
        self.summary = tf.summary.merge_all()

        # reduce memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True # pylint: disable=E1101
        self.sess = tf.Session(config=config)
        self.train_writer = tf.summary.FileWriter(logdir+'/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(logdir+'/test', self.sess.graph)

        self.m_continue = m_continue

    def init_model(self):
        if self.m_continue == False:
            self.logger.info('Initializing model')
            self.sess.run(tf.global_variables_initializer())
        else:
            self.new_saver = tf.train.import_meta_graph('/home/jiandong/cysu_lbt/tmp/ckpt/model.ckpt.meta')
            self.new_saver.restore(self.sess, tf.train.latest_checkpoint('/home/jiandong/cysu_lbt/tmp/ckpt', latest_filename='checkpoint'))

    def get_train_op(self):
        # This reset the optimizer variables after lr/momentum changes
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum)
        train_op = optimizer.apply_gradients(self.model.grads_and_vars(), self.global_step)
        self.sess.run(tf.variables_initializer(optimizer.variables()))
        return train_op

    def get_dataset_iterators(self, dataset):
        (X_train, y_train), (X_test, y_test) = dataset

        with tf.device('/cpu:0'):
            self.X_train_placeholder = tf.placeholder(tf.float32, X_train.shape)
            self.X_test_placeholder = tf.placeholder(tf.float32, X_test.shape)
            self.y_train_placeholder = tf.placeholder(tf.int32, y_train.shape)
            self.y_test_placeholder = tf.placeholder(tf.int32, y_test.shape)

            train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train_placeholder, self.y_train_placeholder))
            train_dataset = (train_dataset.shuffle(buffer_size=X_train.shape[0])
                            .map(preprocess_image, num_parallel_calls=4)
                            .batch(self.batch_size)
                            .prefetch(1)
                            )
            train_iterator = train_dataset.make_initializable_iterator()

            test_dataset = tf.data.Dataset.from_tensor_slices(
                (self.X_test_placeholder, self.y_test_placeholder)).batch(1000)
            test_iterator = test_dataset.make_initializable_iterator()

        return train_iterator, test_iterator

    def train(self):
        self.logger.info('Start of training')

        next_train_op = self.train_iterator.get_next()
        next_test_op = self.test_iterator.get_next()
        train_iter_init_op = self.train_iterator.initializer
        test_iter_init_op = self.test_iterator.initializer

        for epoch in range(self.n_epoch):
            if epoch == 0:
                self.logger.info('New training optimizer with lr=%f' % self.lr)
                train_op = self.get_train_op()
            elif epoch == 75:
                self.lr *= self.lr_decay_factor
                self.logger.info('New training optimizer with lr=%f' % self.lr)
                train_op = self.get_train_op()
            elif epoch == 120:
                self.lr *= self.lr_decay_factor
                self.logger.info('New training optimizer with lr=%f' % self.lr)
                train_op = self.get_train_op()                     
                   
            # if epoch % self.lr_decay_epoch == 0:
            #     self.logger.info('New training optimizer with lr=%f' % self.lr)
            #     train_op = self.get_train_op()
            #     self.lr *= self.lr_decay_factor

            self.sess.run([self.model.set_training, train_iter_init_op], feed_dict={
                self.X_train_placeholder: self.X_train,
                self.y_train_placeholder: self.y_train,
            })
            b = 0
            while True:
                try:
                    X, y = self.sess.run(next_train_op)
                    b += 1
                    if b % 100 == 0: 
                        _, _, loss, acc, summary, step, _, _, _ = self.sess.run([train_op, self.update_range_op,
                            self.model.loss, self.model.accuracy, self.summary, self.global_step,
                            dfxp.pre_dense_op,
                            dfxp.pre_conv_op, dfxp.pre_rescale_op
                            ],
                            feed_dict={self.model.input_X: X, self.model.input_y: y})
                        self.train_writer.add_summary(summary, step)
                        self.logger.info('Batch %d loss %f acc %f' % (b, loss, acc))
                    else:
                        self.sess.run([train_op, self.update_range_op,
                            dfxp.pre_dense_op,
                            dfxp.pre_conv_op, dfxp.pre_rescale_op
                            ], 
                            feed_dict={self.model.input_X: X, self.model.input_y: y})
                except tf.errors.OutOfRangeError:
                    break

            self.sess.run(self.model.set_testing)   # change mode to test
            self.sess.run(test_iter_init_op, feed_dict={
                self.X_test_placeholder: self.X_test,
                self.y_test_placeholder: self.y_test,
            })
            test_acc = 0
            test_loss = 0
            batch_cnt = 0
            while True:
                try:
                    X, y = self.sess.run(next_test_op)
                    batch_cnt += 1
                    acc, loss, summary, step = self.sess.run([self.model.accuracy, self.model.loss,
                        self.summary, self.global_step],
                        feed_dict={self.model.input_X: X, self.model.input_y: y})
                    test_acc += acc
                    test_loss += loss
                    self.test_writer.add_summary(summary, step+batch_cnt*50) # some trick
                except tf.errors.OutOfRangeError:
                    break
            test_acc /= batch_cnt
            test_loss /= batch_cnt
            self.logger.info('Epoch %d test accuracy %f' % (epoch+1, test_acc))
            # checkpoint: save_model
            self.save_model_per('/home/jiandong/cysu_lbt/tmp/ckpt')

    def save_model_per(self, exp_path):
        # saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=0.5)
        saver = tf.train.Saver(max_to_keep=3)
        saver.save(self.sess, exp_path+'/model.ckpt')

    def save_model(self, exp_path):
        self.logger.info('Saving model')
        saver = tf.train.Saver()
        saver.save(self.sess, exp_path+'/model.ckpt')
