import numpy as np
import functools
import tensorflow as tf
import dynamic_fixed_point as dfxp

from random import randrange
from time import time


def preprocess_image_MNIST(image, label):
    return image, label


def preprocess_image_CIFAR10(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)
    image = tf.random_crop(image, [32, 32, 3])
    return image, label


def parse_input_ImageNet(line, dataset_root):
    items = tf.decode_csv(line, [[''], [0]], field_delim=' ')
    image = tf.image.decode_jpeg(tf.read_file(dataset_root+items[0]), channels=3)
    label = tf.cast(items[1], tf.int32)
    return image, label


def random_resize_image(image):
    height_orig, width_orig = tf.shape(image)[0], tf.shape(image)[1]
    size = tf.random_uniform([], minval=256, maxval=480+1, dtype=tf.int32)
    flag = tf.greater(height_orig, width_orig)
    height = tf.cond(flag,
        lambda : tf.cast(size*height_orig/width_orig, tf.int32),
        lambda : size)
    width = tf.cond(flag,
        lambda : size,
        lambda : tf.cast(size*width_orig/height_orig, tf.int32))
    return tf.image.resize_images(image, [height, width])

def resize_image(image):
    height_orig, width_orig = tf.shape(image)[0], tf.shape(image)[1]
    size = 256
    flag = tf.greater(height_orig, width_orig)
    height = tf.cond(flag,
        lambda : tf.cast(size*height_orig/width_orig, tf.int32),
        lambda : size)
    width = tf.cond(flag,
        lambda : size,
        lambda : tf.cast(size*width_orig/height_orig, tf.int32))
    return tf.image.resize_images(image, [height, width])


def preprocess_image_ImageNet(image, label, training):
    if training:
        image = random_resize_image(image)
        image = tf.random_crop(image, [224, 224, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
    else:
        image = resize_image(image)
        image_shape = tf.shape(image)
        h_offset = tf.cast((image_shape[0]-224)/2, tf.int32)
        w_offset = tf.cast((image_shape[1]-224)/2, tf.int32)
        image = tf.slice(image, [h_offset, w_offset, 0], [224, 224, 3])

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float)
    image = (image / 255 - mean) / std
    image = tf.transpose(image, [2, 0, 1])
    return image, label


def average_gradients(tower_grads):
    """Calculate the average gradient for shared variables.

    Args:
        tower_grads: list of grads_and_vars

    Returns:
        averaged grads_and_vars
    """
    avg_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(tf.stack(grads), axis=0)
        v = grad_and_vars[0][1]
        avg_grads.append((grad, v))
    return avg_grads

# (m)
def average_vars(tower_steps_vars):
    """Calculate the average vars for variables been quantized

    Args:
        tower_steps_vars: list of steps_and_vars
    Returns:
        averaged steps_and_vars
    """
    avg_vars = []
    for steps_and_vars in zip(*tower_steps_vars):
        _vars = [v for _, v in steps_and_vars]
        _var = tf.reduce_mean(tf.stack(_vars), axis=0)
        step = steps_and_vars[0][0]
        avg_vars.append((step, _var))
    return avg_vars
# (m)

def tower_reduce_mean(towers):
    return tf.reduce_mean(tf.stack(towers), axis=0)


class LearningRateScheduler:
    def __init__(self, lr, lr_decay_epoch, lr_decay_factor):
        '''
        Learning rate scheduler.

        Args:
            lr: learning rate variable
            lr_decay_epoch: learning rate decay epoch
            lr_decay_factor: learning rate decay factor
        '''
        self.lr = tf.get_variable('learning_rate', initializer=lr)
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_factor = lr_decay_factor
        self.epoch = tf.get_variable('lr_scheduler_step',
            dtype=tf.int32, initializer=1)

        update_epoch = tf.assign(self.epoch, self.epoch+1)
        with tf.control_dependencies([update_epoch]):
            self.update_lr = tf.assign(self.lr, tf.cond(
                tf.equal(tf.mod(self.epoch, self.lr_decay_epoch), 0),
                lambda : self.lr * self.lr_decay_factor,
                lambda : self.lr,
            ))

    def step(self):
        '''
        Op for updating learning rate.

        Should be called at the end of an epoch.
        '''
        return self.update_lr


class Trainer:
    def __init__(self, model, dataset, dataset_name, logger, params):
        self.n_epoch = params.n_epoch
        self.exp_path = params.exp_path

        self.logger = logger
        self.logger.info('Model info:')

        self.logger.info('Trainer info:')
        self.logger.info('lr %f decay %f in %f epoch' % (
            params.lr, params.lr_decay_factor, params.lr_decay_epoch))
        self.logger.info('momentum %f' % params.momentum)
        self.logger.info('training epoch %d' % params.n_epoch)
        self.logger.info('batch_size %d' % params.batch_size)
        self.logger.info('logdir %s' % params.exp_path)

        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            global_step = tf.train.get_or_create_global_step()

            (self.X_train, self.y_train), (self.X_test, self.y_test) = dataset
            self.train_iterator, self.test_iterator = self.get_dataset_iterators(params, dataset, dataset_name)

            self.lr_scheduler = LearningRateScheduler(params.lr,
                params.lr_decay_epoch, params.lr_decay_factor)  # fixed
            optimizer = tf.train.MomentumOptimizer(self.lr_scheduler.lr, params.momentum)

            tower_grads, tower_loss = [], []
            # (m)
            tower_steps_vars = []
            # (m)
            tower_top1, tower_top5 = [], []
            with tf.variable_scope(tf.get_variable_scope()):

                ### Traing Graph ###
                for i in range(params.n_gpu):
                    with tf.device('/gpu:%d' % i):
                        self.logger.info('Building tower on gpu %d' % i)

                        net = model(params.bits, params.dropout,
                            params.weight_decay, params.stochastic, training=True)
                        tf.get_variable_scope().reuse_variables()
                        self.logger.info('\n' + net.info())

                        images, labels = self.train_iterator.get_next()
                        logits, _ = net.forward(images)
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=labels, logits=logits)
                        loss = tf.reduce_mean(loss)
                        net.backward(tf.gradients(loss, logits)[0])

                        tower_grads.append(net.grads_and_vars())
                        # (m)
                        tower_steps_vars.append(net.steps_and_vars())
                        # (m)
                        with tf.device('/cpu:0'):
                            tower_top1.append(tf.reduce_mean(tf.cast(tf.nn.in_top_k(
                                logits, labels, k=1), tf.float32)))
                            tower_top5.append(tf.reduce_mean(tf.cast(tf.nn.in_top_k(
                                logits, labels, k=5), tf.float32)))
                        tower_loss.append(loss)

                ### Validation Graph ###
                with tf.device('/gpu:0'):
                    self.logger.info('Building validation graph')
                    net = model(params.bits, params.dropout, params.weight_decay,
                        params.stochastic, training=False)
                    images, labels = self.test_iterator.get_next()
                    logits, _ = net.forward(images)
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits)
                    self.val_loss =  tf.reduce_mean(loss)
                    with tf.device('/cpu:0'):
                        self.val_top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(
                            logits, labels, k=1), tf.float32))
                        self.val_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(
                            logits, labels, k=5), tf.float32))

            grads_and_vars = average_gradients(tower_grads)
            # (m)
            steps_and_vars = average_vars(tower_steps_vars)
            for s, v in steps_and_vars:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, dfxp.update_step(
                    v, target_overflow_rate=0, bits=params.bits, step=s))                
            # (m)
            self.train_op = optimizer.apply_gradients(grads_and_vars, global_step)
            self.train_loss = tower_reduce_mean(tower_loss)
            self.train_top1 = tower_reduce_mean(tower_top1)
            self.train_top5 = tower_reduce_mean(tower_top5)
            self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.saver = tf.train.Saver()

        # add summary
        self.summary = tf.summary.merge_all()

    def get_dataset_iterators(self, params, dataset, dataset_name='CIFAR10'):
        if dataset_name == 'MNIST':
            preprocessor = preprocess_image_MNIST
        elif dataset_name == 'CIFAR10':
            preprocessor = preprocess_image_CIFAR10
        elif dataset_name == 'ImageNet':
            preprocessor = preprocess_image_ImageNet
        else:
            assert False, 'Invalid value for `dataset`: %s' % dataset

        if dataset_name in ['MNIST', 'CIFAR10']:
            with tf.device('/cpu:0'):
                (X_train, y_train), (X_test, y_test) = dataset
                self.X_train_placeholder = tf.placeholder(tf.float32, X_train.shape)
                self.X_test_placeholder = tf.placeholder(tf.float32, X_test.shape)
                self.y_train_placeholder = tf.placeholder(tf.int32, y_train.shape)
                self.y_test_placeholder = tf.placeholder(tf.int32, y_test.shape)

                train_dataset = tf.data.Dataset.from_tensor_slices(
                    (self.X_train_placeholder, self.y_train_placeholder))
                train_dataset = (train_dataset.shuffle(buffer_size=X_train.shape[0])
                                .map(preprocessor, num_parallel_calls=4)
                                .batch(params.batch_size)
                                .prefetch(1)
                                )
                train_iterator = train_dataset.make_initializable_iterator()

                test_dataset = tf.data.Dataset.from_tensor_slices(
                    (self.X_test_placeholder, self.y_test_placeholder)).batch(1000)
                test_iterator = test_dataset.make_initializable_iterator()
        elif dataset_name == 'ImageNet':
            with tf.device('/cpu:0'):
                (train_list, train_root), (test_list, test_root) = dataset
                train_root, test_root = tf.constant(train_root), tf.constant(test_root)

                parse_input = functools.partial(parse_input_ImageNet, dataset_root=train_root)
                preprocessor = functools.partial(preprocess_image_ImageNet, training=True)
                train_dataset = (tf.data.TextLineDataset([train_list])
                                .shuffle(buffer_size=10000)
                                .map(parse_input, num_parallel_calls=24)
                                .apply(tf.data.experimental.map_and_batch(
                                    preprocessor, params.batch_size // params.n_gpu,
                                    num_parallel_calls=8))
                                .prefetch(2 * params.n_gpu)
                                )
                train_iterator = train_dataset.make_initializable_iterator()

                parse_input = functools.partial(parse_input_ImageNet, dataset_root=test_root)
                preprocessor = functools.partial(preprocess_image_ImageNet, training=False)
                test_dataset = (tf.data.TextLineDataset([test_list])
                                .map(parse_input, num_parallel_calls=24)
                                .map(preprocessor, num_parallel_calls=24)
                                .batch(params.batch_size)
                                .prefetch(2)
                                )
                test_iterator = test_dataset.make_initializable_iterator()
        else:
            assert False, 'Invalid value for `dataset`: %s' % dataset

        return train_iterator, test_iterator

    def train(self):
        self.logger.info('Start of training')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        config.log_device_placement = False
        with tf.Session(config=config, graph=self.graph) as sess:
            self.logger.info('Initializing variables..')
            sess.run(tf.global_variables_initializer())
            best_val_top1, best_val_top5 = 0, 0

            for epoch in range(self.n_epoch):
                # start training
                sess.run(self.train_iterator.initializer)
                b = 0
                while True:
                    try:
                        t1 = time()
                        b += 1
                        _, _, loss, top1, top5 = sess.run([self.train_op, self.update_op,
                            self.train_loss, self.train_top1, self.train_top5])
                        t2 = time()
                        if b % 50 == 0:
                            self.logger.info('Batch %d loss %f top1 %f top5 %f time %f' %
                                (b, loss, top1, top5, t2-t1))
                    except tf.errors.OutOfRangeError:
                        break

                # start validation
                sess.run(self.test_iterator.initializer)
                val_top1, val_top5, val_loss, batch_cnt = 0, 0, 0, 0
                while True:
                    try:
                        batch_cnt += 1
                        top1, top5, loss = sess.run([self.val_top1, self.val_top5, self.val_loss])
                        val_top1 += top1
                        val_top5 += top5
                        val_loss += loss
                    except tf.errors.OutOfRangeError:
                        break
                val_top1 /= batch_cnt
                val_top5 /= batch_cnt
                val_loss /= batch_cnt
                best_val_top1 = max(best_val_top1, val_top1)
                best_val_top5 = max(best_val_top5, val_top5)
                self.logger.info('Epoch %d loss %f top1 %f (%f) top5 %f (%f)' %
                    (epoch+1, val_loss, val_top1, best_val_top1, val_top5, best_val_top5))

                # lr_scheduler
                sess.run(self.lr_scheduler.step())
                self.saver.save(sess, self.exp_path+('/epoch%d.ckpt'%(epoch+1)))
