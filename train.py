#!/usr/bin/env python

import os
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import sys
import select
# from IPython import embed

import imagenet_input as data_input
import resnet
import argparse


parser = argparse.ArgumentParser(description='Resnet18')

# Dataset Configuration
# tf.app.flags.DEFINE_string('train_dataset', 'scripts/train_shuffle.txt', """Path to the ILSVRC2012 the training dataset list file""")
parser.add_argument('--train_dataset', type=str, default='scripts/train_shuffle.txt',
                    help='Path to the ILSVRC2012 the training dataset list file')
# tf.app.flags.DEFINE_string('train_image_root', '/data1/common_datasets/imagenet_resized/', """Path to the root of ILSVRC2012 training images""")
parser.add_argument('--train_image_root', type=str, default='/data1/common_datasets/imagenet_resized/',
                    help='Path to the root of ILSVRC2012 training images')
# tf.app.flags.DEFINE_string('val_dataset', 'scripts/val.txt', """Path to the test dataset list file""")
parser.add_argument('--val_dataset', type=str, default='scripts/val.txt', help='Path to the test dataset list file')
# tf.app.flags.DEFINE_string('val_image_root', '/data1/common_datasets/imagenet_resized/ILSVRC2012_val/', """Path to the root of ILSVRC2012 test images""")
parser.add_argument('--val_image_root', type=str, default='/data1/common_datasets/imagenet_resized/ILSVRC2012_val/',
                    help='Path to the root of ILSVRC2012 test images')
# tf.app.flags.DEFINE_string('mean_path', './ResNet_mean_rgb.pkl', """Path to the imagenet mean""")
parser.add_argument('--mean_path', type=str, default='./ResNet_mean_rgb.pkl', help='Path to the imagenet mean')
# tf.app.flags.DEFINE_integer('num_classes', 1000, """Number of classes in the dataset.""")
parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes in the dataset')
# tf.app.flags.DEFINE_integer('num_train_instance', 1281167, """Number of training images.""")
parser.add_argument('--num_train_instance', type=int, default=1281167, help='Number of training images')
# tf.app.flags.DEFINE_integer('num_val_instance', 50000, """Number of val images.""")
parser.add_argument('--num_val_instance', type=int, default=50000, help='Number of val images')

# Network Configuration
# tf.app.flags.DEFINE_integer('batch_size', 256, """Number of images to process in a batch.""")
parser.add_argument('--batch_size', type=int, default=256, help='Number of images to process in a batch')
# tf.app.flags.DEFINE_integer('num_gpus', 1, """Number of GPUs.""")
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs')

# Optimization Configuration
# tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
parser.add_argument('--l2_weight', type=float, default=0.0001, help='L2 loss weight applied all the weights')
# tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
parser.add_argument('--momentum', type=float, default=0.9, help='The momentum of MomentumOptimizer')
# tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
parser.add_argument('--initial_lr', type=float, default=0.1, help='Initial learning rate')
# tf.app.flags.DEFINE_string('lr_step_epoch', "30.0,60.0", """Epochs after which learing rate decays""")
parser.add_argument('--lr_step_epoch', type=str, default="30.0,60.0", help='Epochs after which learing rate decays')
# tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")
parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay factor')
# tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")
parser.add_argument('--finetune', action='store_false', help='Whether to finetune')

# Training Configuration
# tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
parser.add_argument('--train_dir', type=str, default='./train', help='Directory where to write log and checkpoint')
# (added)
# tf.app.flags.DEFINE_integer('init_step', 0, """Change if training from an existing model""")
parser.add_argument('--init_step', type=int, default=0, help='Change if training from an existing model')
# tf.app.flags.DEFINE_integer('max_steps', 500000, """Number of batches to run.""")
parser.add_argument('--max_steps', type=int, default=500000, help='Number of steps to run')
# tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
parser.add_argument('--display', type=int, default=100, help='Number of iterations to display training info')
# tf.app.flags.DEFINE_integer('val_interval', 1000, """Number of iterations to run a val""")
parser.add_argument('--val_interval', type=int, default=1000, help='Number of iterations to run a val')
# tf.app.flags.DEFINE_integer('val_iter', 100, """Number of iterations during a val""")
parser.add_argument('--val_iter', type=int, default=100, help='Number of iterations during a val')
# tf.app.flags.DEFINE_integer('checkpoint_interval', 10000, """Number of iterations to save parameters as a checkpoint""")
parser.add_argument('--checkpoint_interval', type=int, default=5000, help='Number of iterations to save parameters as a checkpoint')
# tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
parser.add_argument('--gpu_fraction', type=float, default=0.95, help='The fraction of GPU memory to be allocated')
# tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
parser.add_argument('--log_device_placement', action='store_false', help='Whether to log device placement')
# tf.app.flags.DEFINE_string('basemodel', None, """Base model to load paramters""")
parser.add_argument('--basemodel', type=str, default=None, help='Base model to load paramters')
# tf.app.flags.DEFINE_string('checkpoint', None, """Model checkpoint to load""")
parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint to load')

# FLAGS = tf.app.flags.FLAGS
params = parser.parse_args()

def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
    lr = initial_lr
    for s in lr_decay_steps:
        if global_step >= s:
            lr *= lr_decay
    return lr


def train():
    print('[Dataset Configuration]')
    print('\tImageNet training root: %s' % params.train_image_root)
    print('\tImageNet training list: %s' % params.train_dataset)
    print('\tImageNet val root: %s' % params.val_image_root)
    print('\tImageNet val list: %s' % params.val_dataset)
    print('\tNumber of classes: %d' % params.num_classes)
    print('\tNumber of training images: %d' % params.num_train_instance)
    print('\tNumber of val images: %d' % params.num_val_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % params.batch_size)
    print('\tNumber of GPUs: %d' % params.num_gpus)
    print('\tBasemodel file: %s' % params.basemodel)

    print('[Optimization Configuration]')
    print('\tL2 loss weight: %f' % params.l2_weight)
    print('\tThe momentum optimizer: %f' % params.momentum)
    print('\tInitial learning rate: %f' % params.initial_lr)
    print('\tEpochs per lr step: %s' % params.lr_step_epoch)
    print('\tLearning rate decay: %f' % params.lr_decay)

    print('[Training Configuration]')
    print('\tTrain dir: %s' % params.train_dir)
    print('\tTraining max steps: %d' % params.max_steps)
    print('\tSteps per displaying info: %d' % params.display)
    print('\tSteps per validation: %d' % params.val_interval)
    print('\tSteps during validation: %d' % params.val_iter)
    print('\tSteps per saving checkpoints: %d' % params.checkpoint_interval)
    print('\tGPU memory fraction: %f' % params.gpu_fraction)
    print('\tLog device placement: %d' % params.log_device_placement)


    with tf.Graph().as_default():
        init_step = params.init_step
        # init_step = 0   # can be fixed if trained from an existing model
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of ImageNet
        import multiprocessing
        num_threads = multiprocessing.cpu_count() / params.num_gpus  # CPU is used to deliver data to several GPUs
        print('Load ImageNet dataset(%d threads)' % num_threads)
        with tf.device('/cpu:0'):
            print('\tLoading training data from %s' % params.train_dataset)
            with tf.variable_scope('train_image'):
                train_images, train_labels = data_input.distorted_inputs(params.train_image_root, params.train_dataset
                                               , params.batch_size, True, num_threads=num_threads, num_sets=params.num_gpus)
            print('\tLoading validation data from %s' % params.val_dataset)
            with tf.variable_scope('test_image'):
                val_images, val_labels = data_input.inputs(params.val_image_root, params.val_dataset
                                               , params.batch_size, False, num_threads=num_threads, num_sets=params.num_gpus)
            tf.summary.image('images', train_images[0][:2]) # [0:2]

        # Build model
        lr_decay_steps = map(float,params.lr_step_epoch.split(','))  # mapping the element in the list according to function
        lr_decay_steps = map(int,[s*params.num_train_instance/params.batch_size/params.num_gpus for s in lr_decay_steps])
        hp = resnet.HParams(batch_size=params.batch_size,
                            num_gpus=params.num_gpus,
                            num_classes=params.num_classes,
                            weight_decay=params.l2_weight,   # weight decay
                            momentum=params.momentum,
                            finetune=params.finetune)    # create a HPrams object
        network_train = resnet.ResNet(hp, train_images, train_labels, global_step, name="train")    # create an class object
    # (dummy) model = get_model(params)  # get_model returns a Model instance
    # (dummy) network_train = Trainer.trainer(model, hp, train_images, train_labels, global_step, name="train")
        network_train.build_model()
        network_train.build_train_op()
        train_summary_op = tf.summary.merge_all()  # Summaries(training)
        network_val = resnet.ResNet(hp, val_images, val_labels, global_step, name="val", reuse_weights=True)
    # (dummy) network_val = Trainer.trainer(model, hp, val_images, val_labels, global_step, name="val", reuse_weights=True)
        network_val.build_model()   # two models are built, one for training, one for validation
        print('Number of Weights: %d' % network_train._weights)
        print('FLOPs: %d' % network_train._flops)   # self._flops


        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=params.gpu_fraction),
            allow_soft_placement=False,
            # allow_soft_placement=True,
            log_device_placement=params.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
        if params.checkpoint is not None:
            print('Load checkpoint %s' % params.checkpoint)
            saver.restore(sess, params.checkpoint)
            init_step = global_step.eval(session=sess)
        elif params.basemodel:
            # Define a different saver to save model checkpoints
            print('Load parameters from basemodel %s' % params.basemodel)
            variables = tf.global_variables()
            vars_restore = [var for var in variables
                            if not "Momentum" in var.name and
                               not "global_step" in var.name]
            saver_restore = tf.train.Saver(vars_restore, max_to_keep=10000)
            saver_restore.restore(sess, params.basemodel)
        else:
            print('No checkpoint file of basemodel found. Start from the scratch.')

        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)

        if not os.path.exists(params.train_dir):
            os.mkdir(params.train_dir)
        summary_writer = tf.summary.FileWriter(os.path.join(params.train_dir, str(global_step.eval(session=sess))),
                                                sess.graph)

        # Training!
        val_best_acc = 0.0
        for step in range(init_step, params.max_steps):
            # val
            if step % params.val_interval == 0:
                val_loss, val_acc = 0.0, 0.0
                for i in range(params.val_iter):
                    loss_value, acc_value = sess.run([network_val.loss, network_val.acc],
                                feed_dict={network_val.is_train:False})
                    val_loss += loss_value
                    val_acc += acc_value
                val_loss /= params.val_iter
                val_acc /= params.val_iter
                val_best_acc = max(val_best_acc, val_acc)
                format_str = ('%s: (val)      step %d, loss=%.4f, acc=%.4f')
                print (format_str % (datetime.now(), step, val_loss, val_acc))

                val_summary = tf.Summary()
                val_summary.value.add(tag='val/loss', simple_value=val_loss)
                val_summary.value.add(tag='val/acc', simple_value=val_acc)
                val_summary.value.add(tag='val/best_acc', simple_value=val_best_acc)
                summary_writer.add_summary(val_summary, step)
                summary_writer.flush()

            # Train
            lr_value = get_lr(params.initial_lr, params.lr_decay, lr_decay_steps, step)
            start_time = time.time()
            _, loss_value, acc_value, train_summary_str = \
                    sess.run([network_train.train_op, network_train.loss, network_train.acc, train_summary_op],
                            feed_dict={network_train.is_train:True, network_train.lr:lr_value})
            
            duration = time.time() - start_time # evaluate training speed, not important

            assert not np.isnan(loss_value) # make sure model converge

            # Display & Summary(training)
            if step % params.display == 0 or step < 5:
                num_examples_per_step = params.batch_size * params.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: (Training) step %d, loss=%.4f, acc=%.4f, lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, acc_value, lr_value,
                                     examples_per_sec, sec_per_batch))
                summary_writer.add_summary(train_summary_str, step)

            # Save the model checkpoint periodically.
            if (step > init_step and step % params.checkpoint_interval == 0) or (step + 1) == params.max_steps:
                checkpoint_path = os.path.join(params.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            # if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            #   char = sys.stdin.read(1)
            #   if char == 'b':
            #     embed()


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
#   tf.app.run()
    main()
