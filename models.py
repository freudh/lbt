import numpy as np
import tensorflow as tf

import dynamic_fixed_point as dfxp


class Model:
    def __init__(self, bits, input_shape, dropout, weight_decay, stochastic):
        self.bits = bits
        self.input_X = tf.placeholder(tf.float32, input_shape)
        self.input_y = tf.placeholder(tf.int32, [None])

        self.training = tf.Variable(True, tf.bool)
        self.set_training = tf.assign(self.training, True)
        self.set_testing = tf.assign(self.training, False)
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.stochastic = stochastic

        self.layers = self.get_layers()

        X = self.input_X
        for layer in self.layers:
            X = layer.forward(X)
        self.logits = X

        self.predictions = tf.argmax(self.logits, axis=1, output_type=tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            self.predictions, self.input_y), tf.float32))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.input_y, logits=self.logits)
        self.loss = tf.reduce_mean(self.loss)

        with tf.name_scope('metric'):
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('loss', self.loss)

    def get_layers(self):
        return []

    def grads_and_vars(self):
        res = []
        for layer in self.layers:
            res += layer.grads_and_vars()
        return res

    def backward(self):
        grad = tf.gradients(self.loss, self.logits)[0]
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.stochastic)
        return grad

    def info(self):
        return '\n'.join([layer.info() for layer in self.layers])


class PI_MNIST_Model(Model):
    def __init__(self, bits, dropout=0.5, weight_decay=0, stochastic=False):
        super().__init__(bits, [None, 784], dropout, weight_decay, stochastic)

    def get_layers(self):
        return [
            dfxp.Dense_q(
                name='dense1',
                bits=self.bits,
                in_units=784,
                units=1024,
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Dense_q(
                name='dense2',
                bits=self.bits,
                in_units=1024,
                units=1024,
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Dense_q(
                name='softmax',
                bits=self.bits,
                in_units=1024,
                units=10,
                weight_decay=self.weight_decay,
            ),
        ]


class MNIST_Model(Model):
    def __init__(self, bits, dropout=0.5, weight_decay=0, stochastic=False):
        super().__init__(bits, [None, 28, 28, 1], dropout, weight_decay, stochastic)

    def get_layers(self):
        return [
            dfxp.Conv2d_q(
                name='conv1',
                bits=self.bits,
                ksize=[5, 5, 1, 6],
                strides=[1, 1, 1, 1],
                padding='SAME',
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),
            dfxp.MaxPool_q(
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID'
            ),
            dfxp.Conv2d_q(
                name='conv2',
                bits=self.bits,
                ksize=[5, 5, 6, 16],
                strides=[1, 1, 1, 1],
                padding='VALID',
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),
            dfxp.MaxPool_q(
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID'
            ),
            dfxp.Conv2d_q(
                name='conv3',
                bits=self.bits,
                ksize=[5, 5, 16, 120],
                strides=[1, 1, 1, 1],
                padding='VALID',
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),
            dfxp.Flatten_q(120),
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Dense_q(
                name='dense1',
                bits=self.bits,
                in_units=120,
                units=84,
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Dense_q(
                name='softmax',
                bits=self.bits,
                in_units=84,
                units=10,
                weight_decay=self.weight_decay,
            ),
        ]


class CIFAR10_Model(Model):
    def __init__(self, bits, dropout=0.5, weight_decay=0, stochastic=False):
        super().__init__(bits, [None, 32, 32, 3], dropout, weight_decay, stochastic)

    def get_layers(self):
        return [
            # conv1
            # dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Conv2d_q(
                name='conv1',
                bits=self.bits,
                ksize=[5, 5, 3, 64],
                strides=[1, 1, 1, 1],
                padding='SAME',
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),
            dfxp.MaxPool_q(
                ksize=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding='SAME'
            ),

            # conv2
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Conv2d_q(
                name='conv2',
                bits=self.bits,
                ksize=[5, 5, 64, 128],
                strides=[1, 1, 1, 1],
                padding='SAME',
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),
            dfxp.MaxPool_q(
                ksize=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding='SAME'
            ),

            # conv3
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Conv2d_q(
                name='conv3',
                bits=self.bits,
                ksize=[5, 5, 128, 128],
                strides=[1, 1, 1, 1],
                padding='SAME',
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),
            dfxp.MaxPool_q(
                ksize=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding='SAME'
            ),

            dfxp.Flatten_q(128*4*4),

            # dense1
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Dense_q(
                name='dense1',
                bits=self.bits,
                in_units=128*4*4,
                units=400,
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),

            # softmax
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Dense_q(
                name='softmax',
                bits=self.bits,
                in_units=400,
                units=10,
                weight_decay=self.weight_decay,
            ),
        ]


class CIFAR10_VGG_Model(Model):
    def __init__(self, bits, dropout=0.5, weight_decay=0, stochastic=False):
        super().__init__(bits, [None, 32, 32, 3], dropout, weight_decay, stochastic)

    def get_layers(self):
        return [
            # conv1-1
            dfxp.Conv2d_q(
                name='conv1-1',
                bits=self.bits,
                ksize=[3, 3, 3, 128],
                strides=[1, 1, 1, 1],
                padding='SAME',
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),

            # conv1-2
            # dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Conv2d_q(
                name='conv1-2',
                bits=self.bits,
                ksize=[3, 3, 128, 128],
                strides=[1, 1, 1, 1],
                padding='SAME',
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),

            # pool1
            dfxp.MaxPool_q(
                ksize=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding='SAME'
            ),

            # conv2-1
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Conv2d_q(
                name='conv2-1',
                bits=self.bits,
                ksize=[3, 3, 128, 256],
                strides=[1, 1, 1, 1],
                padding='SAME',
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),

            # conv2-2
            # dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Conv2d_q(
                name='conv2-2',
                bits=self.bits,
                ksize=[3, 3, 256, 256],
                strides=[1, 1, 1, 1],
                padding='SAME',
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),

            # pool2
            dfxp.MaxPool_q(
                ksize=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding='SAME'
            ),

            # conv3-1
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Conv2d_q(
                name='conv3-1',
                bits=self.bits,
                ksize=[3, 3, 256, 512],
                strides=[1, 1, 1, 1],
                padding='SAME',
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),

            # conv3-2
            # dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Conv2d_q(
                name='conv3-2',
                bits=self.bits,
                ksize=[3, 3, 512, 512],
                strides=[1, 1, 1, 1],
                padding='SAME',
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),

            # pool3
            dfxp.MaxPool_q(
                ksize=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding='SAME'
            ),

            dfxp.Flatten_q(512*4*4),

            # dense1
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Dense_q(
                name='dense1',
                bits=self.bits,
                in_units=512*4*4,
                units=1024,
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),

            # dense2
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Dense_q(
                name='dense2',
                bits=self.bits,
                in_units=1024,
                units=1024,
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),

            # softmax
            dfxp.Dropout_q(self.dropout, self.training),
            dfxp.Dense_q(
                name='softmax',
                bits=self.bits,
                in_units=1024,
                units=10,
                weight_decay=self.weight_decay,
            ),
        ]


class CIFAR10_Resnet(Model):
    def __init__(self, bits, num_blocks, block, dropout=0.5, weight_decay=0, stochastic=False):
        self.num_blocks = num_blocks
        self.block = block
        super().__init__(bits, [None, 32, 32, 3], dropout, weight_decay, stochastic)

    def _build_blocks(self, channels, num_blocks, stride):
        # blocks = [dfxp.Dropout_q(self.dropout, self.training)]
        blocks = []
        for i in range(1, 1+num_blocks):
            blocks.append(
                self.block(
                    name='block%d-%d' % (channels, i),
                    bits=self.bits,
                    in_channels=self.channels,
                    channels=channels,
                    stride=1 if i>1 else stride,
                    training=self.training,
                    weight_decay=self.weight_decay,
                )
            )
            self.channels = channels * self.block.expansion
        return blocks

    def get_layers(self):
        self.channels = 16
        return [
            dfxp.Conv2d_q(
                name='conv1',
                bits=self.bits,
                ksize=[3, 3, 3, 16],
                strides=[1, 1, 1, 1],
                padding='SAME',
                use_bias=False,
                weight_decay=self.weight_decay,
            ),
            dfxp.BatchNorm_q(
                name='conv1-bn',
                bits=self.bits,
                num_features=16,
                training=self.training,
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),
        ] + self._build_blocks(16, self.num_blocks[0], 1) \
          + self._build_blocks(32, self.num_blocks[1], 2) \
          + self._build_blocks(64, self.num_blocks[2], 2) \
          + [
            dfxp.AvgPool_q(
                ksize=[1, 8, 8, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
            ),
            dfxp.Flatten_q(64),
            dfxp.Dense_q(
                name='softmax',
                bits=self.bits,
                in_units=64,
                units=10,
                use_bias=False,
                weight_decay=self.weight_decay,
            ),
            dfxp.GradientBuffer_q(
                name='gradient_buffer',
                bits=self.bits,
                shape=[32, 10], # TODO use batch size
            ),
            # dfxp.BatchNorm_q(
            #     name='softmax-bn',
            #     bits=self.bits,
            #     num_features=10,
            #     training=self.training,
            #     weight_decay=self.weight_decay,
            # ),
        ]


def CIFAR10_Resnet20(bits, dropout=0.5, weight_decay=0, stochastic=False):
    return CIFAR10_Resnet(bits, [3, 3, 3],
        dfxp.ResidualBlock_q, dropout, weight_decay, stochastic)


def CIFAR10_Resnet32(bits, dropout=0.5, weight_decay=0, stochastic=False):
    return CIFAR10_Resnet(bits, [5, 5, 5],
        dfxp.ResidualBlock_q, dropout, weight_decay, stochastic)


def CIFAR10_Resnet44(bits, dropout=0.5, weight_decay=0, stochastic=False):
    return CIFAR10_Resnet(bits, [7, 7, 7],
        dfxp.ResidualBlock_q, dropout, weight_decay, stochastic)


def CIFAR10_Resnet56(bits, dropout=0.5, weight_decay=0, stochastic=False):
    return CIFAR10_Resnet(bits, [9, 9, 9],
        dfxp.ResidualBlock_q, dropout, weight_decay, stochastic)
