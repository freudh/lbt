import tensorflow as tf
import numpy as np


def weight_quantization(X, target_overflow_rate, bits, step, stochastic=False):
    '''
    Quantize input tensor according to the DFXP format.

    When bits == 32, output the original tensor.
    The update_range op is added to the 'update_range' collection.

    `step` is always an integer (could be negative) power of two

    Args:
        X: input tensor
        target_overflow_rate: target overflow rate
        bits: total number of bits for DFXP, including sign bit
        step: step size after quantization
        stochastic: stochastic rounding flag

    Returns:
        Quantized tensor
    '''
    assert 1 <= bits <= 32, 'invalid value for bits: %d' % bits
    if bits == 32:
        return X

    limit = 2.0 ** (bits - 1)

    @tf.custom_gradient
    def identity(X):
        X = tf.round(tf.clip_by_value(X / step, tf.negative(limit), limit-1)) * step
        return X, lambda dy : dy

    @tf.custom_gradient
    def stochastic_identity(X):
        X = tf.floor(tf.clip_by_value(X / step + tf.random_uniform(X.shape[1:], 0, 1),
            tf.negative(limit), limit-1)) * step
        return X, lambda dy : dy

    # TODO remove comment
    # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_step(
    #     X, target_overflow_rate, bits, step))
    if not stochastic:
        return identity(X)
    else:
        return stochastic_identity(X)


def overflow_rate(X, bits, step):
    '''
    Calculate the overflow rate of a tensor according to the DFXP format.

    Args:
        X: input tensor
        bits: total number of for DFXP
        step: step size after quantization

    Returns:
        overflow_rate(X), overflow_rate(2*X)
    '''
    limit = 2.0 ** (bits - 1)
    X = X / step
    mask_X = tf.cast(tf.greater_equal(X, limit), tf.float32) + \
        tf.cast(tf.less(X, tf.negative(limit)), tf.float32)
    mask_2X = tf.cast(tf.greater_equal(X, limit/2), tf.float32) + \
        tf.cast(tf.less(X, tf.negative(limit/2)), tf.float32)
    return tf.reduce_mean(mask_X), tf.reduce_mean(mask_2X)


def update_step(X, target_overflow_rate, bits, step):
    '''
    Step update op for a quantized tensor.

    Args:
        X: input tensor
        target_overflow_rate: target overflow rate
        bits: number of bits in DFXP
        step: step size after quantization

    Returns:
        op for step update
    '''
    overflow_X, overflow_2X = overflow_rate(X, bits, step)
    multiplier = tf.cond(
        overflow_X > target_overflow_rate,
        lambda : 2.0,           # X overflow, need larger step
        lambda : tf.cond(
            overflow_2X <= target_overflow_rate,
            lambda : 0.5,       # 2X does not overflow, need smaller step
            lambda : 1.0,       # step is suitable
        )
    )
    return tf.assign(step, step * multiplier)


class Layer_q:
    '''
    Base class for quantized layers.
    '''
    def forward(self, X):
        '''
        Default forward propagation.
        '''
        self.X = X
        self.y = self.X
        return self.y

    def backward(self, grad, stochastic):
        '''
        Default backward propagation.
        '''
        grad = tf.gradients(self.y, self.X, grad)[0]
        return grad

    def grads_and_vars(self):
        '''
        Default grads_and_vars.
        '''
        return []

    # (m)
    def steps_and_vars(self):
        '''
        Default steps_and_vars.
        '''
        return []        
    # (m)

    def info(self):
        '''
        Returns a one-line description for a quantized layer.
        '''
        return 'quantized layer (default identity)'


class Conv2d_q(Layer_q):
    def __init__(self, name, bits, ksize, strides, padding, use_bias=True, weight_decay=0, target_overflow_rate=0):
        '''
        Quantized 2d convolution.

        Args:
            name: name of the layer
            bits: total number of bits for DFXP
            ksize: kernel size, [h, w, Cin, Cout]
            strides: convolution strides, 4-dimension in NHWC format
            padding: padding scheme, 'SAME' or 'VALID'
            use_bias: whether to use bias
            weight_decay: L2 normalization factor
            target_overflow_rate: target overflow rate
        '''
        h, w, Cin, Cout = self.ksize = ksize
        # self.strides = strides
        self.strides = [strides[0], strides[3], strides[1], strides[2]] # NHWC -> NCHW
        self.padding = padding
        in_units = h * w * Cin
        limit = (6 / in_units) ** 0.5 # He initialization

        self.name = name
        self.use_bias = use_bias

        step = 2.0 ** -5

        with tf.variable_scope(self.name):
            self.W = tf.get_variable('W', initializer=
                tf.random_uniform(ksize, -limit, limit))

            self.W_step = tf.get_variable('W_step', initializer=step, trainable=False)
            self.X_step = tf.get_variable('X_step', initializer=step, trainable=False)
            self.grad_step = tf.get_variable('grad_step', initializer=step, trainable=False)

            if self.use_bias:
                self.b = tf.get_variable('b', [Cout, 1, 1], initializer=tf.zeros_initializer())
                self.b_step = tf.get_variable('b_step', initializer=step, trainable=False)

        self.bits = bits
        self.target_overflow_rate = target_overflow_rate
        self.weight_decay = weight_decay

    def forward(self, X):
        self.X = X

        self.Xq = weight_quantization(self.X, self.target_overflow_rate,
            self.bits, self.X_step)
        self.Wq = weight_quantization(self.W, self.target_overflow_rate,
            self.bits, self.W_step)
        self.y = tf.nn.conv2d(self.Xq, self.Wq, self.strides, self.padding, data_format='NCHW')

        if self.use_bias:
            self.bq = weight_quantization(self.b, self.target_overflow_rate,
                self.bits, self.b_step)
            self.y = self.y + self.bq

        return self.y


    def backward(self, grad, stochastic):
        self.grad = grad
        self.gradq = weight_quantization(grad, self.target_overflow_rate,
            self.bits, self.grad_step, stochastic=stochastic)
        self.dW = tf.gradients(self.y, self.W, self.gradq)[0] + 2 * self.weight_decay * self.W

        if self.use_bias:
            self.db = tf.gradients(self.y, self.b, self.gradq)[0]
        return tf.gradients(self.y, self.X, self.gradq)[0]

    def grads_and_vars(self):
        if self.use_bias:
            return [(self.dW, self.W), (self.db, self.b)]
        else:
            return [(self.dW, self.W)]

    # (m)
    def steps_and_vars(self):
        if self.use_bias:
            return [(self.W_step, self.W), (self.X_step, self.X), (self.grad_step, self.grad), (self.b_step, self.b)]
        else:
            return [(self.W_step, self.W), (self.X_step, self.X), (self.grad_step, self.grad)]
    # (m)

    def info(self):
        return '%d bits conv2d: %dx%dx%d stride %dx%d pad %s weight_decay %f' % (
            self.bits, self.ksize[0], self.ksize[1], self.ksize[3],
            self.strides[2], self.strides[3], self.padding, self.weight_decay)


class Dense_q(Layer_q):
    def __init__(self, name, bits, in_units, units, use_bias=True, weight_decay=0, target_overflow_rate=0):
        '''
        Quantized fully connected layer.

        Args:
            name: name of the layer
            bits: total number of bits for DFXP
            in_units: number of input units
            units: number of output units
            use_bias: whether to use bias
            weight_decay: L2 normalization factor
            target_overflow_rate: target overflow rate
        '''
        limit = (6 / (in_units + units)) ** 0.5
        self.name = name
        self.use_bias = use_bias

        step = 2.0 ** -5

        with tf.variable_scope(self.name):
            self.W = tf.get_variable('W', initializer=
                tf.random_uniform([in_units, units], -limit, limit))

            self.W_step = tf.get_variable('W_step', initializer=step, trainable=False)
            self.X_step = tf.get_variable('X_step', initializer=step, trainable=False)
            self.grad_step = tf.get_variable('grad_step', initializer=step, trainable=False)

            if self.use_bias:
                self.b = tf.get_variable('b', units, initializer=tf.zeros_initializer())
                self.b_step = tf.get_variable('b_step', initializer=step, trainable=False)

        self.bits = bits
        self.target_overflow_rate = target_overflow_rate
        self.weight_decay = weight_decay

    def forward(self, X):
        self.X = X

        self.Xq = weight_quantization(self.X, self.target_overflow_rate,
            self.bits, self.X_step)
        self.Wq = weight_quantization(self.W, self.target_overflow_rate,
            self.bits, self.W_step)
        self.y = tf.matmul(self.Xq, self.Wq)

        if self.use_bias:
            self.bq = weight_quantization(self.b, self.target_overflow_rate,
                self.bits, self.b_step)
            self.y = self.y + self.bq

        return self.y

    def backward(self, grad, stochastic):
        self.grad = grad
        self.gradq = weight_quantization(grad, self.target_overflow_rate,
            self.bits, self.grad_step, stochastic=stochastic)
        self.dW = tf.gradients(self.y, self.W, self.gradq)[0] + 2 * self.weight_decay * self.W

        if self.use_bias:
            self.db = tf.gradients(self.y, self.b, self.gradq)[0]
        return tf.gradients(self.y, self.X, self.gradq)[0]

    def grads_and_vars(self):
        if self.use_bias:
            return [(self.dW, self.W), (self.db, self.b)]
        else:
            return [(self.dW, self.W)]

    # (m)
    def steps_and_vars(self):
        if self.use_bias:
            return [(self.W_step, self.W), (self.X_step, self.X), (self.grad_step, self.grad), (self.b_step, self.b)]
        else:
            return [(self.W_step, self.W), (self.X_step, self.X), (self.grad_step, self.grad)]
    # (m)

    def info(self):
        return '%d bits dense: %dx%d weight_decay %f' % (
            self.bits, self.W.shape[0], self.W.shape[1], self.weight_decay)


class Sequential_q(Layer_q):
    def __init__(self, *args):
        self.layers = args

    def forward(self, X):
        self.X = X
        for layer in self.layers:
            X = layer.forward(X)
        self.y = X
        return self.y

    def backward(self, grad, stochastic):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, stochastic)
        return grad

    def grads_and_vars(self):
        res = []
        for layer in self.layers:
            res += layer.grads_and_vars()
        return res

    # (m)
    def steps_and_vars(self):
        res = []
        for layer in self.layers:
            res += layer.steps_and_vars()
        return res        
    # (m)

    def info(self):
        return '\n\t'.join(['Sequential layer:'] +
            [layer.info() for layer in self.layers])


class Normalization_q(Layer_q):
    def __init__(self, name, bits, num_features, training, momentum=0.99, eps=1e-5, target_overflow_rate=0):
        '''
        Normalization layer in BatchNorm.

        Args:
            name: name of the layer
            bits: total number of bits for DFXP
            num_features: number of input features
            training: training flag
            momentum: running average momentum factor
            eps: divided by sqrt(var+eps)
            target_overflow_rate: target overflow rate
        '''
        self.name = name
        self.train = training

        step = 2.0 ** -5

        with tf.variable_scope(self.name):
            self.X_step = tf.get_variable('X_step', initializer=step, trainable=False)
            self.grad_step = tf.get_variable('grad_step', initializer=step, trainable=False)

            self.X_mean_running = tf.get_variable('X_mean_running', [1, num_features, 1, 1],
                initializer=tf.zeros_initializer())
            self.X_var_running = tf.get_variable('X_var_running', [1, num_features, 1, 1],
                initializer=tf.ones_initializer())

        self.eps = eps
        self.momentum = momentum
        self.bits = bits
        self.target_overflow_rate = target_overflow_rate

    def forward(self, X):
        self.X = X

        self.Xq = weight_quantization(self.X, self.target_overflow_rate,
            self.bits, self.X_step)

        rank = X._rank()
        if rank == 2:
            self.X = tf.expand_dims(self.X, -1)
            self.X = tf.expand_dims(self.X, -1)
        elif rank == 4:
            pass
        else:
            assert False, 'Invalid rank %d' % rank
        self.X_mean_batch, self.X_var_batch = tf.nn.moments(self.Xq, axes=[0, 2, 3], keep_dims=True)

        if self.train:
            self.X_mean = self.X_mean_batch
            self.X_var = self.X_var_batch

            # running average
            def update_op(average, variable, momentum):
                return tf.assign(average, momentum * average + (1-momentum) * variable)
            mean_update_op = update_op(self.X_mean_running, self.X_mean_batch, self.momentum)
            var_update_op = update_op(self.X_var_running, self.X_var_batch, self.momentum)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_update_op)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, var_update_op)
        else:
            self.X_mean = self.X_mean_running
            self.X_var = self.X_var_running

        # TODO: quantize X_mean and X_var?
        self.y = (self.Xq - self.X_mean) / ((self.X_var + self.eps) ** 0.5)
        return self.y

    def backward(self, grad, stochastic):
        self.grad = grad
        self.gradq = weight_quantization(grad, self.target_overflow_rate,
            self.bits, self.grad_step, stochastic=stochastic)
        return tf.gradients(self.y, self.X, self.gradq)[0]
    
    # (m)
    def steps_and_vars(self):
        return [(self.X_step, self.X), (self.grad_step, self.grad)]
    # (m)


class Rescale_q(Layer_q):
    def __init__(self, name, bits, num_features, use_beta=True, weight_decay=0,
        target_overflow_rate=0, gamma_initializer=None):
        '''
        Rescaling layer in BatchNorm.

        Args:
            name: name of the layer
            bits: total number of bits for DFXP
            num_features: number of input features
            use_beta: whether to use beta
            weight_decay: L2 normalization factor
            target_overflow_rate: target overflow rate
            gamma_initializer: initializer for gamma, default ones_initializer
        '''
        self.name = name
        self.use_beta = use_beta

        step = 2 ** -5

        with tf.variable_scope(self.name):
            self.gamma = tf.get_variable('g', [num_features, 1, 1], initializer=
                gamma_initializer or tf.ones_initializer())

            self.g_step = tf.get_variable('g_step', initializer=step, trainable=False)
            self.X_step = tf.get_variable('X_step', initializer=step, trainable=False)
            self.grad_step = tf.get_variable('grad_step', initializer=step, trainable=False)

            if self.use_beta:
                self.beta = tf.get_variable('b', [num_features, 1, 1], initializer=tf.zeros_initializer())
                self.b_step = tf.get_variable('b_step', initializer=step, trainable=False)

        self.bits = bits
        self.target_overflow_rate = target_overflow_rate
        self.weight_decay = weight_decay

    def forward(self, X):
        self.X = X

        rank = X._rank()
        if rank == 2:
            self.X = tf.expand_dims(self.X, -1)
            self.X = tf.expand_dims(self.X, -1)
        elif rank == 4:
            pass
        else:
            assert False, 'Invalid rank %d' % rank

        self.Xq = weight_quantization(self.X, self.target_overflow_rate,
            self.bits, self.X_step)
        self.gq = weight_quantization(self.gamma, self.target_overflow_rate,
            self.bits, self.g_step)
        self.y = self.Xq * self.gq

        if self.use_beta:
            self.bq = weight_quantization(self.beta, self.target_overflow_rate,
                self.bits, self.b_step)
            self.y += self.bq

        return self.y

    def backward(self, grad, stochastic):
        self.grad = grad
        self.gradq = weight_quantization(grad, self.target_overflow_rate,
            self.bits, self.grad_step, stochastic=stochastic)
        self.dgamma = tf.gradients(self.y, self.gamma, self.gradq)[0] + 2 * self.weight_decay * self.gamma
        if self.use_beta:
            self.dbeta = tf.gradients(self.y, self.beta, self.gradq)[0]
        return tf.gradients(self.y, self.X, self.gradq)[0]

    def grads_and_vars(self):
        if self.use_beta:
            return [(self.dgamma, self.gamma), (self.dbeta, self.beta)]
        else:
            return [(self.dgamma, self.gamma)]

    # (m)
    def steps_and_vars(self):
        if self.use_beta:
            return [(self.X_step, self.X), (self.g_step, self.gamma), (self.grad_step, self.grad), (self.b_step, self.beta)]
        else:
            return [(self.X_step, self.X), (self.g_step, self.gamma), (self.grad_step, self.grad)]
    # (m)


class BatchNorm_q(Sequential_q):
    def __init__(self, name, bits, num_features, training, momentum=0.99, eps=1e-5,
        use_beta=True, weight_decay=0, target_overflow_rate=0, gamma_initializer=None):
        '''
        Quantized batch normalization layer.

        Args:
            name: name of the layer
            bits: total number of bits for DFXP
            num_features: number of input features
            training: training flag
            momentum: running average momentum factor
            eps: divided by sqrt(var+eps)
            use_beta: whether to use beta
            weight_decay: L2 normalization factor
            target_overflow_rate: target overflow rate
            gamma_initializer: initializer for gamma
        '''
        self.bits = bits

        super().__init__(
            Normalization_q(
                name=name+'-norm',
                bits=self.bits,
                num_features=num_features,
                training=training,
                momentum=momentum,
                eps=eps,
                target_overflow_rate=target_overflow_rate,
            ),
            Rescale_q(
                name=name+'-rescale',
                bits=self.bits,
                num_features=num_features,
                use_beta=use_beta,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
                gamma_initializer=gamma_initializer,
            )
        )

    def info(self):
        return '%d bits BatchNorm' % self.bits


class ResidualBlock_q(Layer_q):
    expansion = 1

    def __init__(self, name, bits, in_channels, channels, stride, training,
        batch_norm=True, weight_decay=0, decay_bn=True, target_overflow_rate=0):
        '''
        Quantized residual block.

        Args:
            name: name of the layer
            bits: total number of bits for DFXP
            in_channels: number of input channels
            channels: number of output channels
            stride: stride of the first convolution
            training: training flag
            batch_norm: use batch normalization
            weight_decay: L2 normalization factor
            decay_bn: flag for bn weight decay
            target_overflow_rate: target overflow rate
        '''
        self.train = training
        self.bn_weight_decay = weight_decay if decay_bn else 0

        self.residual = Sequential_q(
            Conv2d_q(
                name=name+'-1',
                bits=bits,
                ksize=[3, 3, in_channels, channels],
                strides=[1, stride, stride, 1],
                padding='SAME',
                use_bias=not batch_norm,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
            ),
            BatchNorm_q(
                name=name+'-bn1',
                bits=bits,
                num_features=channels,
                training=self.train,
                weight_decay=self.bn_weight_decay,
                target_overflow_rate=target_overflow_rate,
            ) if batch_norm else Layer_q(),
            ReLU_q(),
            Conv2d_q(
                name=name+'-2',
                bits=bits,
                ksize=[3, 3, channels, channels],
                strides=[1, 1, 1, 1],
                padding='SAME',
                use_bias=not batch_norm,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
            ),
            BatchNorm_q(
                name=name+'-bn2',
                bits=bits,
                num_features=channels,
                training=self.train,
                weight_decay=self.bn_weight_decay,
                target_overflow_rate=target_overflow_rate,
                gamma_initializer=tf.zeros_initializer(), # start at identity
            ) if batch_norm else Layer_q(),
        )

        self._build_shortcut(name, bits, in_channels, channels, stride,
            batch_norm, weight_decay, target_overflow_rate)
        self.relu = ReLU_q()

    def _build_shortcut(self, name, bits, in_channels, channels, stride,
        batch_norm, weight_decay, target_overflow_rate):
        # when in/out dimensions are the same
        if stride == 1 and in_channels == self.expansion * channels:
            self.shortcut = Sequential_q()
        else:
            self.shortcut = Sequential_q(
                Conv2d_q(
                    name=name+'-shortcut',
                    bits=bits,
                    ksize=[1, 1, in_channels, self.expansion * channels],
                    strides=[1, stride, stride, 1],
                    padding='SAME',
                    use_bias=not batch_norm,
                    weight_decay=weight_decay,
                    target_overflow_rate=target_overflow_rate,
                ),
                BatchNorm_q(
                    name=name+'-shortcut-bn',
                    bits=bits,
                    num_features=self.expansion * channels,
                    training=self.train,
                    weight_decay=self.bn_weight_decay,
                    target_overflow_rate=target_overflow_rate,
                ) if batch_norm else Layer_q(),
            )

    def forward(self, X):
        self.X = X
        self.y1 = self.residual.forward(self.X)
        self.y2 = self.shortcut.forward(self.X)
        self.y = self.relu.forward(self.y1 + self.y2)
        return self.y

    def backward(self, grad, stochastic):
        grad = self.relu.backward(grad, stochastic)
        grad1 = self.residual.backward(grad, stochastic)
        grad2 = self.shortcut.backward(grad, stochastic)
        return grad1 + grad2

    def grads_and_vars(self):
        return self.residual.grads_and_vars() + self.shortcut.grads_and_vars()

    def info(self):
        return 'Residual block with ' + self.residual.info()


class ResidualBottleneck_q(ResidualBlock_q):
    expansion = 4

    def __init__(self, name, bits, in_channels, channels, stride, training,
        batch_norm=True, weight_decay=0, decay_bn=True, target_overflow_rate=0):
        '''
        Quantized residual bottleneck.

        Args:
            name: name of the layer
            bits: total number of bits for DFXP
            in_channels: number of input channels
            channels: number of effective channels
            stride: stride of the first convolution
            training: training flag
            batch_norm: use batch normalization
            weight_decay: L2 normalization factor
            decay_bn: flag for bn weight decay
            target_overflow_rate: target overflow rate
        '''
        self.train = training
        self.bn_weight_decay = weight_decay if decay_bn else 0

        out_channels = 4 * channels # expansion
        self.residual = Sequential_q(
            Conv2d_q(
                name=name+'-1',
                bits=bits,
                ksize=[1, 1, in_channels, channels],
                strides=[1, 1, 1, 1],
                padding='SAME',
                use_bias=not batch_norm,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
            ),
            BatchNorm_q(
                name=name+'-bn1',
                bits=bits,
                num_features=channels,
                training=self.train,
                weight_decay=self.bn_weight_decay,
                target_overflow_rate=target_overflow_rate,
            ) if batch_norm else Layer_q(),
            ReLU_q(),
            Conv2d_q(
                name=name+'-2',
                bits=bits,
                ksize=[3, 3, channels, channels],
                strides=[1, stride, stride, 1],
                padding='SAME',
                use_bias=not batch_norm,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
            ),
            BatchNorm_q(
                name=name+'-bn2',
                bits=bits,
                num_features=channels,
                training=self.train,
                weight_decay=self.bn_weight_decay,
                target_overflow_rate=target_overflow_rate,
            ) if batch_norm else Layer_q(),
            ReLU_q(),
            Conv2d_q(
                name=name+'-3',
                bits=bits,
                ksize=[1, 1, channels, out_channels],
                strides=[1, 1, 1, 1],
                padding='SAME',
                use_bias=not batch_norm,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
            ),
            BatchNorm_q(
                name=name+'-bn3',
                bits=bits,
                num_features=out_channels,
                training=self.train,
                weight_decay=self.bn_weight_decay,
                target_overflow_rate=target_overflow_rate,
                gamma_initializer=tf.zeros_initializer(), # start at identity
            ) if batch_norm else Layer_q(),
        )

        self._build_shortcut(name, bits, in_channels, channels, stride,
            batch_norm, weight_decay, target_overflow_rate)
        self.relu = ReLU_q()


class ReLU_q(Layer_q):
    def forward(self, X):
        self.X = X
        self.y = tf.maximum(0.0, self.X)
        return self.y

    def info(self):
        return 'ReLU'


class MaxPool_q(Layer_q):
    def __init__(self, ksize, strides, padding):
        self.ksize = [ksize[0], ksize[3], ksize[1], ksize[2]]
        self.strides = [strides[0], strides[3], strides[1], strides[2]]
        self.padding = padding

    def forward(self, X):
        self.X = X
        self.y = tf.nn.max_pool(self.X, self.ksize, self.strides, self.padding, data_format='NCHW')
        return self.y

    def info(self):
        return 'max pool: %dx%d stride %dx%d' % (
            self.ksize[2], self.ksize[3], self.strides[2], self.strides[3])


class GlobalAvgPool_q(Layer_q):
    def forward(self, X):
        self.X = X
        self.y = tf.reduce_mean(self.X, axis=[2, 3])
        return self.y

    def info(self):
        return 'global avg pool'


class Dropout_q(Layer_q):
    def __init__(self, keep_prob, training):
        self.keep_prob = keep_prob
        self.train = training

    def forward(self, X):
        self.X = X
        if self.train:
            self.y = tf.nn.dropout(self.X, self.keep_prob)
        else:
            self.y = tf.identity(self.X)
        return self.y

    def info(self):
        return 'dropout: %f' % self.keep_prob


class Flatten_q(Layer_q):
    def __init__(self, dim):
        self.dim = dim

    def forward(self, X):
        self.X = X
        self.y = tf.reshape(X, [-1, self.dim])
        return self.y

    def info(self):
        return 'flatten'


def main():
    n = 32
    bits = 8
    A = tf.random_uniform([n,n,n,n], minval=-1, maxval=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    with tf.Session(config=config) as sess:
        for p in range(-15, 3, 1):
            step = tf.Variable(2.0 ** p)
            sess.run(tf.global_variables_initializer())
            B = sess.run(weight_quantization(A, 0, bits, step))
            assert len(set(B.reshape(-1).tolist())) <= 2**bits, 'invalid quantization'
    print('Test finished')


if __name__ == '__main__':
    main()
