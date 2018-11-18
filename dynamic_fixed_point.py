import tensorflow as tf
import numpy as np


def weight_quantization(X, target_overflow_rate, bits, integer_bits, stochastic=False):
    '''
    Quantize input tensor according to the DFXP format.

    When bits == 32, output the original tensor.
    The update_range op is added to the 'update_range' collection.

    Args:
        X: input tensor
        target_overflow_rate: target overflow rate
        bits: total number of bits for DFXP, including sign bit
        integer bits: number of integer bits for DFXP, excluding sign bit
        stochastic: stochastic rounding flag

    Returns:
        Quantized tensor
    '''
    assert 1 <= bits <= 32, 'invalid value for bits: %d' % bits
    if bits == 32:
        return X

    @tf.custom_gradient
    def identity(X):
        multiplier = tf.cast(2 ** (bits - integer_bits - 1), tf.float32)
        limit = tf.cast(2 ** (bits - 1), tf.float32)
        X = tf.round(tf.clip_by_value(X * multiplier, tf.negative(limit), limit-1)) / multiplier
        return X, lambda dy : dy

    @tf.custom_gradient
    def stochastic_identity(X):
        multiplier = tf.cast(2 ** (bits - integer_bits - 1), tf.float32)
        limit = tf.cast(2 ** (bits - 1), tf.float32)
        X = tf.floor(tf.clip_by_value(X * multiplier + tf.random_uniform(X.shape[1:], 0, 1),
            tf.negative(limit), limit-1)) / multiplier
        return X, lambda dy : dy

    tf.add_to_collection('update_range', update_range(
        X, target_overflow_rate, bits, integer_bits))
    if not stochastic:
        return identity(X)
    else:
        return stochastic_identity(X)


def overflow_rate(X, bits, integer_bits):
    '''
    Calculate the overflow rate of a tensor according to the DFXP format.

    Args:
        X: input tensor
        bits: total number of for DFXP
        integer_bits: number of integer bits for DFXP

    Returns:
        overflow_rate(X), overflow_rate(2*X)
    '''
    multiplier = tf.cast(2 ** (bits - integer_bits - 1), tf.float32)
    limit = tf.cast(2 ** (bits - 1), tf.float32)
    X = X * multiplier
    mask_X = tf.cast(tf.greater_equal(X, limit), tf.float32) + \
        tf.cast(tf.less(X, tf.negative(limit)), tf.float32)
    mask_2X = tf.cast(tf.greater_equal(X, limit/2), tf.float32) + \
        tf.cast(tf.less(X, tf.negative(limit/2)), tf.float32)
    return tf.reduce_mean(mask_X), tf.reduce_mean(mask_2X)


def update_range(X, target_overflow_rate, bits, integer_bits):
    '''
    Range update op for a quantized tensor.

    Args:
        X: input tensor
        target_overflow_rate: target overflow rate
        bits: number of bits in DFXP
        integer_bits: current number of integer bits in DFXP

    Returns:
        op for range update
    '''
    overflow_X, overflow_2X = overflow_rate(X, bits, integer_bits)
    delta = tf.cond(
        overflow_X > target_overflow_rate,
        lambda : 1,
        lambda : tf.cond(
            overflow_2X <= target_overflow_rate,
            lambda : -1,
            lambda : 0
        )
    )
    # prevent updated integer_bits being too large
    return tf.assign(integer_bits, tf.minimum(bits-1, integer_bits + delta))
    # return tf.assign( integer_bits, tf.clip_by_value(tf.minimum(bits-1, integer_bits + delta), 1-bits, bits-1) )


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

    def info(self):
        '''
        Returns a one-line description for a quantized layer.
        '''
        return 'quantized layer (default identity)'


class Conv2d_q(Layer_q):
    def __init__(self, name, bits, ksize, strides, padding, use_bias=True, weight_decay=0,
        target_overflow_rate=0, input_range=2, weight_range=2, bias_range=2, grad_range=0):
        '''
        Quantized 2d convolution.

        Args:
            name: name of the layer
            bits: total number of bits for DFXP
            ksize: kernel size, [h, w, Cin, Cout]
            strides: convolution strides, 4-dimension as input_size
            padding: padding scheme, 'SAME' or 'VALID'
            use_bias: whether to use bias
            weight_decay: L2 normalization factor
            target_overflow_rate: target overflow rate
            input_range: initial DFXP range for inputs
            weight_range: initial DFXP range for weights
            bias_range: initial DFXP range for bias
            grad_range: initial DFXP range for backward gradients
        '''
        h, w, Cin, Cout = self.ksize = ksize
        self.strides = strides
        self.padding = padding
        in_units = h * w * Cin
        limit = (3 / in_units) ** 0.5

        self.name = name
        self.use_bias = use_bias

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.W = tf.Variable(tf.random_uniform(ksize, -limit, limit))

            self.W_range = tf.get_variable('W_range', dtype=tf.int32,
                initializer=weight_range, trainable=False)
            self.X_range = tf.get_variable('X_range', dtype=tf.int32,
                initializer=input_range, trainable=False)
            self.grad_range = tf.get_variable('grad_range', dtype=tf.int32,
                initializer=grad_range, trainable=False)

            if self.use_bias:
                self.b = tf.get_variable('b', Cout, initializer=tf.zeros_initializer())
                self.b_range = tf.get_variable('b_range', dtype=tf.int32,
                    initializer=bias_range, trainable=False)

        self.bits = bits
        self.target_overflow_rate = target_overflow_rate
        self.weight_decay = weight_decay

        self.init_flag = True
        self.rem_flag = False
        self.init_f = True

    def forward(self, X):
        self.X = X

        with tf.name_scope(self.name):
            tf.summary.scalar('W_range', self.W_range)
            tf.summary.scalar('X_range', self.X_range)
            tf.summary.scalar('grad_range', self.grad_range)

            tf.summary.scalar('W_mean', tf.reduce_mean(self.W))
            tf.summary.scalar('X_mean', tf.reduce_mean(self.X))

            if self.use_bias:
                tf.summary.scalar('b_range', self.b_range)
                tf.summary.scalar('b_mean', tf.reduce_mean(self.b))

        self.Xq = weight_quantization(self.X, self.target_overflow_rate,
            self.bits, self.X_range)
        self.Wq = weight_quantization(self.W, self.target_overflow_rate,
            self.bits, self.W_range)
        self.y = tf.nn.conv2d(self.Xq, self.Wq, self.strides, self.padding)

        if self.use_bias:
            self.bq = weight_quantization(self.b, self.target_overflow_rate,
                self.bits, self.b_range)
            self.y = self.y + self.bq

        return self.y


    def pre_conv_func(self):
        out = tf.py_func(self._pre_conv_func,
                [self.grad_avg, self.grad, self.eps, self.accu_value, self.reminder],
                tf.float32
            )

        return out

    def _pre_conv_func(self, grad_avg_np, grad_np, eps_np, accu_value_np, reminder_np):
        if self.init_flag == True:
            # print("------ Gotya ------")
            if eps_np > grad_avg_np:
                # print("------ Aha ------")
                self.init_flag = False
                if self.rem_flag == True:
                    accu_value_np = reminder_np.copy() + grad_np.copy()
                else:
                    accu_value_np = grad_np.copy()
                self.accu_value = tf.convert_to_tensor(accu_value_np, dtype=tf.float32)
        
        else:
            # print("--- Small value collected ---")
            accu_value_np += grad_np.copy()
            self.accu_value = tf.convert_to_tensor(accu_value_np, dtype=tf.float32)

            if (np.mean(np.absolute(accu_value_np)) > eps_np):
                self.init_flag = True
                if (np.mean(accu_value_np) > 0):
                    reminder_np = accu_value_np.copy() - (accu_value_np.copy() // eps_np.copy()) * eps_np.copy()
                else:
                    reminder_np = accu_value_np.copy() + ((-accu_value_np.copy()) // eps_np.copy()) * eps_np.copy()
                self.reminder = tf.convert_to_tensor(reminder_np, dtype=tf.float32)

                self.rem_flag = True
                grad_np = accu_value_np.copy()
                self.grad = tf.convert_to_tensor(grad_np, dtype=tf.float32)

        return grad_np


    def backward(self, grad, stochastic):
        global pre_conv_op
        
        self.grad = grad
        self.eps =  tf.cast( 1/ (2 ** (self.bits - self.grad_range)), tf.float32)    # the smallest value
        # compute the avg value
        self.grad_avg = tf.reduce_mean(tf.abs(self.grad))

        if self.init_f == True:
            self.init_f = False
            self.reminder = self.grad   # buffer
            self.accu_value = self.grad + 0.001 # buffer

        pre_conv_op = self.pre_conv_func()    # (m)

        self.gradq = weight_quantization(self.grad, self.target_overflow_rate,
            self.bits, self.grad_range, stochastic=stochastic)
        self.dW = tf.gradients(self.y, self.W, self.gradq)[0] + 2 * self.weight_decay * self.W
        if self.use_bias:
            self.db = tf.gradients(self.y, self.b, self.gradq)[0]
        return tf.gradients(self.y, self.X, self.gradq)[0]

    def grads_and_vars(self):
        if self.use_bias:
            return [(self.dW, self.W), (self.db, self.b)]
        else:
            return [(self.dW, self.W)]

    def info(self):
        return '%d bits conv2d: %dx%dx%d stride %dx%d pad %s weight_decay %f' % (
            self.bits, self.ksize[0], self.ksize[1], self.ksize[3],
            self.strides[1], self.strides[2], self.padding, self.weight_decay)


class Dense_q(Layer_q):
    def __init__(self, name, bits, in_units, units, use_bias=True, weight_decay=0,
        target_overflow_rate=0, input_range=2, weight_range=2, bias_range=2, grad_range=0):
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
            input_range: initial DFXP range for inputs
            weight_range: initial DFXP range for weights
            bias_range: initial DFXP range for bias
            grad_range: initial DFXP range for backward gradients
        '''
        limit = (6 / (in_units + units)) ** 0.5
        self.name = name
        self.use_bias = use_bias

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.W = tf.Variable(tf.random_uniform([in_units, units], -limit, limit))

            self.W_range = tf.get_variable('W_range', dtype=tf.int32,
                initializer=weight_range, trainable=False)
            self.X_range = tf.get_variable('X_range', dtype=tf.int32,
                initializer=input_range, trainable=False)
            self.grad_range = tf.get_variable('grad_range', dtype=tf.int32,
                initializer=grad_range, trainable=False)

            if self.use_bias:
                self.b = tf.get_variable('b', units, initializer=tf.zeros_initializer())
                self.b_range = tf.get_variable('b_range', dtype=tf.int32,
                    initializer=bias_range, trainable=False)

        self.bits = bits
        self.target_overflow_rate = target_overflow_rate
        self.weight_decay = weight_decay

        self.init_flag = np.ones([32, 10], dtype=int)   # special
        self.rem_flag = np.zeros([32, 10], dtype=int)   # special
        
        self.init_f = True

    def forward(self, X):
        self.X = X

        with tf.name_scope(self.name):
            tf.summary.scalar('W_range', self.W_range)
            tf.summary.scalar('X_range', self.X_range)
            tf.summary.scalar('grad_range', self.grad_range)

            tf.summary.scalar('W_mean', tf.reduce_mean(self.W))
            tf.summary.scalar('X_mean', tf.reduce_mean(self.X))

            if self.use_bias:
                tf.summary.scalar('b_range', self.b_range)
                tf.summary.scalar('b_mean', tf.reduce_mean(self.b))

        self.Xq = weight_quantization(self.X, self.target_overflow_rate,
            self.bits, self.X_range)
        self.Wq = weight_quantization(self.W, self.target_overflow_rate,
            self.bits, self.W_range)
        self.y = tf.matmul(self.Xq, self.Wq)

        if self.use_bias:
            self.bq = weight_quantization(self.b, self.target_overflow_rate,
                self.bits, self.b_range)
            self.y = self.y + self.bq

        return self.y


    def pre_dense_func(self):
        out = tf.py_func(self._pre_dense_func,
                [self.grad, self.eps, self.accu_value, self.reminder, self.grad_range],
                tf.float32
            )

        return out

    def _pre_dense_func(self, grad_np, eps_np, accu_value_np, reminder_np, grad_range_np):      
        dim1 = np.shape(grad_np)[0]
        dim2 = np.shape(grad_np)[1]
        for i in range(dim1):
            for j in range(dim2):
                if self.init_flag[i,j] == 1:
                    # print("------ Gotya ------")
                    if eps_np > np.absolute(grad_np)[i,j]:
                        # print("------ Aha ------")
                        self.init_flag[i,j] = 0
                        # print("rem_flag[i,j] " + str(self.rem_flag[i,j]))
                        if self.rem_flag[i,j] == 1:
                            accu_value_np[i,j] = reminder_np.copy()[i,j] + grad_np.copy()[i,j]
                        else:
                            accu_value_np[i,j] = grad_np.copy()[i,j]
                        self.accu_value = tf.convert_to_tensor(accu_value_np, dtype=tf.float32)
                
                else:
                    # print("--- Small value collected ---")
                    accu_value_np[i,j] += grad_np.copy()[i,j]
                    self.accu_value = tf.convert_to_tensor(accu_value_np, dtype=tf.float32)

                    if (np.absolute(accu_value_np)[i,j] > eps_np):
                        self.init_flag[i,j] = 1
                        # print("abs_accu_value_1 [i,j]" + str(np.absolute(accu_value_np)[i,j]))
                        if (accu_value_np[i,j] > 0):
                            reminder_np[i,j] = accu_value_np.copy()[i,j] - (accu_value_np.copy()[i,j] // eps_np.copy()) * eps_np.copy()
                        else:
                            reminder_np[i,j] = accu_value_np.copy()[i,j] + ((-accu_value_np.copy()[i,j]) // eps_np.copy()) * eps_np.copy()

                        self.reminder = tf.convert_to_tensor(reminder_np, dtype=tf.float32)
                        # print("abs_accu_value_2 [i,j]" + str(np.absolute(accu_value_np)[i,j]))
                        self.rem_flag[i,j] = 1
                        grad_np[i,j] = accu_value_np.copy()[i,j]
                        self.grad = tf.convert_to_tensor(grad_np, dtype=tf.float32)
                        # self.grad = tf.print(self.grad)
                        # print("eps_np " + str(eps_np))
                        # print("reminder_np[i,j] " + str(reminder_np[i,j]))

        return grad_np


    def backward(self, grad, stochastic):
        global pre_dense_op     
        self.grad = grad

        self.eps =  tf.cast( 1/ (2 ** (self.bits - self.grad_range)), tf.float32)    # the smallest value
        # # compute the avg value
        # self.grad_avg = tf.reduce_mean(tf.abs(self.grad))
        if self.init_f == True:
            self.init_f = False
            self.reminder = self.grad # buffer
            self.accu_value =  self.grad + 0.001 # buffer            
            # self.reminder = tf.constant(0.0, shape=[32,10], dtype=tf.float32) # buffer
            # self.accu_value =  tf.constant(0.001, shape=[32,10], dtype=tf.float32) # buffer

        pre_dense_op = self.pre_dense_func()

        self.gradq = weight_quantization(self.grad, self.target_overflow_rate,
            self.bits, self.grad_range, stochastic=stochastic)
        self.dW = tf.gradients(self.y, self.W, self.gradq)[0] + 2 * self.weight_decay * self.W
        if self.use_bias:
            self.db = tf.gradients(self.y, self.b, self.gradq)[0]
        return tf.gradients(self.y, self.X, self.gradq)[0]

    def grads_and_vars(self):
        if self.use_bias:
            return [(self.dW, self.W), (self.db, self.b)]
        else:
            return [(self.dW, self.W)]

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

    def info(self):
        return '\n\t'.join(['Sequential layer:'] +
            [layer.info() for layer in self.layers])


class Normalization_q(Layer_q):
    def __init__(self, name, bits, num_features, training, momentum=0.999, eps=1e-5, target_overflow_rate=0,
        input_range=2, grad_range=2):
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
            input_range: initial DFXP range for inputs
            grad_range: initial DFXP range for backward gradients
        '''
        self.name = name
        self.train = training

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.X_range = tf.get_variable('X_range', dtype=tf.int32,
                initializer=input_range, trainable=False)
            self.grad_range = tf.get_variable('grad_range', dtype=tf.int32,
                initializer=grad_range, trainable=False)

            self.X_mean_running = tf.get_variable('X_mean_running', num_features,
                initializer=tf.zeros_initializer())
            self.X_var_running = tf.get_variable('X_var_running', num_features,
                initializer=tf.ones_initializer())

        self.eps = eps
        self.momentum = momentum
        self.bits = bits
        self.target_overflow_rate = target_overflow_rate

    def forward(self, X):
        self.X = X

        with tf.name_scope(self.name):
            tf.summary.scalar('X_range', self.X_range)
            tf.summary.scalar('grad_range', self.grad_range)

            tf.summary.scalar('X_mean', tf.reduce_mean(self.X))

        self.Xq = weight_quantization(self.X, self.target_overflow_rate,
            self.bits, self.X_range)

        rank = X._rank()
        self.X_mean_batch, self.X_var_batch = tf.nn.moments(self.Xq, axes=list(range(rank-1)))

        self.X_mean = tf.cond(
            self.train,
            lambda : self.X_mean_batch,
            lambda : self.X_mean_running,
        )
        self.X_var = tf.cond(
            self.train,
            lambda : self.X_var_batch,
            lambda : self.X_var_running,
        )

        # running average
        def update_op(average, variable, momentum):
            return tf.assign(
                average,
                tf.cond(
                    self.train,
                    lambda : momentum * average + (1-momentum) * variable,
                    lambda : average
                )
            )
        self.mean_update_op = update_op(self.X_mean_running, self.X_mean_batch, self.momentum)
        self.var_update_op = update_op(self.X_var_running, self.X_var_batch, self.momentum)

        with tf.control_dependencies([self.mean_update_op, self.var_update_op]):
            # TODO: quantize X_mean and X_var?
            self.y = (self.Xq - self.X_mean) / ((self.X_var + self.eps) ** 0.5)

        return self.y

    def backward(self, grad, stochastic):
        self.gradq = weight_quantization(grad, self.target_overflow_rate,
            self.bits, self.grad_range, stochastic=stochastic)
        return tf.gradients(self.y, self.X, self.gradq)[0]


class Rescale_q(Layer_q):
    def __init__(self, name, bits, num_features, weight_decay=0, target_overflow_rate=0,
        input_range=2, gamma_range=2, beta_range=2, grad_range=0):
        '''
        Rescaling layer in BatchNorm.

        Args:
            name: name of the layer
            bits: total number of bits for DFXP
            num_features: number of input features
            weight_decay: L2 normalization factor
            target_overflow_rate: target overflow rate
            input_range: initial DFXP range for inputs
            gamma_range: initial DFXP range for gamma
            beta_range: initial DFXP range for beta
            grad_range: initial DFXP range for backward gradients
        '''
        self.name = name

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.gamma = tf.get_variable('g', num_features,
                initializer=tf.ones_initializer())
            self.beta = tf.get_variable('b', num_features,
                initializer=tf.zeros_initializer())

            self.g_range = tf.get_variable('g_range', dtype=tf.int32,
                initializer=gamma_range, trainable=False)
            self.b_range = tf.get_variable('b_range', dtype=tf.int32,
                initializer=beta_range, trainable=False)
            self.X_range = tf.get_variable('X_range', dtype=tf.int32,
                initializer=input_range, trainable=False)
            self.grad_range = tf.get_variable('grad_range', dtype=tf.int32,
                initializer=grad_range, trainable=False)

        self.bits = bits
        self.target_overflow_rate = target_overflow_rate
        self.weight_decay = weight_decay

        self.init_flag = True
        self.rem_flag = False
        self.init_f = True

    def forward(self, X):
        self.X = X

        with tf.name_scope(self.name):
            tf.summary.scalar('g_range', self.g_range)
            tf.summary.scalar('b_range', self.b_range)
            tf.summary.scalar('X_range', self.X_range)
            tf.summary.scalar('grad_range', self.grad_range)

            tf.summary.scalar('g_mean', tf.reduce_mean(self.gamma))
            tf.summary.scalar('b_mean', tf.reduce_mean(self.beta))
            tf.summary.scalar('X_mean', tf.reduce_mean(self.X))

        self.Xq = weight_quantization(self.X, self.target_overflow_rate,
            self.bits, self.X_range)
        self.gq = weight_quantization(self.gamma, self.target_overflow_rate,
            self.bits, self.g_range)
        self.bq = weight_quantization(self.beta, self.target_overflow_rate,
            self.bits, self.b_range)
        self.y = self.Xq * self.gq + self.bq
        return self.y


    def pre_rescale_func(self):
        out = tf.py_func(self._pre_rescale_func,
                [self.grad_avg, self.grad, self.eps, self.accu_value, self.reminder],
                tf.float32
            )

        return out

    def _pre_rescale_func(self, grad_avg_np, grad_np, eps_np, accu_value_np, reminder_np):
        if self.init_flag == True:
            # print("------ Gotya ------")
            if eps_np > grad_avg_np:
                # print("------ Aha ------")
                self.init_flag = False
                if self.rem_flag == True:
                    accu_value_np = reminder_np.copy() + grad_np.copy()
                else:
                    accu_value_np = grad_np.copy()
                self.accu_value = tf.convert_to_tensor(accu_value_np, dtype=tf.float32)
        
        else:
            # print("--- Small value collected ---")
            accu_value_np += grad_np.copy()
            self.accu_value = tf.convert_to_tensor(accu_value_np, dtype=tf.float32)
            if (np.mean(np.absolute(accu_value_np)) > eps_np):
                self.init_flag = True
                if (np.mean(accu_value_np) > 0):
                    reminder_np = accu_value_np.copy() - (accu_value_np.copy() // eps_np.copy()) * eps_np.copy()
                else:
                    reminder_np = accu_value_np.copy() + ((-accu_value_np.copy()) // eps_np.copy()) * eps_np.copy()
                self.reminder = tf.convert_to_tensor(reminder_np, dtype=tf.float32)

                self.rem_flag = True
                grad_np = accu_value_np.copy()
                self.grad = tf.convert_to_tensor(grad_np, dtype=tf.float32)

        return grad_np


    def backward(self, grad, stochastic):
        global pre_rescale_op     
        self.grad = grad

        self.eps =  tf.cast( 1/ (2 ** (self.bits - self.grad_range)), tf.float32)    # the smallest value
        # compute the avg value
        self.grad_avg = tf.reduce_mean(tf.abs(self.grad))
        if self.init_f == True:
            self.init_f = False
            self.reminder = self.grad   # buffer
            self.accu_value = self.grad + 0.001 # buffer

        pre_rescale_op = self.pre_rescale_func()  # (m)

        self.gradq = weight_quantization(self.grad, self.target_overflow_rate,
                                         self.bits, self.grad_range, stochastic=stochastic)
        self.dgamma = tf.gradients(self.y, self.gamma, self.gradq)[0] + 2 * self.weight_decay * self.gamma
        self.dbeta = tf.gradients(self.y, self.beta, self.gradq)[0]
        return tf.gradients(self.y, self.X, self.gradq)[0]

    def grads_and_vars(self):
        return [(self.dgamma, self.gamma), (self.dbeta, self.beta)]


class BatchNorm_q(Sequential_q):
    def __init__(self, name, bits, num_features, training, momentum=0.999, eps=1e-5, weight_decay=0,
        target_overflow_rate=0, input_range=2, gamma_range=2, beta_range=2, grad_range=2):
        '''
        Quantized batch normalization layer.

        Args:
            name: name of the layer
            bits: total number of bits for DFXP
            num_features: number of input features
            training: training flag
            momentum: running average momentum factor
            eps: divided by sqrt(var+eps)
            weight_decay: L2 normalization factor
            target_overflow_rate: target overflow rate
            input_range: initial DFXP range for inputs
            gamma_range: initial DFXP range for gamma
            beta_range: initial DFXP range for beta
            grad_range: initial DFXP range for backward gradients
        '''
        super().__init__(
            Normalization_q(
                name=name+'-norm',
                bits=bits,
                num_features=num_features,
                training=training,
                momentum=momentum,
                eps=eps,
                target_overflow_rate=target_overflow_rate,
                input_range=input_range,
                grad_range=grad_range
            ),
            Rescale_q(
                name=name+'-rescale',
                bits=bits,
                num_features=num_features,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
                input_range=2,
                gamma_range=gamma_range,
                beta_range=beta_range,
                grad_range=grad_range
            )
        )

    def info(self):
        return 'BatchNorm'


class ResidualBlock_q(Layer_q):
    expansion = 1

    def __init__(self, name, bits, in_channels, channels, stride, training, batch_norm=True, weight_decay=0,
        target_overflow_rate=0, input_range=2, weight_range=2, bias_range=2, grad_range=2):
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
            target_overflow_rate: target overflow rate
            input_range: initial DFXP range for inputs
            weight_range: initial DFXP range for weights
            bias_range: initial DFXP range for bias
            grad_range: initial DFXP range for backward gradients
        '''
        self.train = training

        self.residual = Sequential_q(
            Conv2d_q(
                name=name+'-1',
                bits=bits,
                ksize=[3, 3, in_channels, channels],
                strides=[1, stride, stride, 1],
                padding='SAME',
                use_bias=not batch_norm,
                weight_decay=weight_decay,
                input_range=input_range,
                weight_range=weight_range,
                bias_range=bias_range,
                grad_range=grad_range,
            ),
            BatchNorm_q(
                name=name+'-bn1',
                bits=bits,
                num_features=channels,
                training=self.train,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
                input_range=input_range,
                grad_range=grad_range,
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
                input_range=input_range,
                weight_range=weight_range,
                bias_range=bias_range,
                grad_range=grad_range,
            ),
            BatchNorm_q(
                name=name+'-bn2',
                bits=bits,
                num_features=channels,
                training=self.train,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
                input_range=input_range,
                grad_range=grad_range,
            ) if batch_norm else Layer_q(),
        )

        self._build_shortcut(name, bits, in_channels, channels, stride, batch_norm, weight_decay,
            target_overflow_rate, input_range, weight_range, bias_range, grad_range)
        self.relu = ReLU_q()

    def _build_shortcut(self, name, bits, in_channels, channels, stride, batch_norm, weight_decay,
        target_overflow_rate, input_range, weight_range, bias_range, grad_range):
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
                    input_range=input_range,
                    weight_range=weight_range,
                    bias_range=bias_range,
                    grad_range=grad_range,
                ),
                BatchNorm_q(
                    name=name+'-shortcut-bn',
                    bits=bits,
                    num_features=self.expansion * channels,
                    training=self.train,
                    weight_decay=weight_decay,
                    target_overflow_rate=target_overflow_rate,
                    input_range=input_range,
                    grad_range=grad_range,
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

    def __init__(self, name, bits, in_channels, channels, stride, training, batch_norm=True, weight_decay=0,
        target_overflow_rate=0, input_range=2, weight_range=2, bias_range=2, grad_range=2):
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
            target_overflow_rate: target overflow rate
            input_range: initial DFXP range for inputs
            weight_range: initial DFXP range for weights
            bias_range: initial DFXP range for bias
            grad_range: initial DFXP range for backward gradients
        '''
        self.train = training

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
                input_range=input_range,
                weight_range=weight_range,
                bias_range=bias_range,
                grad_range=grad_range,
            ),
            BatchNorm_q(
                name=name+'-bn1',
                bits=bits,
                num_features=channels,
                training=self.train,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
                input_range=input_range,
                grad_range=grad_range,
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
                input_range=input_range,
                weight_range=weight_range,
                bias_range=bias_range,
                grad_range=grad_range,
            ),
            BatchNorm_q(
                name=name+'-bn2',
                bits=bits,
                num_features=channels,
                training=self.train,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
                input_range=input_range,
                grad_range=grad_range,
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
                input_range=input_range,
                weight_range=weight_range,
                bias_range=bias_range,
                grad_range=grad_range,
            ),
            BatchNorm_q(
                name=name+'-bn3',
                bits=bits,
                num_features=out_channels,
                training=self.train,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
                input_range=input_range,
                grad_range=grad_range,
            ) if batch_norm else Layer_q(),
        )

        self._build_shortcut(name, bits, in_channels, channels, stride, batch_norm,
            weight_decay, target_overflow_rate, input_range, weight_range, bias_range, grad_range)
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
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def forward(self, X):
        self.X = X
        self.y = tf.nn.max_pool(self.X, self.ksize, self.strides, self.padding)
        return self.y

    def info(self):
        return 'max pool: %dx%d stride %dx%d' % (
            self.ksize[1], self.ksize[2], self.strides[1], self.strides[2])


class AvgPool_q(Layer_q):
    def __init__(self, ksize, strides, padding):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def forward(self, X):
        self.X = X
        self.y = tf.nn.avg_pool(self.X, self.ksize, self.strides, self.padding)
        return self.y

    def info(self):
        return 'avg pool: %dx%d stride %dx%d' % (
            self.ksize[1], self.ksize[2], self.strides[1], self.strides[2])


class Dropout_q(Layer_q):
    def __init__(self, keep_prob, training):
        self.keep_prob = keep_prob
        self.train = training

    def forward(self, X):
        self.X = X
        self.y = tf.cond(
            self.train, 
            lambda : tf.nn.dropout(self.X, self.keep_prob),
            lambda : self.X
        )
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
    '''Tester.'''
    x = tf.placeholder(tf.float32, [None, 3, 3, 1])
    conv = Conv2d_q(8, [2, 2, 1, 1], [1, 1, 1, 1], 'VALID', 0.01, 5, 2, 2, 5)
    y = conv.forward(x)
    loss = tf.reduce_sum(y ** 2)
    dy = tf.gradients(loss, y)[0]
    conv.backward(dy)
    update_range_op = tf.get_collection('update_range')
    optimizer = tf.train.MomentumOptimizer(2e-3, 0.9)
    grads_and_vars = conv.grads_and_vars()
    train_op = optimizer.apply_gradients(grads_and_vars)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        a = np.arange(9).reshape([1, 3, 3, 1])

        for _ in range(300):
            _, loss_f, _, _ = sess.run([train_op, loss, conv.W, update_range_op], feed_dict={x: a})
            print('loss_f: %f' % loss_f)


if __name__ == '__main__':
    main()
