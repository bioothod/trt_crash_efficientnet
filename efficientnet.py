import collections
import math
import re
import six

import numpy as np
import tensorflow as tf

from six.moves import xrange  # pylint: disable=redefined-builtin

import logging
logger = logging.getLogger('gclass')

batchnorm = tf.keras.layers.BatchNormalization
#batchnorm = utils.TpuBatchNormalization  # TPU-specific requirement.
#relu_fn = tf.nn.swish

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'relu_fn',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type',
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

def drop_connect(inputs, is_training, drop_connect_rate):
    """Apply drop connect."""

    def _drop_connect():
        # Compute keep_prob
        # TODO(tanmingxing): add support for training progress.
        keep_prob = 1.0 - drop_connect_rate

        # Compute drop_connect tensor
        batch_size = tf.shape(inputs)[0]
        random_tensor = keep_prob
        random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.math.divide(inputs, keep_prob) * binary_tensor
        return output
    def _no_drop_connect():
        return inputs

    training = tf.compat.v2.get_static_value(is_training)
    if not training:
        return inputs

    return _drop_connect()
    return tf.cond(is_training, lambda: _drop_connect(), lambda: _no_drop_connect())

def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.
    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    Returns:
      an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for dense kernels.
    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.
    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    Returns:
      an initialization for the variable
    """
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    tf.compat.v1.logging.debug('round_filter input={} output={}'.format(orig_f, new_filters))
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class MBConvBlock(tf.keras.layers.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.
    Attributes:
      endpoints: dict. A list of internal tensors.
    """

    def __init__(self, block_args, global_params):
        """Initializes a MBConv block.
        Args:
          block_args: BlockArgs, arguments to create a Block.
          global_params: GlobalParams, a set of global parameters.
        """
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._data_format = global_params.data_format
        if self._data_format == 'channels_first':
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

        self._relu_fn = global_params.relu_fn or tf.nn.swish
        self._has_se = (self._block_args.se_ratio is not None) and (self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)

        self.endpoints = None

        # Builds the block accordings to arguments.
        self._build()

    def block_args(self):
        return self._block_args

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=False)
            self._bn0 = batchnorm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon)

        kernel_size = self._block_args.kernel_size
        # Depth-wise convolution phase:
        self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=self._block_args.strides,
            depthwise_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False)
        self._bn1 = batchnorm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

        if self._has_se:
            num_reduced_filters = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # Squeeze and Excitation layer.
            self._se_reduce = tf.keras.layers.Conv2D(
                num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=True)
            self._se_expand = tf.keras.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=True)

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False)
        self._bn2 = batchnorm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

    def _call_se(self, input_tensor):
        """Call Squeeze and Excitation layer.
        Args:
          input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.
        Returns:
          A output tensor, which should have the same shape as input.
        """
        se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keepdims=True)
        se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
        #logger.info('%s: Built Squeeze and Excitation with tensor shape: %s' % (self.name, se_tensor.shape))
        return tf.sigmoid(se_tensor) * input_tensor

    def call(self, inputs, training=True, drop_connect_rate=None):
        """Implementation of call().
        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
          drop_connect_rate: float, between 0 to 1, drop connect rate.
        Returns:
          A output tensor.
        """
        #tf.compat.v1.logging.debug('Block input: %s shape: %s' % (inputs.name, inputs.shape))
        if self._block_args.expand_ratio != 1:
            with tf.name_scope('expand'):
                x = self._relu_fn(self._bn0(self._expand_conv(inputs), training=training))
        else:
            with tf.name_scope('plain'):
                x = inputs
        #logger.info('%s: Expand: %s shape: %s' % (self.name, x.name, x.shape))

        with tf.name_scope('bn1'):
            x = self._relu_fn(self._bn1(self._depthwise_conv(x), training=training))
        #logger.info('%s: DWConv: %s shape: %s' % (self.name, x.name, x.shape))

        if self._has_se:
            with tf.name_scope('se'):
                x = self._call_se(x)

        self.endpoints = {'expansion_output': x}

        with tf.name_scope('bn2'):
            x = self._bn2(self._project_conv(x), training=training)

        with tf.name_scope('skip'):
            if self._block_args.id_skip:
                if all(s == 1 for s in self._block_args.strides) and self._block_args.input_filters == self._block_args.output_filters:
                    # only apply drop_connect if skip presents.
                    if drop_connect_rate:
                        x = drop_connect(x, training, drop_connect_rate)
                    x = tf.add(x, inputs)
        #logger.info('%s: Project: %s shape: %s, trainable variables: %d' % (self.name, x.name, x.shape, len(self.trainable_variables)))
        return x


class MBConvBlockWithoutDepthwise(MBConvBlock):
    """MBConv-like block without depthwise convolution and squeeze-and-excite."""

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False)
            self._bn0 = batchnorm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon)

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=self._block_args.strides,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = batchnorm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

    def call(self, inputs, training=True, drop_connect_rate=None):
        """Implementation of call().
        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
          drop_connect_rate: float, between 0 to 1, drop connect rate.
        Returns:
          A output tensor.
        """
        tf.compat.v1.logging.debug('Block input: %s shape: %s' % (inputs.name, inputs.shape))
        if self._block_args.expand_ratio != 1:
            x = self._relu_fn(self._bn0(self._expand_conv(inputs), training=training))
        else:
            x = inputs
        tf.compat.v1.logging.debug('Expand: %s shape: %s' % (x.name, x.shape))

        self.endpoints = {'expansion_output': x}

        x = self._bn1(self._project_conv(x), training=training)
        if self._block_args.id_skip:
            if all(s == 1 for s in self._block_args.strides) and self._block_args.input_filters == self._block_args.output_filters:
                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = drop_connect(x, training, drop_connect_rate)
                x = tf.add(x, inputs)
        tf.compat.v1.logging.debug('Project: %s shape: %s, trainable variables: %d' % (x.name, x.shape, len(self.trainable_variables)))
        return x


class Model(tf.keras.models.Model):
    """A class implements tf.keras.Model for MNAS-like model.
      Reference: https://arxiv.org/abs/1807.11626
    """

    def __init__(self, blocks_args=None, global_params=None):
        """Initializes an `Model` instance.
        Args:
          blocks_args: A list of BlockArgs to construct block modules.
          global_params: GlobalParams, a set of global parameters.
        Raises:
          ValueError: when blocks_args is not specified as a list.
        """
        super(Model, self).__init__()
        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._relu_fn = global_params.relu_fn or tf.nn.swish

        self.endpoints = None

        self._build()

    def _get_conv_block(self, conv_type):
        conv_block_map = {0: MBConvBlock, 1: MBConvBlockWithoutDepthwise}
        return conv_block_map[conv_type]

    def _build(self):
        """Builds a model."""
        self._blocks = []
        # Builds blocks.
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params))

            # The first block needs to take care of stride and filter size increase.
            conv_block = self._get_conv_block(block_args.conv_type)
            self._blocks.append(conv_block(block_args, self._global_params))
            if block_args.num_repeat > 1:
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
            for _ in xrange(block_args.num_repeat - 1):
                self._blocks.append(conv_block(block_args, self._global_params))

        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon
        if self._global_params.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        # Stem part.
        self._conv_stem = tf.keras.layers.Conv2D(
            filters=round_filters(32, self._global_params),
            kernel_size=[3, 3],
            strides=[2, 2],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._global_params.data_format,
            use_bias=False)
        self._bn0 = batchnorm(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)

        # Head part.
        self._conv_head = tf.keras.layers.Conv2D(
            filters=round_filters(1280, self._global_params),
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = batchnorm(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)

        self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
            data_format=self._global_params.data_format)
        if self._global_params.num_classes:
            self._fc = tf.keras.layers.Dense(
                self._global_params.num_classes,
                kernel_initializer=dense_kernel_initializer)
        else:
            self._fc = None

        if self._global_params.dropout_rate > 0:
            self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)
        else:
            self._dropout = None

    def call(self, inputs, training=True, features_only=None):
        """Implementation of call().
        Args:
          inputs: input tensors.
          training: boolean, whether the model is constructed for training.
          features_only: build the base feature network only.
        Returns:
          output tensors.
        """
        outputs = None
        self.endpoints = {}
        # Calls Stem layers
        with tf.name_scope('stem'):
            outputs = self._relu_fn(self._bn0(self._conv_stem(inputs), training=training))
        tf.compat.v1.logging.debug('Built stem layers with output shape: %s' % outputs.shape)
        self.endpoints['stem'] = outputs

        # Calls blocks.
        reduction_idx = 0
        for idx, block in enumerate(self._blocks):
            is_reduction = False
            if ((idx == len(self._blocks) - 1) or self._blocks[idx + 1].block_args().strides[0] > 1):
                is_reduction = True
                reduction_idx += 1

            with tf.name_scope('blocks_%s' % idx):
                drop_rate = self._global_params.drop_connect_rate
                if drop_rate:
                    drop_rate *= float(idx) / len(self._blocks)
                    tf.compat.v1.logging.debug('block_%s drop_connect_rate: %s' % (idx, drop_rate))
                outputs = block.call(outputs, training=training, drop_connect_rate=drop_rate)
                self.endpoints['block_%s' % idx] = outputs
                if is_reduction:
                    self.endpoints['reduction_%s' % reduction_idx] = outputs
                if block.endpoints:
                    for k, v in six.iteritems(block.endpoints):
                        self.endpoints['block_%s/%s' % (idx, k)] = v
                        if is_reduction:
                            self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
        self.endpoints['features'] = outputs

        if not features_only:
            # Calls final layers and returns logits.
            with tf.name_scope('head'):
                outputs = self._relu_fn(self._bn1(self._conv_head(outputs), training=training))
                outputs = self._avg_pooling(outputs)
                if self._dropout:
                    outputs = self._dropout(outputs, training=training)
                self.endpoints['global_pool'] = outputs
                if self._fc:
                    outputs = self._fc(outputs)
                self.endpoints['head'] = outputs
        return outputs

def efficientnet_params(model_name):
    """Get efficientnet params based on model name."""
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=[int(options['s'][0]), int(options['s'][1])],
            conv_type=int(options['c']) if 'c' in options else 0)

    def _encode_block_string(self, block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
            'c%d' % block.conv_type,
        ]
        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.
        Args:
          string_list: a list of strings, each string is a notation of block.
        Returns:
          A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.
        Args:
          blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
          a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None,
                 depth_coefficient=None,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2):
    """Creates a efficientnet model."""
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        data_format='channels_last',
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        relu_fn=tf.nn.swish)
    decoder = BlockDecoder()
    return decoder.decode(blocks_args), global_params


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model."""
    if model_name.startswith('efficientnet'):
        width_coefficient, depth_coefficient, image_size, dropout_rate = (efficientnet_params(model_name))
        blocks_args, global_params = efficientnet(width_coefficient, depth_coefficient, dropout_rate)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:
        # ValueError will be raised here if override_params has fields not included
        # in global_params.
        global_params = global_params._replace(**override_params)

    tf.compat.v1.logging.debug('global_params= %s', global_params)
    tf.compat.v1.logging.debug('blocks_args= %s', blocks_args)
    return blocks_args, global_params, image_size

def build_model(model_name, override_params):
    blocks_args, global_params, image_size = get_model_params(model_name, override_params)

    return Model(blocks_args, global_params), image_size
