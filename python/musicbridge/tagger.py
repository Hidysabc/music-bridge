"""
Music tagger using CRNN
"""

from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.models import Model
from keras.layers.advanced_activations import ELU
from keras.layers import Dense, Input
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import GRU
from keras.regularizers import l2


def _bn_elu(input, blockname):
    norm = BatchNormalization(axis = 2, name = "bn" + str(blockname))(input)
    return ELU(name = "elu" + str(blockname))(norm)

def _conv_bn_elu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    blockname = conv_params.setdefault("block", 0)
    def f(input):
        conv = Conv1D(filters = filters,
                      kernel_size = kernel_size,
                      strides = strides,
                      kernel_initializer = kernel_initializer,
                      padding = padding,
                      kernel_regularizer = kernel_regularizer,
                      name = "conv" + str(blockname))(input)
        return _bn_elu(conv, blockname)
    return f

class MusicBridgeTagger(object):
    @staticmethod
    def build(include_top=True, input_tensor=None,
              input_shape=None, num_outputs=None):
        """Build a music tagger
        :param input_shape: The input shape in the form (channels, length)
        :param num_outputs: number of outputs at final softmax layer
        :returns: :class:`keras.Model`"""
        if input_shape and len(input_shape) != 2:
            raise Exception("Input shape should be a tuple (channels, length)")
        if input_tensor is None:
            music_input = Input(shape = input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                music_input = Input(tensor = input_tensor, shape = input_shape)
            else:
                music_input = input_tensor

        conv0 = _conv_bn_elu(filters = 128,
                             kernel_size = 2048,
                             strides = 512,
                             block = 0)(music_input)
        conv1 = _conv_bn_elu(filters = 64,
                             kernel_size = 3,
                             strides = 1,
                             block = 1)(conv0)
        pool1 = MaxPooling1D(pool_size = 2,
                             strides = 2,
                             padding = "same",
                             name = "pool1")(conv1)
        conv2 = _conv_bn_elu(filters = 128,
                             kernel_size = 3,
                             strides = 1,
                             block = 2)(pool1)
        pool2 = MaxPooling1D(pool_size = 3,
                             strides = 3,
                             padding = "same",
                             name = "pool2")(conv2)
        conv3 = _conv_bn_elu(filters = 128,
                             kernel_size = 3,
                             strides = 1,
                             block = 3)(pool2)
        pool3 = MaxPooling1D(pool_size = 4,
                             strides = 4,
                             padding = "same",
                             name = "pool3")(conv3)
        conv4 = _conv_bn_elu(filters = 128,
                             kernel_size = 3,
                             strides = 1,
                             block = 4)(pool3)
        pool4 = MaxPooling1D(pool_size = 4,
                             strides = 4,
                             padding = "same",
                             name = "pool4")(conv4)
        gru1 = GRU(units = 32, dropout = 0.3,
                   recurrent_dropout = 0.3,
                   kernel_initializer = "he_normal",
                   kernel_regularizer = l2(1e-4),
                   return_sequences = True,
                   name = "gru1")(pool4)
        gru2 = GRU(units = 32, dropout = 0.3,
                   recurrent_dropout = 0.3,
                   kernel_initializer = "he_normal",
                   kernel_regularizer = l2(1e-4),
                   return_sequences = False,
                   name = "gru2")(gru1)
        dense = Dense(units = num_outputs, kernel_initializer = "he_normal",
                      activation = "softmax")(gru2)
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = music_input

        return Model(inputs = inputs, outputs = dense, name = "music_bridge_tagger")


