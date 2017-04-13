"""
Music tagger using CRNN
"""

from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.models import Model
from keras.layers.advanced_activations import ELU
from keras.layers import Dense, Input, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.recurrent import GRU
from keras.regularizers import l2


def _bn_elu(input, blockname):
    norm = BatchNormalization(axis = 3, name = "bn" + str(blockname))(input)
    return ELU(name = "elu" + str(blockname))(norm)

def _conv_bn_elu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    blockname = conv_params.setdefault("block", 0)
    def f(input):
        conv = Conv2D(filters = filters,
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
        if input_shape and len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (spectral_length, music_length, 2)")
        if input_tensor is None:
            music_input = Input(shape = input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                music_input = Input(tensor = input_tensor, shape = input_shape)
            else:
                music_input = input_tensor
        """
        conv0 = _conv_bn_elu(filters = 128,
                             kernel_size = 2048,
                             strides = 512,
                             block = 0)(music_input)
        """
        conv1 = _conv_bn_elu(filters = 64,
                                kernel_size = (3, 3),
                                strides = (1, 1),
                                block = 1)(music_input)
        pool1 = MaxPooling2D(pool_size = (4, 2),
                                strides = (4, 2),
                                padding = "same",
                                name = "pool1")(conv1)
        conv2 = _conv_bn_elu(filters = 128,
                                kernel_size = (3, 3),
                                strides = (1, 1),
                                block = 2)(pool1)
        pool2 = MaxPooling2D(pool_size = (4, 2),
                                strides = (4, 2),
                                padding = "same",
                                name = "pool2")(conv2)
        conv3 = _conv_bn_elu(filters = 128,
                                kernel_size = (3, 3),
                                strides = (1, 1),
                                block = 3)(pool2)
        pool3 = MaxPooling2D(pool_size = (4, 2),
                                strides = (4, 2),
                                padding = "same",
                                name = "pool3")(conv3)
        conv4 = _conv_bn_elu(filters = 128,
                                kernel_size = (3, 3),
                                strides = (1, 1),
                                block = 4)(pool3)
        pool4 = MaxPooling2D(pool_size = (4, 2),
                                strides = (4, 2),
                                padding = "same",
                                name = "pool4")(conv4)
        conv5 = _conv_bn_elu(filters = 128,
                                kernel_size = (3, 3),
                                strides = (1, 1),
                                block = 5)(pool4)
        pool5 = MaxPooling2D(pool_size = (5, 4),# pool the first dim to 1
                                strides = (5, 4),
                                padding = "same",
                                name = "pool5")(conv5)
        reshape = Reshape((int(pool5.shape[1]) * int(pool5.shape[2]),
                           int(pool5.shape[3])))(pool5)
        gru1 = GRU(units = 32, dropout = 0.3,
                    recurrent_dropout = 0.3,
                    kernel_initializer = "he_normal",
                    kernel_regularizer = l2(1e-4),
                    return_sequences = True,
                    name = "gru1")(reshape)
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


