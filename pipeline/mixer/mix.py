'''Mix music with neural network

Run the script with:
```
python mix.py path_to_base_mp3.mp3 path_to_reference.mp3
```

Optional parameters:
```
--prefix, prefix of the mixed music file name
--iter, To specify the number of iterations the style transfer takes place (Default is 10)
--content_weight, The weight given to the content loss (Default is 0.025)
--style_weight, The weight given to the style loss (Default is 1.0)
--tv_weight, The weight given to the total variation loss (Default is 1.0)
```

It is preferable to run this script on GPU, for speed.

# References
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
'''

from __future__ import print_function
import librosa
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import argparse
import boto3
import os
import logging

from keras import backend as K

from musicbridge.tagger import MusicBridgeTagger

K.set_learning_phase(0)

FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT)
LOG = logging.getLogger("music-bridge-mixer")
LOG.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('model_weights_path', metavar='model_weights_path', type=str,
                    help='Path to the trained model weights')
parser.add_argument('base_music_path', metavar='base', type=str,
                    help='Path to the music to transform.')
parser.add_argument('style_reference_music_path', metavar='ref', type=str,
                    help='Path to the style reference music.')
parser.add_argument('--prefix', type=str, default="mixed",
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')

args = parser.parse_args()
model_weights_path = args.model_weights_path
base_music_path = args.base_music_path
style_reference_music_path = args.style_reference_music_path
result_prefix = args.prefix
iterations = args.iter

# these are the weights of the different loss components
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

"""
scripts for test
"""
"""
model_weights_path = "s3://tagatune/music-bridge-tagger-best-0.9867-0.1156.weights.hdf5"
base_music_path = "/workspace/music-bridge/data/skyfall.wav"
style_reference_music_path = "/workspace/music-bridge/data/crazy.wav"
result_prefix = "crazy-skyfall"
total_variation_weight = 1.
style_weight = 1.
content_weight = 0.025
iterations = 10
"""

channels = 1
music_length = 642185
s3bucket = "tagatune"
output_features = 188


input_tmp_dir = "/tmp"
if model_weights_path.startswith("s3://"):
    LOG.info("download model from s3")
    tokens = model_weights_path.replace("s3://", "").split("/")
    _s3bucket = tokens[0]
    model_weights_file_path = "/".join(tokens[1:])
    model_weights_name = os.path.basename(model_weights_file_path)
    s3_client = boto3.client("s3")
    model_weights_local_path = os.path.join(input_tmp_dir, model_weights_name)
    s3_client.download_file(_s3bucket, model_weights_file_path,
                            model_weights_local_path)
else:
    model_weights_local_path = model_weights_path

def read_music(path):
    arr, sr = librosa.core.load(path, mono=True, sr=22050)
    if arr.shape[0] < music_length:
        pad_length = (music_length - arr.shape[0]) / 2
        arr = np.lib.pad(arr,
                            (pad_length, music_length - pad_length - arr.shape),
                            "constant", constant_values=(0, 0))
    elif arr.shape[0] > music_length:
        cut_length = (arr.shape[0] - music_length) / 2
        arr = arr[cut_length:(cut_length + music_length)]
    return np.expand_dims(np.expand_dims(arr, axis=1), axis=0)

def deprocess_music(x):
    x = x.reshape((music_length, 1))
    x = np.clip(x, -1, 1)
    return x

# get tensor representations of our images
base_music = K.variable(read_music(base_music_path))
style_reference_music = K.variable(read_music(style_reference_music_path))

# this will contain our generated image
combination_music = K.placeholder((1, music_length, 1))

# combine the 3 music into a single Keras tensor
input_tensor = K.concatenate([base_music,
                              style_reference_music,
                              combination_music], axis=0)

model = MusicBridgeTagger.build(input_tensor=input_tensor, include_top=True,
                                num_outputs=output_features)
model.load_weights(model_weights_local_path)


# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 2
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image

def style_loss(style, combination):
    assert K.ndim(style) == 2
    assert K.ndim(combination) == 2
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 1
    size = music_length
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent

def total_variation_loss(x):
    assert K.ndim(x) == 3
    a = K.square(x[:, :music_length - 1, :] - x[:, 1:, :])
    return K.sum(K.pow(a, 1.25))

# combine these loss functions into a single scalar
loss = K.variable(0.)
layer_features = outputs_dict['conv3']
base_music_features = layer_features[0, :, :]
combination_features = layer_features[2, :, :]
loss += content_weight * content_loss(base_music_features,
                                      combination_features)

feature_layers = ['conv1', 'conv2', 'conv3', 'conv4']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :]
    combination_features = layer_features[2, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

loss += total_variation_weight * total_variation_loss(combination_music)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_music)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_music], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, music_length, 1))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
x = np.random.uniform(0, 2, (music_length, 1)) - 1.

for i in range(iterations):
    LOG.info('Start of iteration {}'.format(i))
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    LOG.info('Current loss value: {}'.format(min_val))
    # save current generated image
    if i % 100 == 99:
        music_arr = deprocess_music(x.copy())
        fname = result_prefix + '_at_iteration_%d.wav' % (i+1)
        out_path = os.path.join(input_tmp_dir, fname)
        maxv = np.iinfo(np.int16).max
        librosa.output.write_wav(out_path, (music_arr*maxv).astype(np.int16),
                                 sr=22050)
        s3_client.upload_file(out_path, s3bucket, fname)
        LOG.info('Mixed music uploaded to s3://{}/{}'.format(s3bucket, fname))


music_arr = deprocess_music(x.copy())
fname = result_prefix + '_at_iteration_%d.wav' % iterations
out_path = os.path.join(input_tmp_dir, fname)
maxv = np.iinfo(np.int16).max
librosa.output.write_wav(out_path, (music_arr*maxv).astype(np.int16),
                         sr=22050)
s3_client.upload_file(out_path, s3bucket, fname)
LOG.info('Mixed music uploaded to s3://{}/{}'.format(s3bucket, fname))
LOG.info("Done :)")
