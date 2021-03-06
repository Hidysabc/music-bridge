from __future__ import print_function
from __future__ import division

import boto3
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import logging
import numpy as np
import os
import pandas as pd
import sys

from musicbridge.callbacks import ModelCheckpointS3
from musicbridge.tagger import MusicBridgeTagger

model_name = "music-bridge-tagger"

s3bucket = "tagatune"
input_meta_filename = "metadata.csv"
#input_tmp_dir = "/workspace/music-bridge/data/tagatune"
input_tmp_dir = '/tmp/data'
if not os.path.exists(input_tmp_dir):
    os.makedirs(input_tmp_dir)

batch_size = 2
spectral_size = 1025
music_length = 1255
output_features = 28
epochs = 1000


FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'

logging.basicConfig(format=FORMAT)
LOG = logging.getLogger(model_name)
LOG.setLevel(logging.DEBUG)

s3bucket = "tagatune"
s3_client = boto3.client("s3")

def flatten_complex(z):
    return np.array([np.real(z), np.imag(z)]).transpose((1, 2, 0))

def generate_data_from_directory(files, meta, batch_size, input_dir):
    tmp = meta[meta.filename.isin(files)].iloc[:, :-1].copy()
    tmp.set_index("filename", inplace=True)
    while 1:
        #permute index
        p_files = np.random.permutation(files)
        for i in np.arange(int(len(files)/batch_size+0.5)):
            fs = p_files[i*batch_size:(i+1)*batch_size]
            Xb = np.array([flatten_complex(np.load(os.path.join(input_dir, j)))
                           for j in fs])
            Yb = tmp.loc[fs, :].values
            yield (Xb, Yb)


if len(sys.argv)>1:
    LOG.info("Start loading model...")
    model_path = sys.argv[1]
    local_model_path = os.path.join(input_tmp_dir,
                                    os.path.basename(model_path))
    s3_client.download_file(s3bucket, sys.argv[1], local_model_path)
    model = MusicBridgeTagger.build(input_shape=(spectral_size, music_length, 2),
                                    include_top=True,
                                    num_outputs=output_features)
    model.load_weights(local_model_path)
else:
    LOG.info("Start building model...")
    model = MusicBridgeTagger.build(input_shape=(spectral_size, music_length, 2),
                                    include_top=True,
                                    num_outputs=output_features)


meta = pd.read_csv(os.path.join(input_tmp_dir, input_meta_filename))
meta = meta.loc[:, meta.sum() > 1000]
meta["filename"] = meta.filename + ".stft.npy"
train_files = meta.filename[meta.set == "train"]
valid_files = meta.filename[meta.set == "valid"]

import glob
files = [os.path.basename(f) for f in glob.glob(os.path.join(input_tmp_dir,
                                                             "npy_stft/*.npy"))]
train_files = train_files[train_files.isin(files)]
valid_files = valid_files[valid_files.isin(files)]
LOG.info("{} training files".format(train_files.size))
LOG.info("{} validation files".format(valid_files.size))

train_gen = generate_data_from_directory(
    train_files, meta, batch_size,
    input_dir = os.path.join(input_tmp_dir, "npy_stft")
)
valid_gen = generate_data_from_directory(
    valid_files, meta, batch_size,
    input_dir = os.path.join(input_tmp_dir, "npy_stft")
)


model.compile(loss="binary_crossentropy",
              optimizer=Adam(),
              metrics=["accuracy"])

checkpointer = ModelCheckpointS3(monitor="val_loss",
                                 filepath="/tmp/{}-best.weights.hdf5".format(model_name),
                                 bucket = s3bucket,
                                 verbose=0, save_best_only=True,
                                 save_weights_only=True)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=5, min_lr=1e-5)

LOG.info("start training")
train_steps = max(1, int(len(train_files) / float(batch_size) + 0.5))
valid_steps = max(1, int(len(valid_files) / float(batch_size) + 0.5))
history = model.fit_generator(train_gen,
                              steps_per_epoch = train_steps,
                              epochs = epochs,
                              verbose = 1,
                              validation_data = valid_gen,
                              validation_steps = valid_steps,
                              callbacks= [checkpointer, reduce_lr])

val_loss = history.history["val_loss"]
val_acc = history.history["val_acc"]
best_model_name = "{model_name}-best-{val_acc:.4f}-{val_loss:.4f}.weights.hdf5".format(
    model_name = model_name, val_acc = val_acc[np.argmin(val_loss)], val_loss = np.min(val_loss)
)
copy_source = {
    "Bucket": s3bucket,
    "Key": "{}-best.weights.hdf5".format(model_name)
}
s3_client.copy(copy_source, s3bucket, best_model_name)

LOG.info("save final model")
final_model_name = "{model_name}-{epochs}-{val_acc:.4f}-{val_loss:.4f}.weights.hdf5"\
    .format(model_name = model_name, epochs = epochs,
            val_acc = val_acc[-1], val_loss = val_loss[-1])
final_model_path = "/tmp/{}".format(final_model_name)
model.save_weights(final_model_path)
s3_client.upload_file(final_model_path, s3bucket, final_model_name)

LOG.info("Finish! :)")


