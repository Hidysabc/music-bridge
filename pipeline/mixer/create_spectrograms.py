from __future__ import print_function

import boto3
import librosa
from librosa.display import specshow
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

filenames = sys.argv[1:]
s3_files = filter(lambda s: s.startswith("s3://"), filenames)

def download_s3_file(s3path):
    s3_client = boto3.client("s3")
    tokens = s3path.replace("s3://", "").split("/")
    s3_bucket = tokens[0]
    local_filename = os.path.join("/tmp", tokens[-1])
    s3_client.download_files(s3_bucket, "/".join(tokens[1:]), local_filename)
    return local_filename

if s3_files:
    downloaded_files = [download_s3_file(s3f) for s3f in s3_files]
    filenames = list(set(filenames) - set(s3_files) + set(downloaded_files))

def display_spectrograms(y, sr, filename=None):
    plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

    plt.subplot(4, 2, 1)
    specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')

    plt.subplot(4, 2, 2)
    specshow(D, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')

    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)
    plt.subplot(4, 2, 3)
    specshow(CQT, y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrogram (note)')

    plt.subplot(4, 2, 4)
    specshow(CQT, y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrogram (Hz)')

    C = librosa.feature.chroma_cqt(y=y, sr=sr)
    plt.subplot(4, 2, 5)
    specshow(C, y_axis='chroma')
    plt.colorbar()
    plt.title('Chromagram')

    plt.subplot(4, 2, 6)
    specshow(D, cmap='gray_r', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear power spectrogram (grayscale)')

    plt.subplot(4, 2, 7)
    specshow(D, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log power spectrogram')

    plt.subplot(4, 2, 8)
    Tgram = librosa.feature.tempogram(y=y, sr=sr)
    specshow(Tgram, x_axis='time', y_axis='tempo')
    plt.colorbar()
    plt.title('Tempogram')
    plt.tight_layout()

    if filename:
        plt.savefig(filename)

def specshow_file(filename):
    y, sr = librosa.core.load(filename)
    LOG.debug("File: {}".format(filename))
    LOG.debug("Sample rate: {}".format(sr))
    LOG.debug("Sample max: {}".format(y.max()))
    LOG.debug("Sample min: {}".format(y.min()))
    LOG.debug("Sample var: {}".format(y.var()))
    outfile = os.path.basename(filename) + ".png"
    LOG.info("Save spectrograms to {}".format(outfile))
    display_spectrograms(y, sr, outfile)

for f in filenames:
    LOG.info("Process {}".format(f))
    specshow_file(f)
