"""
Convert mp3 files to trainable mp3 file
"""

from __future__ import print_function
from __future__ import division

#import boto3
import glob
import librosa
import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd

FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

NCPU = mp.cpu_count()

PRJ = "/workspace/music-bridge"
DATA_PATH = os.path.join(PRJ, "data/tagatune")
MP3_PATH = os.path.join(DATA_PATH, "mp3")
OUT_PATH = os.path.join(PRJ, "data/tagatune/npy_stft")
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# mp3 io parameters
SR = 22050
MONO = False

allmp3 = glob.glob(os.path.join(MP3_PATH, "*/*.mp3"))
LOG.info("{} mp3 files found".format(len(allmp3)))

LOG.info("Convert mp3 to npy")
annot = pd.read_table(os.path.join(DATA_PATH, "annotations_final.csv"),
                      index_col="mp3_path").drop("clip_id", axis=1)

def convert_mp3_to_npy(mp3path, sr=SR, mono=MONO):
    LOG.debug(mp3path)
    key = mp3path.replace(MP3_PATH+"/", "")
    outfile = os.path.basename(mp3path).replace(".mp3", ".stft.npy")
    outpath = os.path.join(OUT_PATH, outfile)
    if key not in annot.index:
        return
    if not os.path.exists(outpath):
        try:
            x, sr = librosa.core.load(mp3path, sr=sr, mono=mono)
            D = librosa.stft(x)
            np.save(outpath, D)
        except:
            return

pool = mp.Pool(min(len(allmp3), NCPU))
res = pool.map_async(convert_mp3_to_npy, allmp3)
out = res.get()

LOG.info("Done :)")

