#####
#
# Docker file to build container with Keras and MusicTaggerCRNN
#

FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Wei-Yi Cheng

RUN apt-get update && apt-get install -y git-core libav-tools && \
    rm -rf /var/lib/apt/lists/*

RUN pip install keras pandas h5py && \
    pip install librosa && \
    pip install boto3 && \
    rm -rf /root/.cache/pip/*

