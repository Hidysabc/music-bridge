#####
#
# debian container with avconv libav-tools installed
# U-Net Segmentation Approach to Cancer Diagnosis
#

FROM debian:stable

MAINTAINER Wei-Yi Cheng

RUN apt-get update && apt-get install -y libav-tools && \
    apt-get -qqy clean && rm -rf /var/lib/apt/lists/*

