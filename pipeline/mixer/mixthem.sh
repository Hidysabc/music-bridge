#!/bin/bash

PRJ=/music-bridge
DATA=$PRJ/data

python $PRJ/pipeline/mixer/mix.py \
    s3://tagatune/music-bridge-tagger-best-0.9821-0.0790.weights.hdf5 \
    $DATA/skyfall.wav $DATA/crazy.wav --prefix crazy-skyfall

python $PRJ/pipeline/mixer/mix.py \
    s3://tagatune/music-bridge-tagger-best-0.9821-0.0790.weights.hdf5 \
    $DATA/million_reasons.wav $DATA/ten_thousand_reasons.wav \
    --prefix one_point_o_one_million_reasons

# Mary did you know we are never ever getting back together

# Still single ladies

# you make beautiful wrecking ball

# Build your kingdom uptown

