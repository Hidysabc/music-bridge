#!/bin/bash

PRJ=/music-bridge
DATA=/tmp/data
DOCKERRUN="nvidia-docker run -it --rm -v /tmp:/tmp -v /home/ubuntu/.aws:/root/.aws music-bridge-mix python"
MODEL_PATH="s3://tagatune/music-bridge-tagger-best-0.9254-0.2359.weights.hdf5"
PARAMS="--iter=30 --tv_weight=1e-6 --style_weight=1 --content_weight=1e-3"

# crazy skyfall

$DOCKERRUN $PRJ/pipeline/mixer/mix.py $MODEL_PATH \
    $DATA/skyfall.wav $DATA/crazy.wav --prefix crazy-skyfall $PARAMS

# 1.01 million reasons

$DOCKERRUN $PRJ/pipeline/mixer/mix.py $MODEL_PATH \
    $DATA/million_reasons.wav $DATA/ten_thousand_reasons.wav \
    --prefix one_point_o_one_million_reasons $PARAMS

# Mary did you know we are never ever getting back together

$DOCKERRUN $PRJ/pipeline/mixer/mix.py $MODEL_PATH \
    $DATA/mary_did_you_know.wav $DATA/we_are_never_ever_getting_back_together.wav \
    --prefix one_o_one_million_reasons $PARAMS

# Still single ladies

$DOCKERRUN $PRJ/pipeline/mixer/mix.py $MODEL_PATH \
    $DATA/still.wav $DATA/single_ladies.wav \
    --prefix still_single_ladies $PARAMS

# you make beautiful wrecking ball

$DOCKERRUN $PRJ/pipeline/mixer/mix.py $MODEL_PATH \
    $DATA/beautiful_things.wav $DATA/wrecking_ball.wav \
    --prefix beautiful_wrecking_ball $PARAMS

# Build your kingdom uptown

$DOCKERRUN $PRJ/pipeline/mixer/mix.py $MODEL_PATH \
    $DATA/build_your_kingdom_here.wav $DATA/uptown_funk.wav \
    --prefix build_your_kingdom_uptown $PARAMS

