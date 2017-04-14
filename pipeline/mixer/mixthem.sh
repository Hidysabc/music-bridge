#!/bin/bash

PRJ=/music-bridge
DATA=/tmp/data
DOCKERRUN="docker run -it --rm -v /tmp:/tmp -v /home/ubuntu/.aws:/root/.aws music-bridge-mix python"
MODEL_PATH="s3://tagatune/music-bridge-tagger-best.weights.hdf5"

# crazy skyfall

$DOCKERRUN $PRJ/pipeline/mixer/mix.py $MODEL_PATH \
    $DATA/skyfall.wav $DATA/crazy.wav --prefix crazy-skyfall

# 1.01 million reasons

$DOCKERRUN $PRJ/pipeline/mixer/mix.py $MODEL_PATH \
    $DATA/million_reasons.wav $DATA/ten_thousand_reasons.wav \
    --prefix one_point_o_one_million_reasons

# Mary did you know we are never ever getting back together

$DOCKERRUN $PRJ/pipeline/mixer/mix.py $MODEL_PATH \
    $DATA/mary_did_you_know.wav $DATA/we_are_never_ever_getting_back_together.wav \
    --prefix one_o_one_million_reasons

# Still single ladies

$DOCKERRUN $PRJ/pipeline/mixer/mix.py $MODEL_PATH \
    $DATA/still.wav $DATA/single_ladies.wav \
    --prefix still_single_ladies

# you make beautiful wrecking ball

$DOCKERRUN $PRJ/pipeline/mixer/mix.py $MODEL_PATH \
    $DATA/beautiful_things.wav $DATA/wrecking_ball.wav \
    --prefix beautiful_wrecking_ball

# Build your kingdom uptown

$DOCKERRUN $PRJ/pipeline/mixer/mix.py $MODEL_PATH \
    $DATA/build_your_kingdom_here.wav $DATA/uptown_funk.wav \
    --prefix build_your_kingdom_uptown

