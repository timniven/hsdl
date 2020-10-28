#!/bin/bash

docker run \
    --rm \
    --gpus all \
    -v ${PWD}:/hsdl/ \
    -w /hsdl \
    timniven/hsdl:base \
        python3.8 -m unittest $1
