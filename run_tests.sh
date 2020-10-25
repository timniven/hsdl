#!/bin/bash

docker run \
    --rm \
    -v ${PWD}:/hsdl/ \
    -w /hsdl \
    timniven/hsdl:base \
        python3.8 -m unittest discover tests
