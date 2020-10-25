#!/bin/bash

docker run \
    --rm \
    -v ${PWD}:/hsdl/ \
    -w /hsdl \
    -p 8888:8888 \
    timniven/hsdl:base \
        jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
