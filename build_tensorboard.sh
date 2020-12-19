#!/bin/bash

docker build \
    -f docker/tensorboard.Dockerfile \
    -t timniven/hsdl:tensorboard \
    .
