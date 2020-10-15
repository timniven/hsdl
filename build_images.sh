#!/bin/bash

docker build \
    -f docker/hsdl-base.Dockerfile \
    -t timniven/hsdl:base \
    .

docker build \
    -f docker/hsdl.Dockerfile \
    -t timniven/hsdl:latest \
    .

docker build \
    -f docker/tensorboard.Dockerfile \
    -t timniven/hsdl:tensorboard \
    .

# make sure it's available to dependent code
docker push timniven/hsdl:base
docker push timniven/hsdl:latest
docker push timniven/hsdl:tensorboard
