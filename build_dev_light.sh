#!/bin/bash

docker build \
    $1 \
    -f docker/hsdl-dev-light.Dockerfile \
    -t timniven/hsdl-dev-light:latest \
    .
