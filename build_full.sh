#!/bin/bash

docker build \
    --no-cache \
    -f docker/hsdl.Dockerfile \
    -t timniven/hsdl:latest \
    .
