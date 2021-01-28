#!/bin/bash

docker build \
    --no-cache \
    -f docker/hsdl.Dockerfile \
    -t timniven/hsdl:latest \
    --build-arg OAUTH_KEY=$(cat ~/.secret/oauth_key.txt) \
    .
