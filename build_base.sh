#!/bin/bash


./download_ckip.sh


docker build \
    --no-cache \
    -f docker/hsdl-base.Dockerfile \
    -t timniven/hsdl:base \
    .
