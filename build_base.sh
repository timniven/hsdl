#!/bin/bash

docker build \
    -f docker/hsdl-base.Dockerfile \
    -t timniven/hsdl:base \
    .
