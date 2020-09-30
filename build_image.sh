#!/bin/bash

docker build \
    -f docker/hsdl.Dockerfile \
    -t hsdl:latest \
    .
