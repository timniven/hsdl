#!/bin/bash

./build_base.sh
./build_full.sh
./build_tensorboard.sh

# make sure it's available to dependent code
./push_base.sh
./push_full.sh
./push_tensorboard.sh
