#!/bin/bash

if [ ! -d temp ]; then
  echo "temp dir not found, creating..."
  mkdir temp
fi

# check if data exists and return if so
if [ ! -d temp/ckip ]; then
  echo "downloading data..."
  wget -O temp/ckip.zip http://ckip.iis.sinica.edu.tw/data/ckiptagger/data.zip
  echo "unzipping..."
  unzip temp/ckip.zip -d temp/ckip
  echo "removing zip file..."
  rm temp/ckip.zip
else
  echo "Ckip data already downloaded."
fi
