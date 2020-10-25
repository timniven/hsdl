FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt update
RUN apt -y --no-install-recommends install \
        git \
        python3.8 \
        python3-pip \
        python3-setuptools \
        software-properties-common \
        build-essential \
        python3.8-dev \
        python3-distutils \
        python3-apt \
        python3-wheel

# it is necessary to upgrade pip for tensorflow > 2
RUN python3.8 -m pip install --upgrade pip
RUN mkdir /temp
COPY requirements.txt /temp/requirements.txt
RUN python3.8 -m pip install -r /temp/requirements.txt
RUN rm -r /temp
