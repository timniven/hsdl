# if I remember correctly, 10.1 needed for easy install of tensorflow
FROM nvidia/cuda:10.1-base-ubuntu18.04

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
COPY download_models.py /temp/download_models.py
RUN python3.8 /temp/download_models.py

RUN rm -r /temp

# install spacy and download models we are using
RUN python3.8 -m pip install -U spacy[cuda101]
RUN python3.8 -m pip install -U spacy-lookups-data
RUN python3.8 -m spacy download ja_core_news_lg
RUN python3.8 -m spacy download zh_core_web_lg
RUN python3.8 -m spacy download en_core_web_lg
