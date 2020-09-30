FROM python:3.8

RUN mkdir /temp
COPY requirements.txt /temp
RUN pip install -r temp/requirements.txt
RUN rm -r temp

RUN mkdir hsdl
COPY . /hsdl
WORKDIR hsdl
