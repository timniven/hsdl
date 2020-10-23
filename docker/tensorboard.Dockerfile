# https://hub.docker.com/r/pccl/tensorboard
FROM python:3.8

RUN python3 -m pip install --upgrade pip
RUN pip --no-cache-dir install tensorboard

EXPOSE 6006

#CMD ["tensorboard", "--logdir=/runs", "--bind_all"]
# docker run -p 6006:6006 -d --name=tensorboard --restart=always -v /path/to/runs/:/runs pccl/tensorboard
