FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# install dependencies
COPY requirements-light.txt .
RUN pip install -r requirements-light.txt
RUN rm requirements-light.txt
