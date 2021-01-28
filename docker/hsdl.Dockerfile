FROM timniven/hsdl:base

ARG OAUTH_KEY

RUN python3.8 -m pip install git+https://github.com/timniven/hsdl.git
RUN python3.8 -m pip install git+https://${OAUTH_KEY}:x-oauth-basic@github.com/doublethinklab/dtl-common-python.git
