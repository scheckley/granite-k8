# syntax = docker/dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /usr/local/granite

COPY . /usr/local/granite

# Add granite user
RUN groupadd -g 1001 granite && useradd -u 1001 -g granite granite
RUN mkdir /home/granite

#Add permissions
RUN chown -R granite:granite /usr/local/granite && \
    chgrp -R 0 /usr/local/granite && \
    chmod -R 775 /usr/local/granite && \
    chmod -R 775 /home/granite
#Specify the user with UID as OpenShift assigns random

USER 1001

USER root 

RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install -y curl python3-pip git
RUN curl -sSL https://get.docker.com/ | sh
RUN python3 -m pip install -U pip
RUN git clone https://github.com/huggingface/transformers && \
    cd transformers/ && \
    pip install ./

CMD ["python3", "app.py"]