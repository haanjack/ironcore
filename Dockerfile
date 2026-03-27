ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

RUN apt update && apt install -y --no-install-recommends \
    llvm lldb llvm-dev llvm-runtime \
    libaio-dev &&\
    rm -rf /var/lib/apt/lists/*

ADD ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt &&\
    rm /tmp/requirements.txt
