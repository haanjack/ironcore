ARG NGC_VERSION=23.12
FROM nvcr.io/nvidia/pytorch:${NGC_VERSION}-py3

RUN apt update && apt install -y --no-install-recommends \
    llvm lldb llvm-dev llvm-runtime \
    libaio-dev &&\
    rm -rf /var/lib/apt/lists/*

ADD ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt &&\
    rm /tmp/requirements.txt
