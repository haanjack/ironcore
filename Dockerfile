ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

RUN apt update && apt install -y --no-install-recommends \
    llvm lldb llvm-dev llvm-runtime \
    libaio-dev &&\
    rm -rf /var/lib/apt/lists/*

ADD ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt &&\
    rm /tmp/requirements.txt

# Make the runtime-mounted workspace (/workspace) importable without PYTHONPATH.
# The source is mounted at runtime; ./ironcore-run handles CLI dispatch.
RUN python3 -c "import site; open(site.getsitepackages()[0]+'/ironcore-dev.pth','w').write('/workspace\n')"
