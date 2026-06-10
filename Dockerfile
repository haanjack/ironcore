ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

RUN apt update && apt install -y --no-install-recommends \
    llvm lldb llvm-dev llvm-runtime \
    libaio-dev &&\
    rm -rf /var/lib/apt/lists/*

ADD ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt &&\
    rm /tmp/requirements.txt

# Make the runtime-mounted workspace (/workspace) importable without PYTHONPATH,
# and register the `ironcore` CLI entry point.
# The source is mounted at runtime; this bakes the path plumbing into the image.
RUN python3 -c "import site; open(site.getsitepackages()[0]+'/ironcore-dev.pth','w').write('/workspace\n')" && \
    printf '#!/usr/bin/env python3\nfrom ironcore.__main__ import main\nmain()\n' \
        > /usr/local/bin/ironcore && \
    chmod +x /usr/local/bin/ironcore
