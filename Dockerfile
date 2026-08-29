ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

RUN apt update && apt install -y --no-install-recommends \
    llvm lldb llvm-dev llvm-runtime \
    libaio-dev &&\
    rm -rf /var/lib/apt/lists/*

# ROCDXG — the user-mode bridge that lets the ROCr runtime reach the GPU through
# DXCore (/dev/dxg) instead of /dev/kfd. Required for ROCm under WSL2, where no
# amdgpu kernel driver exists. Self-contained: nothing needs to be installed in
# the WSL distro, only libdxcore.so bind-mounted in at run time (see launch.sh).
# Build with --build-arg ROCDXG_VERSION=1.2.1; empty means "not a WSL build".
ARG ROCDXG_VERSION=""
RUN if [ -n "$ROCDXG_VERSION" ]; then \
        curl -fsSL -o /tmp/rocdxg.deb \
            "https://github.com/ROCm/librocdxg/releases/download/v${ROCDXG_VERSION}/rocdxg-roct_${ROCDXG_VERSION}_amd64.deb" && \
        dpkg -i /tmp/rocdxg.deb && \
        ldconfig && \
        rm /tmp/rocdxg.deb ; \
    fi

ADD ./requirements.txt /tmp/requirements.txt
ADD ./requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install -r /tmp/requirements.txt -r /tmp/requirements-dev.txt &&\
    rm /tmp/requirements.txt /tmp/requirements-dev.txt

# Make the runtime-mounted workspace (/workspace) importable without PYTHONPATH.
# The source is mounted at runtime; ./ironcore-run handles CLI dispatch.
RUN python3 -c "import site; open(site.getsitepackages()[0]+'/ironcore-dev.pth','w').write('/workspace\n')"
