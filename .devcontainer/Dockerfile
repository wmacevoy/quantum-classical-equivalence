FROM devcontainer-latex:latest

# Wigner-function scripts (cross_wigner_*.py) at the repo root use numpy
# and scipy.  The LaTeX toolchain comes from the base image.

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-numpy \
        python3-scipy \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
