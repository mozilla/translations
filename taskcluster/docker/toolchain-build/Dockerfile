FROM $DOCKER_IMAGE_PARENT
LABEL maintainer="Mozilla Release Engineering <release+docker@mozilla.com>"

RUN apt-get update -qq \
    # We need to install tzdata before all of the other packages. Otherwise it will show an interactive dialog that
    # we cannot navigate while building the Docker image.
    && apt-get install -y tzdata \
    && apt-get install -y wget \
                          curl \
                          zip \
                          build-essential \
                          gcc \
                          g++ \
                          make \
                          cmake \
                          libboost-dev \
                          libboost-all-dev \
                          zstd \
                          tar \
                          libxml2 \
                          libhunspell-dev \
                          bc  \
                          autoconf \
                          automake \
                          autopoint \
                          libtool \
    && apt-get clean

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | apt-key add -
COPY intel-mkl.list /etc/apt/sources.list.d/intel-mkl.list

RUN apt-get update -qq \
    && apt-get install -y intel-mkl-64bit-2020.0-088 \
    && apt-get clean

RUN locale-gen "$LANG"

RUN pip install zstandard


# %include-run-task

ENV SHELL=/bin/bash \
    HOME=/builds/worker \
    PATH="/builds/worker/.local/bin:$PATH"

VOLUME /builds/worker/checkouts
VOLUME /builds/worker/.cache

USER root
