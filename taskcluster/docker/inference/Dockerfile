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
                          libpcre2-dev \
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

ARG NVM_VERSION=0.40.1
ARG NVM_CHECKSUM=abdb525ee9f5b48b34d8ed9fc67c6013fb0f659712e401ecd88ab989b3af8f53

ARG NODE_VERSION=23.1.0
ARG NODE_CHECKSUM=c438df636858200cffdfc3b579a1d784047da5c0e70dd647616c215726e254e2

ARG NPM_VERSION=10.9.0
ARG NPM_CHECKSUM=8e5f6f3429f8cdbe693cdc29904e9d5a7b127a494bd15c804bd54c7403bfcbe7

RUN curl -o install.sh -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v${NVM_VERSION}/install.sh && \
    echo "${NVM_CHECKSUM} install.sh" | sha256sum -c - && \
    bash install.sh && \
    export NVM_DIR="$HOME/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && \
    nvm install ${NODE_VERSION} && \
    nvm use ${NODE_VERSION} && \
    echo "${NODE_CHECKSUM} $NVM_DIR/versions/node/$(nvm version)/bin/node" | sha256sum -c - && \
    echo "${NPM_CHECKSUM} $NVM_DIR/versions/node/$(nvm version)/bin/npm" | sha256sum -c - && \
    ln -sf "$NVM_DIR/versions/node/$(nvm version)/bin/node" /usr/local/bin/node && \
    ln -sf "$NVM_DIR/versions/node/$(nvm version)/bin/npm" /usr/local/bin/npm && \
    rm install.sh

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | apt-key add -
COPY intel-mkl.list /etc/apt/sources.list.d/intel-mkl.list

RUN apt-get update -qq \
    && apt-get install -y intel-mkl-64bit-2020.0-088 \
    && apt-get clean

RUN locale-gen "$LANG"

# Install taskfile - https://taskfile.dev/
# Keep the version in sync with docker/Dockerfile.
RUN curl -sSLf "https://github.com/go-task/task/releases/download/v3.35.1/task_linux_amd64.tar.gz" \
    | tar -xz -C /usr/local/bin

ENV SHELL=/bin/bash \
    HOME=/builds/worker \
    PATH="/builds/worker/.local/bin:$PATH"

VOLUME /builds/worker/checkouts
VOLUME /builds/worker/.cache

USER root
