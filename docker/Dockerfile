# A Docker image to run tests or experiment locally on a CPU machine
# It should be based on /taskcluster/docker/test/Dockerfile

ARG DOCKER_IMAGE_PARENT
FROM $DOCKER_IMAGE_PARENT

# Similar to /taskcluster/docker/toolchain/Dockerfile
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
                          bc \
                          libopenblas-dev \
                          openssl \
                          libssl-dev  \
                          pkg-config \
                          libre2-dev \
                          libglib2.0-dev \
                          python3-pybind11  \
                          gdb \
                          valgrind \
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

RUN mkdir /builds/worker/tools && \
    chown worker:worker /builds/worker/tools && \
    mkdir /builds/worker/tools/bin && \
    chown worker:worker /builds/worker/tools/bin

WORKDIR /builds/worker/tools

ADD pipeline/setup/* .

ENV BIN=/builds/worker/tools/bin

RUN git clone https://github.com/marian-nmt/extract-lex.git extract-lex
RUN ./compile-extract-lex.sh extract-lex/build $(nproc)

RUN git clone https://github.com/clab/fast_align.git fast_align
RUN ./compile-fast-align.sh fast_align/build $(nproc)

RUN git clone https://github.com/kpu/preprocess.git preprocess
RUN ./compile-preprocess.sh preprocess/build $(nproc)

RUN git clone https://github.com/marian-nmt/marian-dev.git marian-dev
# Use the same revision as in taskcluster/kinds/fetch/toolchains.yml
# it corresponds to v1.12.14 2d067afb 2024-02-16 11:44:13 -0500
RUN cd marian-dev && git checkout 2d067afb9ce5e3a0b6c32585706affc6e7295920
RUN ./compile-marian.sh marian-dev/build $(nproc) false

ENV MARIAN=/builds/worker/tools/marian-dev/build

# Work around poetry not finding `python`.
# https://github.com/python-poetry/poetry/issues/6371
RUN ln -s /bin/python3 /bin/python

# Have poetry ignore any venvs in the workdir by using a system install.
RUN poetry config virtualenvs.create false

# Install taskfile - https://taskfile.dev/
# Keep the version in sync with taskcluster/docker/test/Dockerfile.
RUN curl -sSLf "https://github.com/go-task/task/releases/download/v3.35.1/task_linux_amd64.tar.gz" \
    | tar -xz -C /usr/local/bin

# In some operating systems, the .git configuration will complain if the users
# do not match for the .git. Since this is a local environment, we don't worry about
# permissions escalation exploits that may be an issue in a production docker container.
RUN git config --global --add safe.directory /builds/worker/checkouts

# Allow scripts to detect if they are running in docker
ENV IS_DOCKER 1
