FROM $DOCKER_IMAGE_PARENT
LABEL maintainer="Mozilla Release Engineering <release+docker@mozilla.com>"

RUN curl -L https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb > /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb

RUN apt-get update -qq \
    && apt-get install -y python3-numpy \
                          python3-fasttext \
                          parallel \
                          zstd \
                          bc \
                          libhunspell-1.7-0 \
                          libboost-program-options1.74.0 \
                          libboost-filesystem1.74.0 \
                          libboost-iostreams1.74.0 \
                          pigz \
                          curl \
                          wget \
                          cuda-toolkit \
                          software-properties-common \
    && apt-get clean

RUN curl -L https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin > /tmp/cuda.pin \
    && mv /tmp/cuda.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
    && apt-get update -qq \
    && apt-get install -y cudnn9-cuda-12 \
                          libcudnn9-dev-cuda-12 \
    && apt-get clean


VOLUME /builds/worker/checkouts
VOLUME /builds/worker/.cache
