FROM $DOCKER_IMAGE_PARENT
LABEL maintainer="Mozilla Release Engineering <release+docker@mozilla.com>"

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
                          pkg-config \
                          libicu-dev \
    && apt-get clean


ENV DEBIAN_FRONTEND=noninteractive
RUN curl -sSO https://developer.download.nvidia.com/compute/nvidia-driver/565.57.01/local_installers/nvidia-driver-local-repo-ubuntu2204-565.57.01_1.0-1_amd64.deb \
    && dpkg -i nvidia-driver-local-repo-ubuntu2204-565.57.01_1.0-1_amd64.deb \
    && cp /var/nvidia-driver-local-repo-ubuntu2204-565.57.01/nvidia-driver-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update -qq \
    && apt-get install -y cuda-drivers-565

VOLUME /builds/worker/checkouts
VOLUME /builds/worker/.cache
