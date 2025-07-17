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
                          libicu-dev \
    && apt-get clean

VOLUME /builds/worker/checkouts
VOLUME /builds/worker/.task-cache/pip
