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

RUN pip3 install sacrebleu[ja,ko]==2.4.2 \
    unbabel-comet==2.2.2 \
    datasets \
    toolz \
    google-cloud-translate==3.2.1 \
    requests \
    toolz \
    tqdm \
    pandas \
    argostranslate==1.9.1 \
    json5==0.12.0 \
    openai==1.93.0 \
    taskcluster==88.0.2 \
    PyYAML

VOLUME /builds/worker/checkouts
VOLUME /builds/worker/.task-cache/pip
