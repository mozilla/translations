ARG DOCKER_IMAGE_PARENT
FROM $DOCKER_IMAGE_PARENT
LABEL maintainer="Mozilla Release Engineering <release+docker@mozilla.com>"

RUN apt-get update -qq \
    && apt-get install -y \
      curl \
      wget \
      parallel \
      pigz \
      unzip \
      zstd \
      bc \
      libhunspell-1.7-0 \
      libboost-program-options1.74.0 \
      libboost-filesystem1.74.0 \
      libboost-iostreams1.74.0 \
      python3.10-venv \
    && apt-get clean


RUN pip install poetry==1.8.4
# we do not run poetry install here because the tracking package is installed from code

# Install taskfile - https://taskfile.dev/
# Keep the version in sync with docker/Dockerfile.
RUN curl -sSLf "https://github.com/go-task/task/releases/download/v3.35.1/task_linux_amd64.tar.gz" \
    | tar -xz -C /usr/local/bin

VOLUME /builds/worker/checkouts
VOLUME /builds/worker/.task-cache/pip
