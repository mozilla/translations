FROM ubuntu:22.04
LABEL maintainer="Mozilla Release Engineering <release+docker@mozilla.com>"

# Add worker user with the home directory /builds/worker and bash as the default shell.
# `/builds` may already exist, `-p` will ensure `mkdir` succeeds regardless
RUN mkdir -p /builds && \
    useradd -d /builds/worker -s /bin/bash -m worker && \
    chown worker:worker /builds/worker && \
    mkdir /builds/worker/artifacts && \
    chown worker:worker /builds/worker/artifacts

COPY known_hosts /etc/ssh/ssh_known_hosts

WORKDIR /builds/worker/

#----------------------------------------------------------------------------------------------------------------------
#-- Configuration -----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

ENV CURL='curl --location --retry 5' \
    LANG='en_US.UTF-8' \
    TERM='dumb'
    
RUN apt-get update -qq \
    # We need to install tzdata before all of the other packages. Otherwise it will show an interactive dialog that
    # we cannot navigate while building the Docker image.
    && apt-get install -y tzdata \
    && apt-get install -y python3 \
                          python3-pip \
                          python3-yaml \
                          locales \
                          git \
                          tmux \
                          htop \
                          vim \
                          nano \
                          ncurses-term \
                          zstd \
                          screen \
                          curl \
                          pkg-config \
                          libicu-dev \
    # Clean up the apt cache to reduce image size.
    && apt-get clean


RUN locale-gen "$LANG"

# uv is installed with pip instead of the recommended copy bin
# from the UV installation with docker page
# because that way uv is present in the python env as a module
# and can be run like /path/to/env/bin/python -m uv pip install ...
RUN pip install "uv==0.10.9"
RUN ln -s /usr/local/bin/uv /usr/local/bin/uvx /bin/
RUN uv pip install --system zstandard

# %include-run-task

# Allow scripts to detect if they are running in docker
ENV IS_DOCKER=1

ENV SHELL=/bin/bash \
    HOME=/builds/worker \
    PATH="/builds/worker/.local/bin:$PATH" \
    TERM=hterm-256color \
    UV_CACHE_DIR=/builds/worker/.task-cache/uv

VOLUME /builds/worker/checkouts
VOLUME /builds/worker/.task-cache/pip
VOLUME /builds/worker/.task-cache/uv

USER root
