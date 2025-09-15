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

RUN pip install zstandard

# %include-run-task

# Allow scripts to detect if they are running in docker
ENV IS_DOCKER=1

ENV SHELL=/bin/bash \
    HOME=/builds/worker \
    PATH="/builds/worker/.local/bin:$PATH" \
    TERM=hterm-256color

VOLUME /builds/worker/checkouts
VOLUME /builds/worker/.task-cache/pip

USER root
