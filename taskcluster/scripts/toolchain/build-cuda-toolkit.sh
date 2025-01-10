#!/bin/bash
set -e
set -x

# CUDA installers do not have a silent mode of operation that allows to run
# them without also installing GPU drivers and other unnecessary things.
# Instead, we extract the raw contents of the installer, and then combine
# the extracted contents into a tarball.

export CUDA_INSTALLER=$MOZ_FETCHES_DIR/cuda-source.run

TARFILE=$UPLOAD_DIR/cuda-toolkit.tar

chmod +x $CUDA_INSTALLER
# This installer advertises a `--extract` option which put
# the contents in a directory of our choosing...but it doesn't
# work when run under alpine linux docker containers. Instead,
# we can use these secret options, which will extract to `pkg`
# in the current working directory. The files we care about
# will end up in `pkg/builds`.
EXTRACT_DIR="$(pwd)/cuda-toolkit"

# On the off chance that two different cuda-toolkit toolchain tasks run
# on the same worker, the second one will end up with an existing toolkit
# as a starting point, which will end up packaging both versions of the
# toolkit at in the subsequent tarball. This confuses downstream tasks, which
# may pick up the wrong version of the toolkit.
rm -rf $EXTRACT_DIR

# it complains on compiler version check on Ubuntu 22 for cuda toolkit 11.2. overriding helps
$CUDA_INSTALLER --toolkit --toolkitpath=$EXTRACT_DIR --silent --override

tar --zstd -cf $TARFILE.zst cuda-toolkit
