#!/bin/bash
##
# Installs and compiles marian
#

set -x
set -euo pipefail

echo "###### Compiling marian"

marian_dir=$1
threads=${2}
use_gpu=${3:-true}
extra_args=( "${@:4}" )

mkdir -p "${marian_dir}"
cd "${marian_dir}"

if [ "${use_gpu}" == "true" ]; then
  # this is a production version that runs on GPU
  test -v CUDA_DIR
  cmake .. -DUSE_SENTENCEPIECE=on -DUSE_FBGEMM=on -DCOMPILE_CPU=on -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_DIR}" "${extra_args[@]}"
else
  # this is a CPU version that we use for testing
  cmake .. -DUSE_SENTENCEPIECE=on -DUSE_FBGEMM=on -DCOMPILE_CPU=on -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCOMPILE_CUDA=off -DCOMPILE_SERVER=on "${extra_args[@]}"
fi

make -j "${threads}"

echo "###### Done: Compiling marian"
