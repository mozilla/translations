#!/bin/bash
##
# Installs and compiles alignment tools
#
# Usage:
#   bash compile-extract-lex.sh $(nproc)
#

set -x
set -euo pipefail

threads=$1

echo "###### Compiling extract-lex"

mkdir -p "${BIN}"

if [ ! -e "${WORKDIR}/bin/extract_lex" ]; then
  echo "### Compiling extract-lex"
  mkdir -p "${WORKDIR}/3rd_party/extract-lex/build"
  cd "${WORKDIR}/3rd_party/extract-lex/build"
  cmake ..
  make -j "${threads}"
  cp extract_lex "${BIN}"
fi

echo "###### Done: Compiling extract-lex"
