#!/bin/bash
##
# Filters student parallel corpus with scores produced by a reversed NMT model.
#

set -x
set -euo pipefail

echo "###### Cross entropy filtering"
test -v SRC
test -v TRG

corpus_prefix=$1
output_prefix=$2
scores=$3

cd "$(dirname "${0}")"

# Part of the data to be removed (0.05 is 5%)
remove=0.05
output_dir=$(dirname "${output_prefix}")
tmp="${output_dir}/tmp"
mkdir -p "${tmp}"

echo "### Sorting scores"
if [ ! -s "${tmp}/sorted.zst" ]; then
  buffer_size="$(echo "$(grep MemTotal /proc/meminfo | awk '{print $2}')"*0.7 | bc | cut -f1 -d.)"
  paste "${scores}" <(zstdmt -dc "${corpus_prefix}.${SRC}.zst") <(zstdmt -dc "${corpus_prefix}.${TRG}.zst") |
  LC_ALL=C sort -n -k1,1 -S "${buffer_size}K" -T "${tmp}" |
  zstdmt >"${tmp}/sorted.zst"
fi

echo "### Cutting the best scored corpus"
if [ ! -s "${tmp}/best.zst" ]; then
  lines=$(zstdmt -dc "${tmp}/sorted.zst" | wc -l)
  startline=$(echo ${lines}*${remove} | bc | cut -f1 -d.)
  zstdmt -dc "${tmp}/sorted.zst" | tail -n +${startline} | cut -f2,3 | zstdmt >"${tmp}/best.zst"
fi

echo "### Writing output corpus"
zstdmt -dc "${tmp}/best.zst" |
  tee >(cut -f1 | zstdmt >"${output_prefix}.${SRC}.zst") |
  cut -f2 | zstdmt >"${output_prefix}.${TRG}.zst"

echo "### Deleting tmp dir"
rm -rf "${tmp}"

echo "###### Done: Cross entropy filtering"
