#!/bin/bash
#
# Collects chunked translation data of the form "file.N.out" where N is a number.
# The datasets are chunked earlier in the pipeline by split-{corpus,mono}.sh so that
# tasks can work on smaller sets of data to better parallelize the work. After processing,
# any chunked data is reassembled with this script.
#
# Example tasks running on chunked data (before this script):
#   extract-best-en-ca-1/10
#   distillation-parallel-src-translate-en-ca-1/10
#   translate-mono-src-en-ca-1/10
#   translate-mono-trg-en-ca-1/10
#
# Kinds:
#   taskcluster/ci/backtranslations-mono-trg-dechunk-translations/kind.yml
#   taskcluster/ci/distillation-mono-src-dechunk-translations/kind.yml
#   taskcluster/ci/distillation-parallel-src-dechunk-translations/kind.yml
#
# Example usage:
#
#   pipeline/translate/collect.sh    \
#      fetches                       \
#      artifacts/mono.en.zst         \
#      $MOZ_FETCHES_DIR/mono.ca.zst

set -x
set -euo pipefail

# The directory with chunks of the form "fetches/file.N.out", where N is a number.
chunks_dir=$1
# The full file name path to the output compressed file, e.g. "artifacts/mono.en.zst"
output_path=$2
# The path to the monolingual data to compare against, e.g. "$MOZ_FETCHES_DIR/mono.hu.zst"
mono_path=$3


echo "### Collecting translations"

find "${chunks_dir}" -name '*.out' |  # For example, finds "fetches/file.1.out", "fetches/file.2.out", etc.
  sort -t '.' -k2,2n |                      # Sort by the number in "file.1.out", e.g. 1 here.
  xargs cat |                               # Combine all of these files together.
  zstdmt >"${output_path}"

echo "### Comparing number of sentences in source and artificial target files"

src_len=$(zstdmt -dc "${mono_path}" | wc -l)
trg_len=$(zstdmt -dc "${output_path}" | wc -l)

if [ "${src_len}" != "${trg_len}" ]; then
  echo "### Error: length of ${mono_path} ${src_len} is different from ${output_path} ${trg_len}"
  exit 1
fi
