#!/bin/bash
##
# Basic cleaning of monolingual corpora.
#
# This script takes in a an archive file, e.g. /builds/worker/artifacts/news_2007.en.zst
# and rewrites in place using a variety of cleaning rules including:
#
#  - De-escape special characters.
#  - Remove non-printing characters.
#  - Filter by language detection (via fastText)

set -x
set -euo pipefail

echo "###### Cleaning monolingual data"

#                   Example inputs:
lang=$1             # en
input_prefix=$2     # $MOZ_FETCHES_DIR/news_2007
output_prefix=$3    # /builds/worker/artifacts/news_2007
threads=$4          # auto
dataset=$5          # news-crawl_news.2007
fluency_threshold=$6 # 0.7

# Example output: /builds/worker/artifacts/news_2007.en.zst

echo "### Cleaning ${input_prefix}"

if [ "$threads" = "auto" ]; then
  threads=$(nproc)
fi
cd "$(dirname "${0}")"
export PYTHONPATH="${PYTHONPATH}:tools"

dir="$(dirname "${output_prefix}")"
mkdir -p "${dir}"

######################################################################
echo "### Basic preprocessing from moses"
test -s "${output_prefix}.${lang}.nrm.zst" ||
  zstdmt -dc "${input_prefix}.${lang}.zst" |
  parallel --no-notice --pipe -k -j "${threads}" --block 50M \
    "perl tools/deescape-special-chars.perl | perl tools/remove-non-printing-char.perl" |
  zstdmt -c >"${output_prefix}.${lang}.nrm.zst"

######################################################################
echo "### Filter by language identification"
test -s "${output_prefix}.${lang}.langid.zst" ||
  zstdmt -dc "${output_prefix}.${lang}.nrm.zst" |
  # memory intensive
  parallel --no-notice --pipe -k -j "$(echo "${threads}"/4 | bc)" --block 50M "python3 tools/langid_fasttext.py -l ${lang}" |
  zstdmt >"${output_prefix}.${lang}.langid.zst"

######################################################################
echo "### Rule-based filtering"

zstdmt -dc "${output_prefix}.${lang}.langid.zst" |
parallel --no-notice --pipe -k -j "${threads}" --block 50M \
  "python3 tools/clean_mono.py -l ${lang} --debug" \
  2>"${output_prefix}.${lang}.clean.debug.txt" |
zstdmt >"${output_prefix}.${lang}.rule-based.zst"

test -s "${output_prefix}.${lang}.rule-based.zst" || exit 1

######################################################################
echo "### Filter by fluency score"

if [ "${fluency_threshold}" == "0" ] || [ "${fluency_threshold}" == "0.0" ]; then
  echo "Threshold is 0, skipping filtering"
  cp "${output_prefix}.${lang}.rule-based.zst" "${output_prefix}.${lang}.zst"
else
  # the model is 125MB, similar in size to the fastText one, so it's ok to download it here
  monocleaner-download $lang ${dir}/monocleaner
  test -s "${output_prefix}.${lang}.zst" ||
    zstd -dc "${output_prefix}.${lang}.rule-based.zst" |
    # memory intensive
    parallel --no-notice --pipe -k -j "$(echo "${threads}"/4 | bc)" --block 50M "monocleaner --disable_hardrules --disable_lang_ident ${dir}/monocleaner/${lang}" |
    awk -F'\t' '$2>'${fluency_threshold} | cut -f1 |
    zstdmt >"${output_prefix}.${lang}.zst"

  test -s "${output_prefix}.${lang}.zst" || exit 1
fi
echo "Lines before filtering: $(zstdmt -dc "${input_prefix}.${lang}.zst" | wc -l)"
echo "Lines after rule-based filtering: $(zstdmt -dc "${output_prefix}.${lang}.rule-based.zst" | wc -l)"
echo "Lines after fluency filtering: $(zstdmt -dc "${output_prefix}.${lang}.zst" | wc -l)"

######################################################################
echo "### Remove data from intermediate steps"
rm -rf "${output_prefix}".*.nrm.zst "${output_prefix}".*.langid.zst \
   "${output_prefix}".*.rule-based.zst ${dir}/monocleaner

echo "### Rule-based cleaning log written to: ${output_prefix}.${lang}.clean.debug.txt"
echo "### Clean data is written to: ${output_prefix}.${lang}.zst"

echo "###### Done: Cleaning monolingual data"
