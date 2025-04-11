#!/bin/bash
##
# Basic cleaning of parallel corpora.
#

set -x
set -euo pipefail

echo "###### Cleaning parallel corpus"


test -v SRC
test -v TRG

input_prefix=$1
output_prefix=$2
threads=$3
dataset=$4

if [ "$threads" = "auto" ]; then
  threads=$(nproc)
fi
cd "$(dirname "${0}")"
export PYTHONPATH="tools"

dir="$(dirname "${output_prefix}")"
mkdir -p "${dir}"

echo "### Cleaning ${input_prefix}"

######################################################################
echo "### Basic preprocessing"
for lng in "${SRC}" "${TRG}"; do
  test -s "${output_prefix}.${lng}.nrm.zst" ||
    zstdmt -dc "${input_prefix}.${lng}.zst" |
    parallel --no-notice --pipe -k -j "${threads}" --block 50M \
      "perl tools/deescape-special-chars.perl | perl tools/remove-non-printing-char.perl" |
    zstdmt >"${output_prefix}.${lng}.nrm.zst"
done

#####################################################################
echo "### Apply monolingual fixes"
for lng in $SRC $TRG; do
    if [[ ! -x fixes/${dataset}.${lng}.sh ]]; then
      test -s "${output_prefix}.${lng}.monofix.zst" ||
        cp "${output_prefix}.${lng}.nrm.zst" "${output_prefix}.${lng}.monofix.zst"
    else
        test -s "${output_prefix}.${lng}.monofix.zst" ||
          zstdmt -dc "${output_prefix}.${lng}.nrm.zst" \
              | fixes/"${dataset}"."${lng}".sh \
              | zstdmt >"${output_prefix}.${lng}.monofix.zst"
    fi
done

######################################################################
echo "### Apply bilingual fixes"
if [[ -x fixes/${dataset}.sh ]]; then
    FIX="fixes/${dataset}.sh ${SRC} ${TRG} ${threads}"
else
    FIX="cat"
fi
test -s "${output_prefix}.${SRC}${TRG}.fix.zst" ||
  paste <(zstdmt -dc "${output_prefix}.${SRC}.monofix.zst") <(zstdmt -dc "${output_prefix}.${TRG}.monofix.zst") \
      | $FIX \
      | zstdmt > "${output_prefix}.${SRC}${TRG}.fix.zst"

######################################################################
echo "### Rule-based filtering"
test -s "${output_prefix}.${SRC}${TRG}.rule-based.zst" ||
  zstdmt -dc "${output_prefix}.${SRC}${TRG}.fix.zst" |
  parallel --no-notice --pipe -k -j "${threads}" --block 50M \
    "python3 tools/clean_parallel.py -l1 ${SRC} -l2 ${TRG} --debug" \
    2>"${output_prefix}.${SRC}${TRG}.clean.debug.txt" |
  zstdmt >"${output_prefix}.${SRC}${TRG}.rule-based.zst"

######################################################################
echo "### Language identification"
test -s "${output_prefix}.${SRC}${TRG}.langid.zst" ||
  # langid_fasttext.py will download this file if it is not already present. When it runs in
  # parallel, this will typically cause the file to be corrupt.
  test -s tools/lid.176.bin || wget -O tools/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
  zstdmt -dc "${output_prefix}.${SRC}${TRG}.rule-based.zst" |
  # memory intensive
  parallel --no-notice --pipe -k -j "$(echo "${threads}"/4 | bc)" --block 50M \
    "python3 -Wi tools/langid_fasttext.py -f 1 | python3 -Wi tools/langid_fasttext.py -f 1" |
  grep -P "^${SRC}\t${TRG}\t" |
  cut -f3,4 |
  zstdmt >"${output_prefix}.${SRC}${TRG}.langid.zst"

######################################################################
echo "### Removing leading and repetitive white spaces"

zstdmt -dc "${output_prefix}.${SRC}${TRG}.langid.zst" |
cut -f1 |
sed -e 's/^[[:space:]]*//' |
tr -s " " |
zstdmt >"${output_prefix}.${SRC}.zst"

zstdmt -dc "${output_prefix}.${SRC}${TRG}.langid.zst" |
cut -f2 |
sed -e 's/^[[:space:]]*//' |
tr -s " " |
zstdmt >"${output_prefix}.${TRG}.zst"

test -s "${output_prefix}.${SRC}.zst" || exit 1
test -s "${output_prefix}.${TRG}.zst" || exit 1

echo "### Remove input_prefix from intermediate steps"
rm -rf "${output_prefix}".*.nrm.zst "${output_prefix}".*.langid.zst \
  "${output_prefix}".*.rule-based.zst "${output_prefix}".*.*fix.zst

echo "### Clean ${input_prefix} is written to  ${output_prefix}"

echo "###### Done: Cleaning corpus"
