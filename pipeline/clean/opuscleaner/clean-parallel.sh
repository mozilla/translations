#!/bin/bash
##
# Cleaning parallel corpora with OpusCleaner
#

set -x
set -euo pipefail

echo "###### Cleaning corpus with OpusCleaner"

test -v SRC
test -v TRG

input_prefix=$1
output_prefix=$2
threads=$3
dataset=$4
mode=$5

if [ "$threads" = "auto" ]; then
  threads=$(nproc)
fi

cd "$(dirname "${0}")"
dir="$(dirname "${output_prefix}")"
mkdir -p "${dir}"

# pre-download fast text model as it's causing constant issues
filters_dir="/builds/worker/.local/lib/python3.10/site-packages/opuscleaner/filters"
if [ -d ${filters_dir} ]; then
  echo "Downloading FastText model."
  # Download both models for retrocompatibility
  test -s "${filters_dir}/large.bin" || wget -q -O "${filters_dir}/large.bin" https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
  test -s "${filters_dir}/nllb.bin" || wget -q -O "${filters_dir}/nllb.bin" https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin
  test -s "${filters_dir}/openlid-v2.bin" || wget -q -O "${filters_dir}/openlid-v2.bin" https://huggingface.co/laurievb/OpenLID-v2/resolve/main/model.bin
fi

echo "### Generating cleaning config: ${dataset}.${SRC}-${TRG}.filters.json"
# save new filter to dataset output dir
filter_path="${output_prefix}.${SRC}-${TRG}.filters.json"
python3 generate_filters.py "${input_prefix}" "${SRC}" "${TRG}" "${dataset}" "${filter_path}" "${mode}"
test -s "${filter_path}" || exit 1

echo "### Cleaning ${input_prefix} with filter ${filter_path}"
src_icu="$(python3 -c "from pipeline.langs.codes import icu_normalize; print(icu_normalize('${SRC}'))")"
trg_icu="$(python3 -c "from pipeline.langs.codes import icu_normalize; print(icu_normalize('${TRG}'))")"
# Clean tabs before feeding into opuscleaner
paste <(zstdmt -dc "${input_prefix}.${SRC}.zst" | sed 's|\t| |g') \
      <(zstdmt -dc "${input_prefix}.${TRG}.zst" | sed 's|\t| |g') |
opuscleaner-clean \
  --parallel ${threads} \
  --batch-size=50000 \
  --input=- \
  "${filter_path}" "${src_icu}" "${trg_icu}" |
  tee >(cut -f1 | zstdmt >"${output_prefix}.${SRC}.zst") |
        cut -f2 | zstdmt >"${output_prefix}.${TRG}.zst"

echo "### Checking length of the files"
test -s "${output_prefix}.${SRC}.zst" || exit 1
test -s "${output_prefix}.${TRG}.zst" || exit 1
new_len_src="$(zstdmt -dc "${output_prefix}.${SRC}.zst" | wc -l)"
new_len_trg="$(zstdmt -dc "${output_prefix}.${TRG}.zst" | wc -l)"
orig_len_src="$(zstdmt -dc "${input_prefix}.${SRC}.zst" | wc -l)"
[[ ${new_len_src} -ge 1 ]] || exit 1
[[ ${new_len_trg} -ge 1 ]] || exit 1
[[ "${new_len_src}" = "${new_len_trg}" ]] || exit 1
echo "### Filtered length: ${new_len_src} / ${orig_len_src}"

echo "### Clean ${input_prefix} is written to  ${output_prefix}"

echo "###### Done: Cleaning corpus with OpusCleaner"
