#!/bin/bash
##
# Runs quantization of the student model.
#

set -x
set -euo pipefail

echo "###### Quantizing a model"

test -v BMT_MARIAN
test -v BIN
test -v SRC
test -v TRG

model=$1
vocab_src=$2
vocab_trg=$3
shortlist=$4
devtest_src=$5
output_dir=$6

cd "$(dirname "${0}")"

res_model="${output_dir}/model.intgemm.alphas.bin"
mkdir -p "${output_dir}"
cp "${vocab_src}" "${output_dir}"
cp "${vocab_trg}" "${output_dir}"

echo "### Decoding a sample test set in order to get typical quantization values"
test -s "${output_dir}/quantmults" ||
  "${BMT_MARIAN}"/marian-decoder \
    --models "${model}" \
    --vocabs "${vocab_src}" "${vocab_trg}" \
    --config "decoder.yml" \
    --input "${devtest_src}" \
    --output "${output_dir}/output.${TRG}" \
    --shortlist "${shortlist}" false \
    --quiet \
    --quiet-translation \
    --log "${output_dir}/cpu.output.log" \
    --dump-quantmult \
    2>"${output_dir}/quantmults"

echo "### Quantizing"
test -s "${output_dir}/model.alphas.npz" ||
  python3 extract_stats.py \
    "${output_dir}/quantmults" \
    "${model}" \
    "${output_dir}/model.alphas.npz"

echo "### Converting"
test -s "${res_model}" ||
  "${BMT_MARIAN}"/marian-conv \
    --from "${output_dir}/model.alphas.npz" \
    --to "${res_model}" \
    --gemm-type intgemm8

echo "### The result models is saved to ${res_model}"

echo "###### Done: Quantizing a model"
