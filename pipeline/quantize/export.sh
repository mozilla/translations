#!/bin/bash
##
# Export the quantized model to bergamot translator format.
#
# This script requires the browsermt fork of Marian for the int8shiftAlphaAll mode.
# https://github.com/browsermt/marian-dev
# https://github.com/browsermt/students/tree/master/train-student#5-8-bit-quantization

set -x
set -euo pipefail

echo "###### Exporting a quantized model"

test -v SRC
test -v TRG
test -v BMT_MARIAN

model_dir=$1
shortlist=$2
vocab_src=$3
vocab_trg=$4
output_dir=$5

mkdir -p "${output_dir}"

model="${output_dir}/model.${SRC}${TRG}.intgemm.alphas.bin"
cp "${model_dir}/model.intgemm.alphas.bin" "${model}"
pigz "${model}"

shortlist_bin="${output_dir}/lex.50.50.${SRC}${TRG}.s2t.bin"
"${BMT_MARIAN}"/marian-conv \
  --shortlist "${shortlist}" 50 50 0 \
  --dump "${shortlist_bin}" \
  --vocabs "${vocab_src}" "${vocab_trg}"
pigz "${shortlist_bin}"

if cmp --silent "${vocab_src}" "${vocab_trg}"; then
  echo "Vocab files are identical, output a joint vocab"
  vocab_out="${output_dir}/vocab.${SRC}${TRG}.spm"
  cp "${vocab_src}" "${vocab_out}"
  pigz "${vocab_out}"
else
  vocab_src_out="${output_dir}/srcvocab.${SRC}${TRG}.spm"
  vocab_trg_out="${output_dir}/trgvocab.${SRC}${TRG}.spm"
  cp "${vocab_src}" "${vocab_src_out}"
  cp "${vocab_trg}" "${vocab_trg_out}"
  pigz "${vocab_src_out}"
  pigz "${vocab_trg_out}"
fi

echo "### Export is completed. Results: ${output_dir}"

echo "###### Done: Exporting a quantized model"
