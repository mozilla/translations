#!/bin/bash
##
# Generates a lexical shortlist for a corpus.
#
#
# It also generate SentencePiece tokenized alignments that are required for extract_lex
#

set -x
set -euo pipefail

echo "###### Generating alignments and shortlist"
[[ -z "${MARIAN}" ]] && echo "MARIAN is empty"
[[ -z "${BIN}" ]] && echo "BIN is empty"
[[ -z "${SRC}" ]] && echo "SRC is empty"
[[ -z "${TRG}" ]] && echo "TRG is empty"

corpus_prefix=$1
vocab_src=$2
vocab_trg=$3
output_dir=$4
threads=$5

if [ "$threads" = "auto" ]; then
  threads=$(nproc)
fi

cd "$(dirname "${0}")"

mkdir -p "${output_dir}"
dir="${output_dir}/tmp_shortlist"
mkdir -p "${dir}"

corpus_src="${corpus_prefix}.${SRC}.zst"
corpus_trg="${corpus_prefix}.${TRG}.zst"


echo "### Subword segmentation with SentencePiece"
zstdmt -dc "${corpus_src}" |
  parallel --no-notice --pipe -k -j "${threads}" --block 50M "${MARIAN}/spm_encode" --model "${vocab_src}" \
   >"${dir}/corpus.spm.${SRC}"

zstdmt -dc "${corpus_trg}" |
  parallel --no-notice --pipe -k -j "${threads}" --block 50M "${MARIAN}/spm_encode" --model "${vocab_trg}" \
   >"${dir}/corpus.spm.${TRG}"

python3 align.py \
  --corpus_src="${dir}/corpus.spm.${SRC}" \
  --corpus_trg="${dir}/corpus.spm.${TRG}" \
  --output_path="${output_dir}/corpus.aln"

echo "### Creating shortlist"
"${BIN}/extract_lex" \
  "${dir}/corpus.spm.${TRG}" \
  "${dir}/corpus.spm.${SRC}" \
  "${output_dir}/corpus.aln" \
  "${dir}/lex.s2t" \
  "${dir}/lex.t2s"

if [ -f "${dir}/lex.s2t" ]; then
  zstdmt "${dir}/lex.s2t"
fi

rm "${dir}/corpus.spm.${TRG}"
rm "${dir}/corpus.spm.${SRC}"
rm "${output_dir}/corpus.aln"

echo "### Shortlist pruning"
"${MARIAN}/spm_export_vocab" --model="${vocab_trg}" --output="${dir}/vocab.txt"
zstdmt -dc "${dir}/lex.s2t.zst" |
  grep -v NULL |
  python3 "prune_shortlist.py" 100 "${dir}/vocab.txt" |
  zstdmt >"${output_dir}/lex.s2t.pruned.zst"

echo "### Deleting tmp dir"
rm -rf "${dir}"

echo "###### Done: Generating alignments and shortlist"
