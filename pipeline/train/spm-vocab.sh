#!/bin/bash
##
# Train the SentencePiece vocabulary model. This outputs a ".spm" binary file, and the
# ".vocab" file which is a human readable list of the vocabulary. The vocab file is
# what is used to tokenize text input for the machine learning model. The vocab that
# is generated is a mix of the source and target languages.
#
# Docs:
#   docs/vocab-size.md
#
# Kinds:
#   taskcluster/ci/build-vocab/kind.yml
#
# Example usage:
#
#   export MARIAN=$MOZ_FETCHES_DIR                 && \
#   spm-vocab.sh                                      \
#       fetches/corpus.en.zst  `# merged_corpus_src`  \
#       fetches/corpus.ca.zst  `# merged_corpus_trg`  \
#       artifacts/vocab.en.spm `# vocab_src_output`   \
#       artifacts/vocab.zh.spm `# vocab_trg_output`   \
#       10000000               `# sample_size`        \
#       auto                   `# threads`            \
#       true                                          \
#       nmt_nfkc.tsv           `# normalization rule  \
#       32000                  `# vocab_size`

set -x
set -euo pipefail

if [[ -z "${MARIAN}" ]]; then
    echo "Error: The MARIAN environment variable was not provided. This is required as"
    echo "the path to the spm_train binary."
    exit 1
fi

# The name of the source corpus, e.g. "fetches/corpus.en.zst".
merged_corpus_src=$1
# The name of the target corpus, e.g. "fetches/corpus.ca.zst".
merged_corpus_trg=$2
# Where the src vocab file will be output, e.g. "artifacts/vocab.en.spm"
vocab_src_output=$3
# Where the trg vocab file will be output, e.g. "artifacts/vocab.zh.spm"
vocab_trg_output=$4
# The maximum number of sentences to train on, e.g. 10000000
sample_size=$5
# The thread count, either "auto" or an int.
num_threads=$6
# Whether to separate SentencePiece vocabularies for source and target languages ("true" or "false")
vocab_split=$7
# Unicode normalization rule to apply
norm_rule_tsv=$8
# The size of the final vocab. Defaults to 32000.
vocab_size=${9:-None}

case "$norm_rule_tsv" in
  *nmt_nfkc.tsv) ;;
  *nfkc.tsv) ;;
  *nfc.tsv) ;;
  *nfd.tsv) ;;
  *nfkc_translit_hbs.tsv) ;;
  None) norm_rule_tsv=nmt_nfkc.tsv ;;
  *)
    echo "Error normalization rule '$norm_rule_tsv' not found" >&2
    exit 1
    ;;
esac

if [ "$vocab_size" == "None" ]; then
  vocab_size=32000
fi

if (( vocab_size % 8 != 0 )); then
  echo "Error: vocab_size must be a multiple of 8 (https://github.com/mozilla/translations/issues/249)"
  exit 1
fi

if [ "$num_threads" = "auto" ]; then
  num_threads=$(nproc)
fi

vocab_dir=$(dirname "${vocab_src_output}")
mkdir -p "${vocab_dir}"
user_symbols="
__source__,__target__,__done__,__start__,__end__,__sep__,
__misc0__,__misc1__,__misc2__,__misc3__,__misc4__,
__misc5__,__misc6__,__misc7__,__misc8__,__misc9__,
"
user_symbols=$(echo "$user_symbols" | tr -d '\n')

if [ "$vocab_split" == "true" ] || [ "$vocab_split" == "True" ]; then
  # Sample the input corpus to at most 10M sentences to avoid running out of disk space
  # that's the amount SP uses, so it doesn't make sense to save in disk more than that
  # after that, remove the tab separator
  paste \
    <(zstdmt -dc "${merged_corpus_src}") \
    <(zstdmt -dc "${merged_corpus_trg}") \
    | shuf -n "${sample_size}" \
    | tee >(cut -f1 >${vocab_dir}/data.src.txt) \
    | cut -f2 \
  >${vocab_dir}/data.trg.txt \

  # The input arguments are available here:
  #   https://github.com/google/sentencepiece/blob/master/doc/options.md
  #
  # https://github.com/hplt-project/OpusTrainer/tree/main#generating-vocabulary-and-tags-before-training
  # byte_fallback - decomposes unknown pieces into UTF-8 bytes
  # user_defined_symbols - placeholders
  "${MARIAN}/spm_train" \
    --bos_id=-1 \
    --eos_id=0 \
    --unk_id=1 \
    --user_defined_symbols=$user_symbols \
    --model_prefix="${vocab_dir}/vocab.src" \
    --vocab_size="${vocab_size}" \
    --input="${vocab_dir}/data.src.txt" \
    --input_sentence_size="${sample_size}" \
    --normalization_rule_tsv="${norm_rule_tsv}" \
    --byte_fallback \
    --split_digits \
    --num_threads "${num_threads}"

  "${MARIAN}/spm_train" \
    --bos_id=-1 \
    --eos_id=0 \
    --unk_id=1 \
    --user_defined_symbols=$user_symbols \
    --model_prefix="${vocab_dir}/vocab.trg" \
    --vocab_size="${vocab_size}" \
    --input="${vocab_dir}/data.trg.txt" \
    --input_sentence_size="${sample_size}" \
    --normalization_rule_tsv="${norm_rule_tsv}" \
    --byte_fallback \
    --split_digits \
    --num_threads "${num_threads}"

    mv "${vocab_dir}/vocab.src.model" "${vocab_src_output}"
    mv "${vocab_dir}/vocab.trg.model" "${vocab_trg_output}"
    rm "${vocab_dir}/data.src.txt" "${vocab_dir}/data.trg.txt"
else
  # Sample the input corpus to at most 10M sentences to avoid running out of disk space
  # that's the amount SP uses, so it doesn't make sense to save in disk more than that
  # after that, remove the tab separator
  paste -d'\n' \
    <(zstdmt -dc "${merged_corpus_src}") \
    <(zstdmt -dc "${merged_corpus_trg}") \
    | shuf -n "${sample_size}" \
  >${vocab_dir}/data.txt

  "${MARIAN}/spm_train" \
    --bos_id=-1 \
    --eos_id=0 \
    --unk_id=1 \
    --user_defined_symbols=$user_symbols \
    --model_prefix="${vocab_dir}/vocab" \
    --vocab_size="${vocab_size}" \
    --input="${vocab_dir}/data.txt" \
    --input_sentence_size="${sample_size}" \
    --normalization_rule_tsv="${norm_rule_tsv}" \
    --byte_fallback \
    --split_digits \
    --num_threads "${num_threads}"

    cp "${vocab_dir}/vocab.model" "${vocab_src_output}"
    mv "${vocab_dir}/vocab.model" "${vocab_trg_output}"
    rm "${vocab_dir}/data.txt"
fi
