#!/bin/bash
##
# Cleans corpus using bicleaner-ai
#
# See:
#   docs/bicleaner.md

set -x
set -euo pipefail

echo "###### Bicleaner filtering"

echo "First check that GPUs are available."
python3 -c "from pipeline.common.marian import assert_gpus_available; assert_gpus_available()"

test -v SRC
test -v TRG
test -v CUDA_DIR

corpus_prefix=$1
output_prefix=$2
bicleaner_threshold=$3
threads=$4
pack_dir=$5

if [ "$threads" = "auto" ]; then
  threads=$(nproc)
fi

output_dir=$(dirname "${output_prefix}")
mkdir -p "${output_dir}"

if [ "${bicleaner_threshold}" == "0" ] || [ "${bicleaner_threshold}" == "0.0" ]; then
  echo "Threshold is 0, skipping filtering"
  cp "${corpus_prefix}.${SRC}.zst" "${output_prefix}.${SRC}.zst"
  cp "${corpus_prefix}.${TRG}.zst" "${output_prefix}.${TRG}.zst"
  # Create a dummy best-scores.zst, if no filtering everyone gets perfect score
  # this is needed for target side dedup in merge-parallel
  num_sents=$(zstdcat "${corpus_prefix}.${TRG}.zst" | wc -l)
  awk -v n=$num_sents 'BEGIN {for(i=0;i<n;i++) print "1.0";}' | zstdmt >"${output_prefix}.best-scores.zst"
else

  export scol=1
  export tcol=2
  # Get model src-trg from metadata.yaml
  model_source_lang=$(grep "source_lang" ${pack_dir}/*.yaml | awk '{print $2}')
  model_target_lang=$(grep "target_lang" ${pack_dir}/*.yaml | awk '{print $2}')
  # for example if SRC-TRG = en-ru
  # the model can be: en-ru, ru-en, en-xx
  src_mapped=$(python3 -c "from pipeline.langs.codes import to_iso6391; from pipeline.langs.maps import BICLEANER_AI_DEFAULTS_MAP; print(to_iso6391('${SRC}', BICLEANER_AI_DEFAULTS_MAP))")
  trg_mapped=$(python3 -c "from pipeline.langs.codes import to_iso6391; from pipeline.langs.maps import BICLEANER_AI_DEFAULTS_MAP; print(to_iso6391('${TRG}', BICLEANER_AI_DEFAULTS_MAP))")
  if [ ${model_source_lang} == ${trg_mapped} ] || [ ${model_target_lang} == ${src_mapped} ]; then
    # swap columns
    export scol=2
    export tcol=1
  fi

  #Export cuda visible devices if empty or not set
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
  fi

  echo "### Classifying"
  if [ ${#CUDA_VISIBLE_DEVICES} -gt 1 ]; then # Use gnu-parallel'd bicleaner-ai if we have more than 1 GPU
       #Convert CUDA_VISIBLE_DEVICES to an array
       export CUDA_VISIBLE_ARRAY=(${CUDA_VISIBLE_DEVICES//,/ })
       #Turn on tensorflow logging in bicleaner-ai
       export TF_CPP_MIN_LOG_LEVEL=0
       #This function expects a bicleaner yaml and a 1-based index into the CUDA_VISIBLE_ARRAY
       #Example: /mnt/nanna0/nbogoych/data/data/fr-en/fr-en-prod/biclean/pack/metadata.yaml index_in_CUDA_VISIBLE_ARRAY+1
       biclean() {
               export CUDA_VISIBLE_ARRAY=(${CUDA_VISIBLE_DEVICES//,/ })
               export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_ARRAY[$(($2-1))]}
               # The GPU devices have failed to be found, and bicleaner AI falls back
               # to operate on the CPU very slowly. To guard against this wasting expensive
               # GPU time, always check that it can find GPUs.
               python3 -c "import tensorflow; exit(0) if tensorflow.config.list_physical_devices('GPU') else exit(75)"
               bicleaner-ai-classify --disable_hardrules --require_gpu --scol ${scol} --tcol ${tcol} - - $1
       }
       export -f biclean
       # {%} is a 1-indexed job slot number from GNU parallel.  We use that as the 1-indexed offset in CUDA_VISIBLE_ARRAY
       paste <(zstdmt -dc "${corpus_prefix}.${SRC}.zst") <(zstdmt -dc "${corpus_prefix}.${TRG}.zst") |
       parallel -j ${#CUDA_VISIBLE_ARRAY[@]} --pipe -k --block 10M biclean "${pack_dir}"/*.yaml {%} |
       zstdmt >"${output_prefix}.scored.zst"
  else
   export BICLEANER_AI_THREADS=${threads}
   paste <(zstdmt -dc "${corpus_prefix}.${SRC}.zst") <(zstdmt -dc "${corpus_prefix}.${TRG}.zst") |
     bicleaner-ai-classify --disable_hardrules --scol ${scol} --tcol ${tcol} "${threads}"  - - "${pack_dir}"/*.yaml |
     zstdmt >"${output_prefix}.scored.zst"
  fi

  echo "### Filtering"
  zstdmt -dc "${output_prefix}.scored.zst" |
    awk -v threshold=${bicleaner_threshold} -F"\t" '{if ($3>threshold) {print $0}}' |
    zstdmt >"${output_prefix}.best.zst"

  zstdmt -dc "${output_prefix}.scored.zst" |
    awk -v threshold=${bicleaner_threshold} -F"\t" '{if ($3<=threshold) {print $0}}' |
    zstdmt >"${output_prefix}.filtered.zst"

  echo "Lines before filtering: $(zstdmt -dc "${output_prefix}.scored.zst" | wc -l)"
  echo "Lines after filtering: $(zstdmt -dc "${output_prefix}.best.zst" | wc -l)"

  echo "### Writing output corpus"
  zstdmt -dc "${output_prefix}.best.zst" |
    tee >(cut -f1 | zstdmt >"${output_prefix}.${SRC}.zst") |
    tee >(cut -f2 | zstdmt >"${output_prefix}.${TRG}.zst") |
    cut -f3 | zstdmt >"${output_prefix}.best-scores.zst"

  # do not delete intermediate files to inspect them and tune the threshold
fi

echo "###### Done: Bicleaner filtering"
