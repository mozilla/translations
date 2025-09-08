#!/bin/bash

set -x
set -euo pipefail

pushd `dirname $0`/../../.. &>/dev/null
VCS_ROOT=$(pwd)
popd &>/dev/null

if [ "$#" -lt 10 ]; then
    echo "Usage: $0 <model_type> <training_type> <src_locale> <trg_locale> <train_set_prefix> <validation_set_prefix> <artifacts> <best_model_metric> <alignments> <pretrained_model_mode> <pretrained_model_type> <pretrained_model_prefix> [extra_marian_args...]"
    exit 1
fi

model_type=$1
training_type=$2
src=$3
trg=$4
train_set_prefixes=$5
validation_set_prefix=$6
artifacts=$7
best_model_metric=$8
alignments=$9
seed=${10}
teacher_mode=${11}
student_model=${12}
pretrained_model_mode=${13}
pretrained_model_type=${14}
pretrained_url_prefix=${15}
extra_marian_args=( "${@:16}" )


[[ -v MOZ_FETCHES_DIR ]] || { echo "MOZ_FETCHES_DIR is not set"; exit 1; }

# This replaces the broken training continuation feature that mounted existing models but doesn't work under Docker
# We should unify all the code related to pretrained models and leave only one train.py eventually
if [[ "$pretrained_model_mode" == "init" || "$pretrained_model_mode" == "use" ]]; then
  python3 $VCS_ROOT/pipeline/continuation/model.py \
      --src_locale $src \
      --trg_locale $trg \
      --url_prefix "$pretrained_url_prefix" \
      --best_model $best_model_metric \
      --artifacts $TASK_WORKDIR/artifacts
fi

case "$pretrained_model_mode" in
    "use")
        echo "The training mode is 'use', using existing model without further training."
        if [ -f "$TASK_WORKDIR/artifacts/vocab.spm" ]; then
            # copy the shared vocab to two separate ones expected by the pipeline
            cp "$TASK_WORKDIR/artifacts/vocab.spm" "$TASK_WORKDIR/artifacts/vocab.$src.spm"
            mv "$TASK_WORKDIR/artifacts/vocab.spm" "$TASK_WORKDIR/artifacts/vocab.$trg.spm"
        fi
        exit 0
        ;;
    "continue"|"init"|"None")
        if [ "$pretrained_model_mode" == "None" ]; then
            # In any non-pretrained mode this file is pulled from an upstream
            # task. We copy it over to the artifacts directory earlier to
            # ensure that it is published even if the task is interrupted
            # (eg: by a spot termination in GCP). This makes resuming training
            # easier.
            mkdir -p "$TASK_WORKDIR/artifacts"
            cp "$MOZ_FETCHES_DIR/vocab.$src.spm" "$TASK_WORKDIR/artifacts/vocab.$src.spm"
            cp "$MOZ_FETCHES_DIR/vocab.$trg.spm" "$TASK_WORKDIR/artifacts/vocab.$trg.spm"
        fi

        if [ "$pretrained_model_mode" == "init" ]; then
            extra_marian_args+=("--pretrained-model" "$TASK_WORKDIR/artifacts/final.model.npz.best-$best_model_metric.npz")
        fi
        python3 $VCS_ROOT/pipeline/train/train.py \
        --model_type "$model_type" \
        --student_model "$student_model" \
        --training_type "$training_type" \
        --src "$src" \
        --trg "$trg" \
        --train_set_prefixes "$train_set_prefixes" \
        --validation_set_prefix "$validation_set_prefix" \
        --artifacts "$artifacts" \
        --src_vocab "$TASK_WORKDIR/artifacts/vocab.$src.spm" \
        --trg_vocab "$TASK_WORKDIR/artifacts/vocab.$trg.spm" \
        --best_model_metric "$best_model_metric" \
        --alignments "$alignments" \
        --seed "$seed" \
        --teacher_mode "$teacher_mode" \
        --gpus "$GPUS" \
        --marian_dir "$MARIAN" \
        --workspace "$WORKSPACE" \
        -- "${extra_marian_args[@]}"
        ;;
esac
