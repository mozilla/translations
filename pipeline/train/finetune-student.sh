#!/bin/bash -v
##
# Finetune a student model.
#
# Usage:
#   bash finetune-student.sh
#

set -x
set -euo pipefail

bash ./train.sh \
  ${WORKDIR}/pipeline/train/configs/model/student.tiny11.yml \
  ${WORKDIR}/pipeline/train/configs/training/student.finetune.yml \
  $SRC \
  $TRG \
  ${DATA_DIR}/final/corpus \
  ${DATA_DIR}/original/devset \
  ${MODELS_DIR}/$SRC-$TRG/student \
  --guided-alignment ${DATA_DIR}/alignment/corpus.aln.gz


