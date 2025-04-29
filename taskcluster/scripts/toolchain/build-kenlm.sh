#!/bin/bash
set -e
set -x

KENLM_DIR=$MOZ_FETCHES_DIR/kenlm-source

cd $KENLM_DIR
MAX_ORDER=7 python3 setup.py bdist_wheel
find .
cp $KENLM_DIR/dist/kenlm-0.0.0-cp310-cp310-linux_x86_64.whl $UPLOAD_DIR/
