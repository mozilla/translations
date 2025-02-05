#!/bin/bash

if [[ ! -e ../models ]]; then
    mkdir -p ../models
    if [[ ! -e ../students ]]; then
        echo "Cloning https://github.com/browsermt/students)"
        git clone --depth 1 --branch main --single-branch https://github.com/browsermt/students ../
    fi
    
    echo "Downloading files"
    ../students/esen/download-models.sh

    echo "Copying downloaded files to models folder"
    cp ../students/esen/esen.student.tiny11/vocab.esen* ../students/esen/esen.student.tiny11/model* ../students/esen/esen.student.tiny11/lex.s2t* ../models/
    sacrebleu -t wmt13 -l es-en --echo src > ../models/newstest2013.es
    head -n300 ../models/newstest2013.es > ../models/newstest2013.es.top300lines
    gunzip ../models/*
else
    echo "models directory already exists in root folder; Using it to package files without downloading anything"
fi

echo "Packaging files for wasm binary"
$EMSDK_PYTHON $EMSDK/upstream/emscripten/tools/file_packager.py model-files.data --preload ../models/@ --js-output=model-files.js

echo "Importing integer (8-bit) gemm implementation"
SCRIPT_ABSOLUTE_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
sed -i.bak 's/"env"[[:space:]]*:[[:space:]]*asmLibraryArg,/"env": asmLibraryArg,\
    "wasm_gemm": createWasmGemm(),/g' marian-decoder.js
cat $SCRIPT_ABSOLUTE_PATH/import-gemm-module.js >> marian-decoder.js
echo "SUCCESS"
