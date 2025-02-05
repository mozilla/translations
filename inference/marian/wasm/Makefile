# -*- mode: makefile-gmake; indent-tabs-mode: true; tab-width: 4 -*-
SHELL   		= bash
PWD     		= $(shell pwd)
WASM_IMAGE	    = local/marian-wasm-docker

# Build the Docker image for WASM builds
wasm-image:
	docker build -t local/marian-wasm-docker .

# Commands for native compilation:
cmake_native_decoder_cmd  = cmake --debug-output -Wno-dev
cmake_native_decoder_cmd += -DUSE_STATIC_LIBS=on
cmake_native_decoder_cmd += -DUSE_SENTENCEPIECE=on
cmake_native_decoder_cmd += -DCOMPILE_CUDA=off
cmake_native_decoder_cmd += -DUSE_DOXYGEN=off
cmake_native_decoder_cmd += -DUSE_FBGEMM=off
cmake_native_decoder_cmd += -DCOMPILE_LIBRARY_ONLY=off
cmake_native_decoder_cmd += -DUSE_MKL=off
cmake_native_decoder_cmd += -DCOMPILE_CPU=on
cmake_native_decoder_cmd += -DUSE_WASM_COMPATIBLE_SOURCE=on

# Commands for wasm compilation:
cmake_wasm_decoder_cmd = ${cmake_native_decoder_cmd}
cmake_wasm_decoder_cmd += -DCOMPILE_WASM=on

# Commands for running things on Docker
docker_mounts  = ${PWD}/..:/repo
docker_mounts += ${HOME}/.ccache:/.ccache
run_on_docker  = docker run --rm
run_on_docker += $(addprefix -v, ${docker_mounts})
run_on_docker += ${INTERACTIVE_DOCKER_SESSION}

${HOME}/.ccache:
	mkdir -p $@

############## Compiling wasm marian-decoder on Docker #################
# Remove the WASM build dir, forcing a clean compilation attempt
clean-wasm-docker: BUILD_DIR = /repo/build-wasm-docker
clean-wasm-docker: ${HOME}/.ccache
	${run_on_docker} ${WASM_IMAGE} bash -c '(rm -rf ${BUILD_DIR} || true)'

# Compile marian-decoder to WASM
compile-wasm-docker: BUILD_DIR = /repo/build-wasm-docker
compile-wasm-docker: ${HOME}/.ccache
	${run_on_docker} ${WASM_IMAGE} bash -c 'mkdir -p ${BUILD_DIR} && \
cd ${BUILD_DIR} && \
(emcmake ${cmake_wasm_decoder_cmd} .. && \
(emmake make -j4))'

# Prepare files to be used with WASM marian-decoder
package-files-wasm-docker: BUILD_DIR = /repo/build-wasm-docker
package-files-wasm-docker: MODELS_DIR = /repo/models
package-files-wasm-docker:
	${run_on_docker} ${WASM_IMAGE} bash -c 'sacrebleu -t wmt13 -l es-en --echo src > /repo/models/newstest2013.es && \
head -n300 /repo/models/newstest2013.es > /repo/models/newstest2013.es.top300lines && \
python3 /emsdk/upstream/emscripten/tools/file_packager.py ${BUILD_DIR}/model-files.data --preload ${MODELS_DIR}/@ --js-output=${BUILD_DIR}/model-files.js'

# Run WASM marian-decoder
run-wasm-docker: BUILD_DIR = /repo/build-wasm-docker
run-wasm-docker:
	${run_on_docker} -p 8000:8000 ${WASM_IMAGE} bash -c 'emrun --no_browser --port 8000 ${BUILD_DIR}'

wasm-shell: INTERACTIVE_DOCKER_SESSION = -it
wasm-shell:
	${run_on_docker} ${WASM_IMAGE} bash


# Compile stdin test directly
cmake_compile_stdin_test_cmd  = em++
cmake_compile_stdin_test_cmd += -O2 -s WASM=1 -s ASSERTIONS=1 -s DISABLE_EXCEPTION_CATCHING=0
cmake_compile_stdin_test_cmd += -s FORCE_FILESYSTEM=1
cmake_compile_stdin_test_cmd += -s ALLOW_MEMORY_GROWTH=1
cmake_compile_stdin_test_cmd += -o test-stdin-wasm.html
cmake_compile_stdin_test_cmd += --pre-js /repo/wasm/pre-module.js --post-js /repo/wasm/post-module.js --shell-file /repo/wasm/custom_shell.html
cmake_compile_stdin_test_cmd += /repo/wasm/test_stdin.cpp

compile-and-run-stdin-test-wasm: BUILD_DIR = /repo/build-test-stdin-wasm
compile-and-run-stdin-test-wasm:
	${run_on_docker} ${WASM_IMAGE} bash -c '(rm -rf ${BUILD_DIR} || true) && \
mkdir -p ${BUILD_DIR} &&\
cd ${BUILD_DIR} &&\
${cmake_compile_stdin_test_cmd} &&\
${run_on_docker} -p 8009:8009 ${WASM_IMAGE} bash -c 'emrun --no_browser --port 8009 ${BUILD_DIR}'
