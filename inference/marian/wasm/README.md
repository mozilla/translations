
## WASM

### Compiling wasm marian-decoder on local machine

For docker based builds, please refer [Docker Compilation Steps](#Docker-Compilation)

1. Download and Install Emscripten using following instructions (skip this step if emsdk tool chain is already installed)
    * Get the sdk: `git clone https://github.com/emscripten-core/emsdk.git`
    * Enter the cloned directory: `cd emsdk`
    * Install the sdk tools: `./emsdk install 3.1.8`
    * Activate the sdk tools: `./emsdk activate 3.1.8`
    * Activate path variables: `source ./emsdk_env.sh`

    `EMSDK` environment variable will point to the valid emsdk repo after executing the instructions above.

2. Compile

    * Create a build directory (e.g. `build-wasm`) and run compilation commands inside it as below:
        ```bash
        mkdir build-wasm; cd build-wasm

        emcmake cmake -DCOMPILE_CUDA=off -DUSE_STATIC_LIBS=on -DUSE_DOXYGEN=off -DUSE_FBGEMM=off -DUSE_MKL=off -DUSE_NCCL=off -DUSE_WASM_COMPATIBLE_SOURCE=on -DCOMPILE_WASM=on ../

        emmake make -j
        ```

        The artifacts (.js and .wasm files) will be available in `build-wasm` folder.

### <a name="Perform-Translation"></a> Performing Translation using wasm marian-decoder

1. Pre-processing (Package files to WASM-compiled runtime)

    This step is required to be able to perform translation using wasm binary.

    The script `package-benchmark.sh` inside `wasm` folder downloads and packages the Bergamot project specific Spanish to English translation models, vocabulary, lexical shortlist files and a News test file as source text for translation. (Please install `sacrebleu` before if not installed already using command: `pip install sacrebleu`).

    From the build directory (`build-wasm` for local builds or `build-wasm-docker` for docker builds), run:

    ```bash
    bash ../wasm/package-benchmark.sh
    ```

2. Perform Translation
    1. Launch the emscripten-generated HTML page in a web browser using following commands:
        ```bash
        emrun --no_browser --port 8000 .
        ```

    2. Open up following link in Firefox nightly browser (replace `stdin-input` with the text that you want to translate and `command-line-args` with the appropriate model, vocabulary files etc.)

        ```bash
        http://localhost:8000/marian-decoder.html?stdinInput=<stdin-input>&arguments=<command-line-args>
        ```

        e.g. To translate "Hola mundo" to English using the Bergamot project specific Spanish to English files (packaged above), open this link:

        ```bash
        http://localhost:8000/marian-decoder.html?stdinInput=Hola mundo&arguments=-m /model.npz -v /vocab.esen.spm /vocab.esen.spm --cpu-threads 1
        ```

        Note: To run in Chrome, launch Chrome with `  --js-flags="--experimental-wasm-simd"`, eg:

        ```bash
        /Applications/Google\ Chrome\ Canary.app/Contents/MacOS/Google\ Chrome\ Canary  --js-flags="--experimental-wasm-simd"
        ```

        Please remember that the Developer Tools must not be open when opening the links or refreshing the page to run the benchmark again.

    3. Open the Developer Tools and you should see the result in console.


### Benchmarking

If you used the script `package-benchmark.sh` mentioned above then open following for benchmarking (please remember that the Developer Tools must not be open when opening the links or refreshing the page to run the benchmark again.)

1. float32

    `http://localhost:8000/marian-decoder.html?arguments=-m /model.npz -v /vocab.esen.spm /vocab.esen.spm -i /newstest2013.es.top300lines --beam-size 1 --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src -w 128 --skip-cost --shortlist /lex.s2t 50 50 --cpu-threads 1`

2. intgemm8

    `http://localhost:8000/marian-decoder.html?arguments=-m /model.npz -v /vocab.esen.spm /vocab.esen.spm -i /newstest2013.es.top300lines --beam-size 1 --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src -w 128 --skip-cost --shortlist /lex.s2t 50 50 --cpu-threads 1 --int8shift`

3. intgemm8 with binary model file

    `http://localhost:8000/marian-decoder.html?arguments=-m /model.intgemm.bin -v /vocab.esen.spm /vocab.esen.spm -i /newstest2013.es.top300lines --beam-size 1 --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src -w 128 --skip-cost --shortlist /lex.s2t 50 50 --cpu-threads 1 --int8shift`

4. intgemm8alphas

    `http://localhost:8000/marian-decoder.html?arguments=-m /model.npz -v /vocab.esen.spm /vocab.esen.spm -i /newstest2013.es.top300lines --beam-size 1 --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src -w 128 --skip-cost --shortlist /lex.s2t 50 50 --cpu-threads 1 --int8shiftAlphaAll`

5. intgemm8alphas with binary model file

    `http://localhost:8000/marian-decoder.html?arguments=-m /model.intgemm.alphas.bin -v /vocab.esen.spm /vocab.esen.spm -i /newstest2013.es.top300lines --beam-size 1 --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src -w 128 --skip-cost --shortlist /lex.s2t 50 50 --cpu-threads 1 --int8shiftAlphaAll`

### <a name="Docker-Compilation"></a> Compiling wasm marian-decoder on Docker

Alternatively, wasm marian-decoder can also be compiled on docker.

1. Prepare docker image for WASM compilation
    This step is required only for the first time (or after any changes to docker image)

    ```bash
    make wasm-image
    ```

2. Compile to wasm

    ```bash
    make compile-wasm-docker
    ```
    The artifacts (.js and .wasm files) will be available in `build-wasm-docker` folder in root of this repository.

3. Performing Translation

    Please follow [Performing Translation](#Perform-Translation) for this.


#### Additional helpful instructions for docker based builds

1. Clean build (to start next compilation from scratch)

    ```bash
    make clean-wasm-docker
    ```
2. Compile and run a wasm stdin test:

    ```bash
    make compile-and-run-stdin-test-wasm
    open "http://localhost:8009/compile-test-stdin-wasm.html"
    ```

3. Enter a docker container shell for manually running commands:

    ```bash
    make wasm-shell
    ```
