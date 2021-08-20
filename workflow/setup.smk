
rule setup:
    message: "Installing dependencies"
    log: f"{log_dir}/install-deps.log"
    conda: "../envs/environment.yml"
    threads: 1
    group: 'setup'
    # specific to local machine
    output: touch("/tmp/flags/setup.done")
    shell: 'bash pipeline/setup/install-deps.sh 2>&1 | tee {log}'

rule marian:
    message: "Compiling marian"
    log: f"{log_dir}/compile-marian.log"
    conda: "../envs/environment.yml"
    threads: workflow.cores
    group: 'setup'
    input: rules.setup.output
    output:
        trainer=protected(f"{marian_dir}/marian"),
        decoder=protected(f"{marian_dir}/marian-decoder"),
        scorer=protected(f"{marian_dir}/marian-scorer"),
        vocab=protected(f'{marian_dir}/spm_train')
    shell: '''
        MARIAN={marian_dir} THREADS={threads} CUDA_DIR={cuda_dir} \
        bash pipeline/setup/compile-marian.sh 2>&1 | tee {log}'''

rule fast_align:
    message: "Compiling fast align"
    log: f"{log_dir}/compile-fast-align.log"
    conda: "../envs/environment.yml"
    threads: workflow.cores
    group: 'setup'
    input: rules.setup.output
    output: protected(f"{bin}/fast_align")
    shell: '''
        BUILD_DIR=3rd_party/fast_align/build BIN={bin} THREADS={threads} \
        bash pipeline/setup/compile-fast-align.sh 2>&1 | tee {log}'''

rule extract_lex:
    message: "Compiling fast align"
    log: f"{log_dir}/compile-extract-lex.log"
    conda: "../envs/environment.yml"
    threads: workflow.cores
    group: 'setup'
    input: rules.setup.output
    output: protected(f"{bin}/extract_lex")
    shell: '''
        BUILD_DIR=3rd_party/extract-lex/build BIN={bin} THREADS={threads} \
        bash pipeline/setup/compile-extract-lex.sh 2>&1 | tee {log}'''
