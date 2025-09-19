#!/usr/bin/env python3
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from urllib.request import urlretrieve

SCRIPTS_PATH = os.path.realpath(os.path.dirname(__file__))
INFERENCE_PATH = os.path.dirname(SCRIPTS_PATH)
BUILD_PATH = os.path.join(INFERENCE_PATH, "build-wasm")
WASM_PATH = os.path.join(INFERENCE_PATH, "wasm")
WASM_TESTS_PATH = os.path.join(WASM_PATH, "tests")
GENERATED_PATH = os.path.join(WASM_TESTS_PATH, "generated")
MODELS_PATH = os.path.join(WASM_TESTS_PATH, "models")
MODELS_GCS_PATH = os.path.join(
    "https://storage.googleapis.com/releng-translations-dev/tests/models"
)
WASM_ARTIFACT = os.path.join(BUILD_PATH, "bergamot-translator.wasm")
JS_ARTIFACT = os.path.join(BUILD_PATH, "bergamot-translator.js")
JS_ARTIFACT_HASH = os.path.join(GENERATED_PATH, "bergamot-translator.js.sha256")


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_models():
    def get_model(src, trg):
        lex = f"lex.50.50.{src}{trg}.s2t.bin.gz"
        model = f"model.{src}{trg}.intgemm.alphas.bin.gz"
        vocab = f"vocab.{src}{trg}.spm.gz"
        gcs_prefix = f"{MODELS_GCS_PATH}/{src}{trg}"
        local_prefix = f"{MODELS_PATH}/{src}{trg}"
        os.makedirs(local_prefix, exist_ok=True)
        for file in (lex, model, vocab):
            urlretrieve(f"{gcs_prefix}/{file}", f"{local_prefix}/{file}")

    for from_lang, to_lang in (("es", "en"), ("fr", "en"), ("zh", "en")):
        get_model(from_lang, to_lang)
        get_model(to_lang, from_lang)


def main():
    parser = argparse.ArgumentParser(
        description="Test WASM by building and handling artifacts.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--clobber", action="store_true", help="Clobber the build artifacts")
    parser.add_argument(
        "--force-rebuild", action="store_true", help="Force rebuilding the artifacts"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build with debug symbols, useful for profiling",
    )
    parser.add_argument(
        "-j",
        type=int,
        help="Number of cores to use for building (default: all available cores)",
    )
    args = parser.parse_args()

    build_wasm_script = os.path.join(SCRIPTS_PATH, "build-wasm.py")
    build_command = [sys.executable, build_wasm_script]
    if args.clobber:
        build_command.append("--clobber")
    if args.force_rebuild:
        build_command.append("--force-rebuild")
    if args.debug:
        build_command.append("--debug")
    if args.j:
        build_command.extend(["-j", str(args.j)])

    print("\n🚀 Starting build-wasm.py")
    subprocess.run(build_command, check=True)

    print("\n📥 Pulling translations model from GCS\n")
    download_models()
    print(f"   Pulled all files in {MODELS_PATH}")

    print("\n📁 Copying generated build artifacts to the WASM test directory\n")

    os.makedirs(GENERATED_PATH, exist_ok=True)
    shutil.copy2(WASM_ARTIFACT, GENERATED_PATH)
    shutil.copy2(JS_ARTIFACT, GENERATED_PATH)

    print(f"   Copied the following artifacts to {GENERATED_PATH}:")
    print(f"     - {JS_ARTIFACT}")
    print(f"     - {WASM_ARTIFACT}")

    print(f"\n🔑 Calculating SHA-256 hash of {JS_ARTIFACT}\n")
    hash_value = calculate_sha256(JS_ARTIFACT)
    with open(JS_ARTIFACT_HASH, "w") as hash_file:
        hash_file.write(f"{hash_value}  {os.path.basename(JS_ARTIFACT)}\n")
    print(f"   Hash of {JS_ARTIFACT} written to")
    print(f"   {JS_ARTIFACT_HASH}")

    print("\n📂 Decompressing model files required for WASM testing\n")
    subprocess.run(["gzip", "-dkrf", MODELS_PATH], check=True)
    print(f"   Decompressed models in {MODELS_PATH}\n")

    print("\n🔧 Installing npm dependencies for WASM JS tests\n")
    subprocess.run(["npm", "install"], cwd=WASM_TESTS_PATH, check=True)

    print("\n📊 Running Translations WASM JS tests\n")
    subprocess.run(["npm", "run", "test"], cwd=WASM_TESTS_PATH, check=True)

    print("\n✅ test-wasm.py completed successfully.\n")


if __name__ == "__main__":
    main()
