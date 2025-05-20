#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import shutil
import subprocess
from typing import Optional

# The emsdk git submodule is set to revision 2346baa7bb44a4a0571cc75f1986ab9aaa35aa03 which
# corresponds to version 3.1.8. The latest version of emsdk had errors building sentencepiece.
EMSDK_VERSION = "3.1.8"

SCRIPTS_PATH = os.path.realpath(os.path.dirname(__file__))
INFERENCE_PATH = os.path.dirname(SCRIPTS_PATH)
PROJECT_ROOT_PATH = os.path.dirname(INFERENCE_PATH)
BUILD_PATH = os.path.join(INFERENCE_PATH, "build-wasm")
THIRD_PARTY_PATH = os.path.join(INFERENCE_PATH, "3rd_party")
MARIAN_PATH = os.path.join(INFERENCE_PATH, "marian")
EMSDK_PATH = os.path.join(THIRD_PARTY_PATH, "emsdk")
EMSDK_ENV_PATH = os.path.join(EMSDK_PATH, "emsdk_env.sh")
WASM_ARTIFACT = os.path.join(BUILD_PATH, "bergamot-translator.wasm")
JS_ARTIFACT = os.path.join(BUILD_PATH, "bergamot-translator.js")
BUILD_DIRECTORY = os.path.join(INFERENCE_PATH, "build-wasm")
WASM_PATH = os.path.join(INFERENCE_PATH, "wasm")
GEMM_SCRIPT = os.path.join(WASM_PATH, "patch-artifacts-import-gemm-module.sh")
DETECT_DOCKER_SCRIPT = os.path.join(SCRIPTS_PATH, "detect-docker.sh")

parser = argparse.ArgumentParser(
    description=__doc__,
    # Preserves whitespace in the help text.
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("--clobber", action="store_true", help="Clobber the build artifacts")
parser.add_argument("--force-rebuild", action="store_true", help="Force rebuilding the artifacts")
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


def ensure_docker():
    """Ensure the script is running inside Docker."""
    subprocess.run(
        [DETECT_DOCKER_SCRIPT, "inference-build-wasm"],
        cwd=INFERENCE_PATH,
        shell=True,
        check=True,
    )


def install_and_activate_emscripten():
    # Run these commands in the shell so that the configuration is saved.
    def run_shell(command):
        return subprocess.run(command, cwd=EMSDK_PATH, shell=True, check=True)

    print(f"\n🛠️ Installing EMSDK version {EMSDK_VERSION}\n")
    run_shell(f"./emsdk install {EMSDK_VERSION}")

    print("\n🛠️ Activating emsdk\n")
    run_shell(f"./emsdk activate {EMSDK_VERSION}")


def to_human_readable(size):
    """Convert sizes to human-readable format"""
    size_in_mb = size / 1048576
    return f"{size_in_mb:.2f}M ({size} bytes)"


def ensure_git_submodules():
    print("\n🔄 Initializing and updating Git submodules recursively.\n")
    subprocess.run(
        ["git", "submodule", "update", "--init", "--checkout", "--recursive"],
        cwd=PROJECT_ROOT_PATH,
        check=True,
    )


def revert_git_patch(repo_path, patch_path):
    print(f"Reverting patch {patch_path} from {os.path.basename(repo_path)}")
    subprocess.check_call(["git", "apply", "-R", "--reject", patch_path], cwd=PROJECT_ROOT_PATH)


def prepare_js_artifact():
    """
    Prepares the Bergamot JS artifact for use in Gecko by adding the proper license header
    to the file, including helpful memory-growth logging, and wrapping the generated code
    in a single function that both takes and returns the Bergamot WASM module.
    """
    # Start with the license header and function wrapper
    source = (
        "\n".join(
            [
                "/* This Source Code Form is subject to the terms of the Mozilla Public",
                " * License, v. 2.0. If a copy of the MPL was not distributed with this",
                " * file, You can obtain one at http://mozilla.org/MPL/2.0/. */",
                "",
                "function loadBergamot(Module) {",
                "",
            ]
        )
        + "\n"
    )

    # Read the original JS file and indent its content
    with open(JS_ARTIFACT, "r", encoding="utf8") as file:
        for line in file:
            source += "  " + line

    # Close the function wrapper
    source += "\n  return Module;\n}"

    # Use the Module's printing
    source = source.replace("console.log(", "Module.print(")

    # Add instrumentation to log memory size information
    source = source.replace(
        "function updateGlobalBufferAndViews(buf) {",
        """
          function updateGlobalBufferAndViews(buf) {
            const mb = (buf.byteLength / 1_000_000).toFixed();
            Module.print(
              `Growing wasm buffer to ${mb}MB (${buf.byteLength} bytes).`
            );
        """,
    )

    print(f"\n📄 Updating {JS_ARTIFACT} in place")
    # Write the modified content back to the original file
    with open(JS_ARTIFACT, "w", encoding="utf8") as file:
        file.write(source)


def build_bergamot(args: Optional[list[str]]):
    if args.clobber and os.path.exists(BUILD_PATH):
        shutil.rmtree(BUILD_PATH)

    if not os.path.exists(BUILD_PATH):
        os.mkdir(BUILD_PATH)

    # These commands require the emsdk environment variables to be set up.
    def run_shell(command):
        if '"' in command or "'" in command:
            raise Exception("This run_shell utility does not support quotes.")

        return subprocess.run(
            # "source" is not available in all shells so explicitly
            f"bash -c 'source {EMSDK_ENV_PATH} && {command}'",
            cwd=BUILD_PATH,
            shell=True,
            check=True,
        )

    flags = ""
    if args.debug:
        flags = "-DCMAKE_BUILD_TYPE=RelWithDebInfo"

    print("\n🏃 Running CMake for Bergamot\n")
    run_shell(f"emcmake cmake -DCOMPILE_WASM=on -DWORMHOLE=off {flags} {INFERENCE_PATH}")

    if args.j:
        # If -j is specified explicitly, use it.
        cores = args.j
    elif os.getenv("HOST_OS") == "Darwin":
        # There is an issue building with multiple cores when the Linux Docker container is
        # running on a macOS host system. If the Docker container was created with HOST_OS
        # set to Darwin, we should use only 1 core to build.
        cores = 1
    else:
        # Otherwise, build with as many cores as we have.
        cores = multiprocessing.cpu_count()

    print(f"\n🏃 Building Bergamot with emmake using {cores} cores\n")

    try:
        run_shell(f"emmake make -j {cores}")
    except:
        print(f"❌ Build failed with {cores} cores.")
        print("This has been known to occur on macOS AArch64.\n")
        print("Please try running again with -j 1.")
        raise

    print("\n🪚  Patching Bergamot for gemm support\n")
    subprocess.check_call(["bash", GEMM_SCRIPT, BUILD_PATH])

    print("\n✅ Build complete\n")
    print("  " + JS_ARTIFACT)
    print("  " + WASM_ARTIFACT)

    # Get the sizes of the build artifacts.
    wasm_size = os.path.getsize(WASM_ARTIFACT)
    gzip_size = int(
        subprocess.run(
            f"gzip -c {WASM_ARTIFACT} | wc -c",
            check=True,
            shell=True,
            capture_output=True,
        ).stdout.strip()
    )
    print(f"  Uncompressed wasm size: {to_human_readable(wasm_size)}")
    print(f"  Compressed wasm size: {to_human_readable(gzip_size)}")

    prepare_js_artifact()


def main():
    args = parser.parse_args()

    if (
        os.path.exists(BUILD_PATH)
        and os.path.isdir(BUILD_PATH)
        and os.listdir(BUILD_PATH)
        and not args.clobber
        and not args.force_rebuild
    ):
        print(f"\n🏗️  Build directory {BUILD_PATH} already exists and is non-empty.\n")
        print("   Pass the --force-rebuild flag or --clobber to rebuild.")
        return

    if not os.path.exists(THIRD_PARTY_PATH):
        os.mkdir(THIRD_PARTY_PATH)

    ensure_docker()

    ensure_git_submodules()

    install_and_activate_emscripten()

    build_bergamot(args)


if __name__ == "__main__":
    main()
