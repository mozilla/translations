#!/usr/bin/env python3

import argparse
from glob import glob
import subprocess
from pathlib import Path
import multiprocessing
import tarfile
from typing import Optional
from detect_docker import detect_docker

ROOT_DIR = (Path(__file__).parent / "../..").resolve()
BUILD_DIR = ROOT_DIR / "inference/build"


def ensure_build_directory():
    if not BUILD_DIR.exists():
        print(f"[build] Creating {BUILD_DIR.relative_to(ROOT_DIR)} directory...")
        BUILD_DIR.mkdir()
    else:
        print(f"[build] {BUILD_DIR.name} directory already exists. Skipping creation.")


def run_cmake(test: bool, cuda_toolkit: Optional[Path], build_cli: bool):
    print(f"[build] Running cmake for {BUILD_DIR.name}...")
    cmake_args = [
        "cmake",
        "../",
        # Always compile CPU support, even if doing a GPU build.
        "-DCOMPILE_CPU=on",
    ]
    if test:
        cmake_args.append("-DCOMPILE_TESTS=ON")

    if cuda_toolkit:
        cmake_args.append(f'-DCUDA_TOOLKIT_ROOT_DIR="{cuda_toolkit}"')
    else:
        cmake_args.append("-DCOMPILE_CUDA=off")

    if build_cli:
        # Do not use USE_FBGEMM when building the translate CLI as we want it to be as close
        # to the Wasm build as possible.
        cmake_args.append("-DBUILD_TRANSLATE_CLI=on")
    else:
        # For native builds, use the FBGEMM library for the matrix math operations.
        # https://github.com/pytorch/FBGEMM
        cmake_args.append("-DUSE_FBGEMM=on")

    subprocess.run(cmake_args, cwd=BUILD_DIR, check=True)


def run_make():
    cpus = multiprocessing.cpu_count()
    print(f"[build] Running make with {cpus} CPUs...")
    subprocess.run(["make", f"-j{cpus}"], cwd=BUILD_DIR, check=True)


def save_archive(archive: Path):
    assert str(archive).endswith(".tar.zst"), f"The archive must end with .tar.zst: {archive.name}"

    if not archive.parent.exists():
        print(f"[build] Creating directory: {archive.parent}")
        archive.mkdir(parents=True)

    print("[build] Collecting build artifacts for compression...")

    files = []
    files.extend(glob(str(BUILD_DIR / "marian*")))
    files.extend(glob(str(BUILD_DIR / "spm*")))
    files = [Path(f) for f in files if Path(f).is_file()]

    # e.g. "marian-fork.tar.zst" -> "marian-fork.tar"
    tar_path = archive.with_suffix("")

    print(f"[build] Creating archive: {tar_path}")
    with tarfile.open(tar_path, "w") as tar:
        for file_path in files:
            tar.add(file_path, arcname=file_path.name)

    print(f"[build] Compressing archive to: {archive}")
    subprocess.run(["zstd", "-f", tar_path, "-o", archive], check=True)

    print(f"[build] Removing uncompressed archive: {tar_path}")
    tar_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Run cmake and make from the inference directory."
    )
    parser.add_argument("--test", action="store_true", help="Compile the test code as well")
    parser.add_argument(
        "--archive", type=Path, default=None, help="Save the binary files to a .tar.zst archive"
    )
    parser.add_argument(
        "--cuda_toolkit",
        type=Path,
        help="If the CUDA toolkit is provided, marian-fork will be built with GPU support.",
    )
    parser.add_argument(
        "--build_cli",
        action="store_true",
        help="Builds a translate cli tool that is as close as possible to the Wasm build",
    )
    args = parser.parse_args()

    test: bool = args.test
    archive: Optional[Path] = args.archive
    cuda_toolkit: Optional[Path] = args.cuda_toolkit
    build_cli: bool = args.build_cli

    detect_docker()
    ensure_build_directory()
    run_cmake(test, cuda_toolkit, build_cli)
    run_make()

    if archive:
        save_archive(archive)


if __name__ == "__main__":
    main()
