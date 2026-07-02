// Compiles the gemmology i8mm shim (src/gemmology_shim.cpp) only under the
// `gemmology` feature. Default builds carry no C++ toolchain dependency.
//
// gemmology and its xsimd dependency are vendored in-tree as submodules of the
// marian-fork; the paths are overridable via GEMMOLOGY_DIR / XSIMD_INCLUDE_DIR.

use std::path::PathBuf;

fn main() {
    if std::env::var_os("CARGO_FEATURE_GEMMOLOGY").is_none() {
        return;
    }

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch != "aarch64" {
        panic!(
            "the `gemmology` feature targets the ARM i8mm kernel and only builds on \
             aarch64 (target arch is `{arch}`)"
        );
    }

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let third_party = manifest
        .parent()
        .expect("crate dir has a parent (repo root)")
        .join("inference/marian-fork/src/3rd_party");
    let gemmology = std::env::var_os("GEMMOLOGY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| third_party.join("gemmology"));
    let xsimd = std::env::var_os("XSIMD_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| third_party.join("xsimd/include"));

    let header = gemmology.join("gemmology.h");
    assert!(
        header.exists(),
        "gemmology.h not found at {} — init submodules or set GEMMOLOGY_DIR",
        header.display()
    );

    println!("cargo:rerun-if-changed=src/gemmology_shim.cpp");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-env-changed=GEMMOLOGY_DIR");
    println!("cargo:rerun-if-env-changed=XSIMD_INCLUDE_DIR");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .flag("-march=armv8.4-a+i8mm") // enables __ARM_FEATURE_MATMUL_INT8 (usdot)
        .opt_level(3)
        .include(&gemmology)
        .include(&xsimd)
        .file("src/gemmology_shim.cpp")
        .compile("gemmology_shim");
}
