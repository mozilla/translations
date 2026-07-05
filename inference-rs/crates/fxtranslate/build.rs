// Compiles the gemmology SIMD shim (src/gemmology_shim.cpp) when the `gemmology`
// feature is on and a kernel is wired for the target. On success it emits
// `--cfg gemmology_simd`; otherwise it falls back to the portable scalar kernel
// *without failing the build*, so the fast default still builds and runs on any
// target. The gemmology/xsimd headers are vendored under vendor/; GEMMOLOGY_DIR /
// XSIMD_INCLUDE_DIR override the include paths.

use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(gemmology_simd)");

    // Scalar kernel unless the SIMD kernel is both requested and not opted out.
    if std::env::var_os("CARGO_FEATURE_GEMMOLOGY").is_none() {
        return;
    }
    if std::env::var_os("CARGO_FEATURE_PORTABLE").is_some() {
        println!("cargo::warning=fxtranslate: `portable` is set — using the scalar kernel (no SIMD, no C++).");
        return;
    }

    // A vendored shim exists only for aarch64 (i8mm) today. Other targets fall
    // back to scalar until their backend is wired (see issues/24). aarch64 is
    // assumed to have i8mm (true for Apple Silicon); there is no runtime probe.
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch != "aarch64" {
        println!("cargo::warning=fxtranslate: no SIMD kernel wired for `{arch}` yet (see issues/24) — using the scalar kernel.");
        return;
    }

    let vendor = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("vendor");
    let gemmology = std::env::var_os("GEMMOLOGY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| vendor.join("gemmology"));
    let xsimd = std::env::var_os("XSIMD_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| vendor.join("xsimd/include"));

    let header = gemmology.join("gemmology.h");
    if !header.exists() {
        println!(
            "cargo::warning=fxtranslate: gemmology headers not found at {} — using the scalar kernel.",
            header.display()
        );
        return;
    }

    println!("cargo:rerun-if-changed=src/gemmology_shim.cpp");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-env-changed=GEMMOLOGY_DIR");
    println!("cargo:rerun-if-env-changed=XSIMD_INCLUDE_DIR");

    let result = cc::Build::new()
        .cpp(true)
        .std("c++17")
        .flag("-march=armv8.4-a+i8mm") // enables __ARM_FEATURE_MATMUL_INT8 (usdot)
        .opt_level(3)
        .include(&gemmology)
        .include(&xsimd)
        .file("src/gemmology_shim.cpp")
        .try_compile("gemmology_shim");

    match result {
        Ok(()) => println!("cargo::rustc-cfg=gemmology_simd"),
        Err(e) => println!(
            "cargo::warning=fxtranslate: gemmology C++ build failed ({e}) — using the scalar kernel."
        ),
    }
}
