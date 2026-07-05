// Compiles the gemmology SIMD shim (src/gemmology_shim.cpp) when the `gemmology`
// feature is on and a kernel is wired for the target. On success it emits
// `--cfg gemmology_simd`; otherwise it falls back to the portable scalar kernel
// *without failing the build*, so the fast default still builds and runs on any
// target. aarch64 uses the i8mm dot-product kernel, x86_64 the AVX2 kernel.
//
// Setting FXTRANSLATE_REQUIRE_SIMD flips that fallback into a hard error: any
// reason we'd otherwise drop to the scalar kernel (unsupported arch, missing
// headers, a failed C++ build, `portable`) becomes a panic. CI sets it so a green
// build proves the SIMD kernel is actually compiled rather than silently absent.
//
// The gemmology/xsimd headers are vendored under vendor/; GEMMOLOGY_DIR /
// XSIMD_INCLUDE_DIR override the include paths.

use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(gemmology_simd)");
    println!("cargo:rerun-if-env-changed=FXTRANSLATE_REQUIRE_SIMD");

    // When required, falling back to scalar is a build failure, not a warning.
    let require_simd = std::env::var_os("FXTRANSLATE_REQUIRE_SIMD").is_some();
    let bail = |reason: String| {
        if require_simd {
            panic!("fxtranslate: FXTRANSLATE_REQUIRE_SIMD is set but {reason}");
        }
        println!("cargo::warning=fxtranslate: {reason} — using the scalar kernel.");
    };

    // Scalar kernel unless the SIMD kernel is both requested and not opted out.
    if std::env::var_os("CARGO_FEATURE_GEMMOLOGY").is_none() {
        return;
    }
    if std::env::var_os("CARGO_FEATURE_PORTABLE").is_some() {
        bail("`portable` is set, which disables the SIMD kernel (no SIMD, no C++)".into());
        return;
    }

    // Pick the arch-specific xsimd kernel + the `-m` flags to compile it with. The
    // define selects the `Arch` in the shim (see gemmology_shim.cpp). Targets not
    // listed here have no wired kernel and fall back to the scalar path.
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let (arch_define, arch_flag) = match arch.as_str() {
        // -march=armv8.4-a+i8mm enables __ARM_FEATURE_MATMUL_INT8 (usdot).
        "aarch64" => ("FXT_GEMM_I8MM", "-march=armv8.4-a+i8mm"),
        // AVX2 baseline: runs on every x86-64-v2+ CPU. The exact VNNI kernels
        // behind CPUID dispatch are a possible follow-up.
        "x86_64" => ("FXT_GEMM_AVX2", "-mavx2"),
        _ => {
            bail(format!("no SIMD kernel wired for `{arch}` yet"));
            return;
        }
    };

    let vendor = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("vendor");
    let gemmology = std::env::var_os("GEMMOLOGY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| vendor.join("gemmology"));
    let xsimd = std::env::var_os("XSIMD_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| vendor.join("xsimd/include"));

    let header = gemmology.join("gemmology.h");
    if !header.exists() {
        bail(format!(
            "gemmology headers not found at {}",
            header.display()
        ));
        return;
    }

    println!("cargo:rerun-if-changed=src/gemmology_shim.cpp");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-env-changed=GEMMOLOGY_DIR");
    println!("cargo:rerun-if-env-changed=XSIMD_INCLUDE_DIR");

    let result = cc::Build::new()
        .cpp(true)
        .std("c++17")
        .flag(arch_flag)
        .define(arch_define, None)
        .opt_level(3)
        .include(&gemmology)
        .include(&xsimd)
        .file("src/gemmology_shim.cpp")
        .try_compile("gemmology_shim");

    match result {
        Ok(()) => println!("cargo::rustc-cfg=gemmology_simd"),
        Err(e) => bail(format!("gemmology C++ build failed ({e})")),
    }
}
