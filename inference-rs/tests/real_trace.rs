//! Integration test against a real reference trace, if one has been recorded.
//!
//! The trace is large and gitignored, so this test skips (rather than fails)
//! when it is absent. To generate it:
//!
//!   task rs:translate-reference -- en fr --text "Hello world." \
//!       --cpu-threads 1 --trace
//!
//! which writes inference-rs/artifacts/enfr.trace. Tests run with the crate
//! root as the working directory, so the relative path resolves there.
//!
//! Exercises the trace reader, which lives behind `instrumentation`.
#![cfg(feature = "instrumentation")]

use inference_rs::trace::{DType, Trace};

const TRACE_PATH: &str = "artifacts/enfr.trace";

#[test]
fn reads_real_trace_end_to_end() {
    if !std::path::Path::new(TRACE_PATH).exists() {
        eprintln!("skipping: {TRACE_PATH} not found (record one with task rs:translate-reference -- ... --trace)");
        return;
    }

    let trace = Trace::load(TRACE_PATH).expect("real trace should parse");
    assert!(!trace.is_empty(), "trace should contain records");

    // The first record of the enfr graph is the float32 embedding parameter.
    let first = &trace.records[0];
    assert_eq!(first.op_type, "param");
    assert_eq!(first.dtype, DType::Float32);

    // Every record's byte count must match shape * element size, and every
    // float32/int8 view must decode to the right element count. This walks the
    // entire multi-hundred-MB file, exercising the parser on real data.
    for (index, record) in trace.records.iter().enumerate() {
        let expected_bytes = record.num_elements() * record.dtype.element_size();
        assert_eq!(
            record.data.len(),
            expected_bytes,
            "record {index} (id {}, {}) byte count",
            record.id,
            record.op_type
        );

        match record.dtype {
            DType::Float32 => assert_eq!(record.to_f32().unwrap().len(), record.num_elements()),
            DType::Int8 | DType::Intgemm8 => {
                assert_eq!(record.to_i8().unwrap().len(), record.num_elements())
            }
            _ => {}
        }
    }

    // The quantized path must be present: intgemmAffine nodes should resolve
    // their inputs (prepared A, prepared B, bias) via backward id search.
    let affine_index = trace
        .records
        .iter()
        .position(|r| r.op_type == "intgemmAffine")
        .expect("trace should contain an intgemmAffine node");
    let inputs = trace.inputs(affine_index).expect("affine inputs resolve");
    assert!(
        !inputs.is_empty(),
        "intgemmAffine should have resolvable inputs"
    );
}
