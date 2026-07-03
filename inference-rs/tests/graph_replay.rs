//! Graph-level replay against the reference trace.
//!
//! Recomputes the whole graph forward from recomputed inputs and reports the
//! first divergence. Skips when the trace or model is absent.
//!
//! Exercises the trace reader + comparator, which live behind `instrumentation`.
#![cfg(feature = "instrumentation")]

use std::sync::OnceLock;

use inference_rs::compare::Tolerance;
use inference_rs::graph::{self, ReplayReport};
use inference_rs::model::Model;
use inference_rs::trace::Trace;

const TRACE_PATH: &str = "artifacts/enfr.trace";
const MODEL_PATH: &str = "../data/models/enfr/model.enfr.intgemm.alphas.bin";

fn trace() -> Option<&'static Trace> {
    static T: OnceLock<Option<Trace>> = OnceLock::new();
    T.get_or_init(|| {
        std::path::Path::new(TRACE_PATH)
            .exists()
            .then(|| Trace::load(TRACE_PATH).expect("trace parses"))
    })
    .as_ref()
}

fn model() -> Option<&'static Model> {
    static M: OnceLock<Option<Model>> = OnceLock::new();
    M.get_or_init(|| {
        std::path::Path::new(MODEL_PATH)
            .exists()
            .then(|| Model::load(MODEL_PATH).expect("model parses"))
    })
    .as_ref()
}

#[test]
fn replays_full_graph() {
    let (Some(trace), Some(model)) = (trace(), model()) else {
        eprintln!("skipping graph replay: trace or model absent");
        return;
    };

    let report: ReplayReport = graph::replay(trace, model, Tolerance::default());
    eprintln!(
        "graph replay: {} nodes total, {} recomputed, {} matched-prefix, {} passthrough",
        report.total, report.compared, report.matched_prefix, report.passthrough
    );
    if let Some(d) = &report.first_divergence {
        eprintln!(
            "first divergence at node {} (id {}, {}): {}",
            d.index, d.id, d.op_type, d.detail
        );
    }

    assert!(
        report.fully_matched(),
        "graph diverged before completing; see first divergence above"
    );
}
