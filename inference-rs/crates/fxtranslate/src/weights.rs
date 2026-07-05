//! Model-weights view for dynamic execution.
//!
//! Wraps [`crate::model::Model`] and resolves parameters by marian's naming
//! convention, exposing exactly what the transformer forward needs:
//! - [`Weights::affine`] runs a shifted int8 affine end to end (quantize the
//!   activation, prepare the bias, integer GEMM, unquantize) from a weight's
//!   base name — the `unquant = 1/(qA·qB)` multiplier is computed from the model's
//!   own quant multipliers, nothing from the trace.
//! - [`Weights::src_embed_row_into`] / [`Weights::trg_embed_row_into`] write one
//!   embedding row, and [`Weights::full_logits`] the tied output projection. Their backing
//!   representation (resident dequantized f32 tables vs. on-the-fly int8) is
//!   chosen by the `lean-embed` feature.
//! - [`Weights::f32`] returns float parameters (biases, layernorm scale/bias).
//! - [`Config`] holds the architecture dims parsed from `special:model.yml`.

use std::collections::HashMap;

use crate::model::Model;
use crate::ops;
#[cfg(not(feature = "lean-embed"))]
use crate::trace::DType;
#[cfg(feature = "gemmology")]
use {crate::gemm::PreparedB, std::cell::RefCell};

/// A gemmology-prepared affine weight. Built once at load; the raw int8 bytes are
/// then freed from the model (the packed form is all the GEMM needs), so each
/// weight is held once, not twice.
#[cfg(feature = "gemmology")]
struct AffineWeight {
    pb: PreparedB,
    /// Shift correction `-127·unquant·colsum(W)` (bias-independent), length `n`.
    correction: Vec<f32>,
    /// Full prepared bias `correction + raw_bias`, cached on first use (the bias
    /// name is known only at call time). `None` until then.
    bias: Option<Vec<f32>>,
    qa: f32,
    unquant: f32,
}

/// Reusable scratch for the shifted int8 affine, so the hot path allocates no
/// per-call activation buffers.
#[cfg(feature = "gemmology")]
#[derive(Default)]
struct GemmScratch {
    a_u8: Vec<u8>,
    /// One int8 embedding row gathered out of the packed projection buffer, reused
    /// across lookups so the on-demand dequant path allocates nothing.
    wemb_row: Vec<i8>,
}

/// Architecture hyperparameters read from the embedded `special:model.yml`.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub dim_emb: usize,
    pub heads: usize,
    pub enc_depth: usize,
    pub dec_depth: usize,
    pub dim_ffn: usize,
    pub vocab: usize,
}

impl Config {
    fn parse(yaml: &str) -> Config {
        let get = |key: &str, default: usize| -> usize {
            for line in yaml.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix(key) {
                    if let Some(v) = rest.trim().strip_prefix(':') {
                        if let Ok(n) = v.trim().parse() {
                            return n;
                        }
                    }
                }
            }
            default
        };
        Config {
            dim_emb: get("dim-emb", 384),
            heads: get("transformer-heads", 8),
            enc_depth: get("enc-depth", 6),
            dec_depth: get("dec-depth", 4),
            dim_ffn: get("transformer-dim-ffn", 1536),
            // dim-vocabs is a YAML list; fall back to the embedding row count.
            vocab: get("dim-vocabs", 0),
        }
    }
}

/// Loaded model weights + parsed config.
///
/// The embedding representation is chosen at compile time by the `lean-embed`
/// feature:
/// - **default**: the int8 `Wemb` is dequantized into resident f32 tables (fast
///   lookups + full-vocab float projection) at the cost of ~49 MB/table.
/// - **`lean-embed`**: no f32 tables. Embedding rows are dequantized on demand
///   and the output projection runs full-vocab in int8, so only the int8 table
///   (already in `model`) is resident. Big memory win; a hot-path change whose
///   perf is still to be measured, hence gated.
pub struct Weights {
    model: Model,
    config: Config,
    trg_vocab: usize,
    dim: usize,
    /// Decoded layer-norm `scale`/`bias` params, keyed by the sublayer base name
    /// (`{base}_ln_scale` → `(scale, bias?)`). Cached once at load so `postnorm`
    /// borrows them instead of decoding a fresh `Vec` from the model every call.
    layer_norms: HashMap<String, (Vec<f32>, Option<Vec<f32>>)>,
    /// Model param name of the target embedding (`Wemb` shared, `decoder_Wemb`
    /// split); the int8 output projection reads it back from the model on demand.
    trg_wemb_param: &'static str,

    /// Dequantized target embedding `[trg_vocab, dim]`; the tied output weight.
    #[cfg(not(feature = "lean-embed"))]
    trg_wemb: Vec<f32>,
    /// Dequantized source embedding; `None` for shared vocab (reuses `trg_wemb`).
    #[cfg(not(feature = "lean-embed"))]
    src_wemb: Option<Vec<f32>>,

    /// Source embedding param name (`== trg_wemb_param` for shared vocab).
    #[cfg(feature = "lean-embed")]
    src_wemb_param: &'static str,
    #[cfg(feature = "lean-embed")]
    src_inv_qmult: f32,
    #[cfg(feature = "lean-embed")]
    trg_inv_qmult: f32,
    /// Prepared bias for the full-vocab int8 output projection (static, so it is
    /// precomputed once here rather than per decode step).
    #[cfg(feature = "lean-embed")]
    proj_bias: Vec<f32>,
    #[cfg(feature = "lean-embed")]
    proj_qa: f32,
    #[cfg(feature = "lean-embed")]
    proj_unquant: f32,

    /// Affine weights packed into gemmology's layout at load, keyed by param name.
    /// Their raw int8 bytes are freed from `model` once packed (no double copy).
    /// Interior-mutable only to cache the full prepared bias on first use.
    #[cfg(feature = "gemmology")]
    affine_cache: RefCell<HashMap<String, AffineWeight>>,
    /// The output-projection `Wemb` packed for the full-vocab GEMM (lean-embed;
    /// built on first projection). The raw `Wemb` is kept (embedding lookups
    /// need it), so this one is the unavoidable second copy.
    #[cfg(all(feature = "gemmology", feature = "lean-embed"))]
    proj_pb: RefCell<Option<PreparedB>>,
    /// Reusable activation scratch for the affine hot path.
    #[cfg(feature = "gemmology")]
    scratch: RefCell<GemmScratch>,
}

/// Pack every affine weight (those with a `{name}_QuantMultA` sibling, excluding
/// the embedding) into gemmology's layout, cache the shift correction, and free
/// the raw int8 bytes from `model` — so each affine weight is resident once
/// (packed) rather than twice (raw + packed). Weights gemmology can't take
/// (`k % 16 != 0`) are left raw for the scalar fallback.
#[cfg(feature = "gemmology")]
fn prepare_affines(model: &mut Model, embed_param: &str) -> HashMap<String, AffineWeight> {
    let names: Vec<String> = model.items.iter().map(|it| it.name.clone()).collect();
    let mut cache = HashMap::new();
    let mut dropped: Vec<String> = Vec::new();
    for name in &names {
        if name.ends_with("_QuantMultA") || name == embed_param {
            continue;
        }
        let qa_name = format!("{name}_QuantMultA");
        let it = match model.get(name) {
            Some(it) if it.shape.len() >= 2 => it,
            _ => continue,
        };
        let (k, n) = (it.shape[0] as usize, it.shape[1] as usize);
        let b = match it.int8_transposed() {
            Ok(b) => b,
            Err(_) => continue,
        };
        let qb = match it.quant_mult() {
            Ok(q) => q,
            Err(_) => continue,
        };
        let qa = match model.get(&qa_name).and_then(|i| i.to_f32().ok()) {
            Some(v) if !v.is_empty() => v[0],
            _ => continue, // no activation quant-mult -> not an affine weight
        };
        let pb = match PreparedB::new(b, n, k) {
            Some(pb) => pb,   // k % 16 == 0
            None => continue, // keep raw for the scalar path
        };
        let unquant = 1.0 / (qa * qb);
        let correction = ops::prepare_bias(b, n, k, &vec![0.0; n], unquant);
        cache.insert(
            name.clone(),
            AffineWeight {
                pb,
                correction,
                bias: None,
                qa,
                unquant,
            },
        );
        dropped.push(name.clone());
    }
    // Free the raw bytes of everything we packed (the packed copy is all the GEMM
    // needs; the correction covers the bias).
    for it in model.items.iter_mut() {
        if dropped.iter().any(|d| d == &it.name) {
            // Release the raw bytes (owned copy, or the mmap view's Arc handle).
            it.data = crate::model::Bytes::Owned(Vec::new());
        }
    }
    cache
}

/// Load and dequantize an embedding parameter into a resident `[vocab, dim]` f32
/// table (int8/quantMult, or float if it shipped dequantized).
#[cfg(not(feature = "lean-embed"))]
fn load_embedding(model: &Model, name: &str) -> Result<Vec<f32>, String> {
    let item = model
        .get(name)
        .ok_or_else(|| format!("model has no {name}"))?;
    match item.dtype {
        DType::Float32 => item.to_f32().map_err(|e| e.to_string()),
        _ => {
            let inv = 1.0 / item.quant_mult().map_err(|e| e.to_string())?;
            let raw = item.int8_transposed().map_err(|e| e.to_string())?;
            Ok(raw.iter().map(|&b| b as f32 * inv).collect())
        }
    }
}

/// Activation quant-mult (qA) for the tied output projection. Shared-vocab models
/// name that node "none" (`none_QuantMultA`, a plain float32 scalar). Split-vocab
/// (CJK) models name it `decoder_Wemb_QuantMultA` and store it as an `intgemm8`
/// scalar: a single int8 value (127) plus an appended quant multiplier, whose
/// *dequantized* value (`127 / quant_mult`) is the alpha.
fn read_output_qa(model: &Model) -> f32 {
    if let Some(v) = model.get("none_QuantMultA").and_then(|it| it.to_f32().ok()) {
        return v[0];
    }
    if let Some(it) = model.get("decoder_Wemb_QuantMultA") {
        let raw = it.int8_transposed().expect("intgemm8 alpha scalar")[0] as f32;
        let qmult = it.quant_mult().expect("intgemm8 alpha quant mult");
        return raw / qmult;
    }
    panic!("model has no output-projection QuantMultA");
}

impl Weights {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Weights, String> {
        let model = Model::load(path).map_err(|e| e.to_string())?;
        Weights::new(model)
    }

    /// Like [`Weights::load`] but memory-maps the model file: weight tensors are
    /// views into the mapping rather than owned heap copies (feature `mmap`).
    #[cfg(feature = "mmap")]
    pub fn load_mmapped(path: impl AsRef<std::path::Path>) -> Result<Weights, String> {
        let model = Model::load_mmapped(path).map_err(|e| e.to_string())?;
        Weights::new(model)
    }

    pub fn new(mut model: Model) -> Result<Weights, String> {
        let yaml = model
            .get("special:model.yml")
            .map(|it| String::from_utf8_lossy(&it.data).into_owned())
            .unwrap_or_default();
        let mut config = Config::parse(&yaml);

        // Shared-vocab models (tied-embeddings-all) ship a single `Wemb` used for
        // source, target, and the output projection. Split-vocab models (CJK)
        // ship separate `encoder_Wemb` (source) and `decoder_Wemb` (target +
        // output projection).
        let (trg_wemb_param, src_wemb_param): (&'static str, &'static str) =
            if model.get("Wemb").is_some() {
                ("Wemb", "Wemb")
            } else {
                ("decoder_Wemb", "encoder_Wemb")
            };
        let trg_item = model
            .get(trg_wemb_param)
            .ok_or_else(|| format!("model has no {trg_wemb_param}"))?;
        let dim = *trg_item.shape.last().ok_or("embedding has no shape")? as usize;
        let trg_vocab = trg_item.num_elements() / dim;
        if config.vocab == 0 {
            config.vocab = trg_vocab;
        }

        // Cache the layer-norm scale/bias params (small, ~72 KB total) so the hot
        // path borrows them instead of decoding a fresh Vec per postnorm call.
        let mut layer_norms: HashMap<String, (Vec<f32>, Option<Vec<f32>>)> = HashMap::new();
        for it in &model.items {
            if let Some(base) = it.name.strip_suffix("_ln_scale") {
                if let Ok(v) = it.to_f32() {
                    layer_norms.entry(base.to_string()).or_default().0 = v;
                }
            } else if let Some(base) = it.name.strip_suffix("_ln_bias") {
                if let Ok(v) = it.to_f32() {
                    layer_norms.entry(base.to_string()).or_default().1 = Some(v);
                }
            }
        }

        #[cfg(not(feature = "lean-embed"))]
        {
            let trg_wemb = load_embedding(&model, trg_wemb_param)?;
            let src_wemb = if src_wemb_param == trg_wemb_param {
                None
            } else {
                Some(load_embedding(&model, src_wemb_param)?)
            };
            #[cfg(feature = "gemmology")]
            let affine_cache = RefCell::new(prepare_affines(&mut model, trg_wemb_param));
            Ok(Weights {
                model,
                config,
                trg_vocab,
                dim,
                layer_norms,
                trg_wemb_param,
                trg_wemb,
                src_wemb,
                #[cfg(feature = "gemmology")]
                affine_cache,
                #[cfg(feature = "gemmology")]
                scratch: RefCell::new(GemmScratch::default()),
            })
        }
        #[cfg(feature = "lean-embed")]
        {
            let qwemb = trg_item.quant_mult().map_err(|e| e.to_string())?;
            let src_inv_qmult = 1.0
                / model
                    .get(src_wemb_param)
                    .ok_or_else(|| format!("model has no {src_wemb_param}"))?
                    .quant_mult()
                    .map_err(|e| e.to_string())?;
            let proj_qa = read_output_qa(&model);
            let proj_unquant = 1.0 / (proj_qa * qwemb);
            // Prepared bias is static — fold the shift correction once here.
            let raw = trg_item.int8_transposed().map_err(|e| e.to_string())?;
            let raw_bias = model
                .get("decoder_ff_logit_out_b")
                .and_then(|it| it.to_f32().ok())
                .unwrap_or_else(|| vec![0.0; trg_vocab]);
            let proj_bias = ops::prepare_bias(raw, trg_vocab, dim, &raw_bias, proj_unquant);
            #[cfg(feature = "gemmology")]
            let affine_cache = RefCell::new(prepare_affines(&mut model, trg_wemb_param));

            // Pack the target `Wemb` for the output projection now (eagerly) rather
            // than on first projection. Embedding lookups happen on the first encode,
            // *before* any projection, so building it here lets those lookups read
            // their rows back out of the packed buffer — and when the pack succeeds
            // (SIMD kernel present, `k % 16 == 0`) we drop the raw int8 copy, since
            // the packed layout is a lossless reblocking that serves both. That
            // removes the last ~15.6 MiB "held twice" duplication. If the pack fails
            // (scalar build), `proj_pb` stays `None` and the raw copy is kept.
            #[cfg(feature = "gemmology")]
            let proj_pb = {
                let packed = {
                    let raw = model
                        .get(trg_wemb_param)
                        .expect("target embedding")
                        .int8_transposed()
                        .map_err(|e| e.to_string())?;
                    PreparedB::new(raw, trg_vocab, dim)
                };
                if packed.is_some() {
                    if let Some(it) = model.items.iter_mut().find(|it| it.name == trg_wemb_param) {
                        it.data = crate::model::Bytes::Owned(Vec::new());
                    }
                }
                RefCell::new(packed)
            };
            Ok(Weights {
                model,
                config,
                trg_vocab,
                dim,
                layer_norms,
                trg_wemb_param,
                src_wemb_param,
                src_inv_qmult,
                trg_inv_qmult: 1.0 / qwemb,
                proj_bias,
                proj_qa,
                proj_unquant,
                #[cfg(feature = "gemmology")]
                affine_cache,
                #[cfg(feature = "gemmology")]
                proj_pb,
                #[cfg(feature = "gemmology")]
                scratch: RefCell::new(GemmScratch::default()),
            })
        }
    }

    pub fn config(&self) -> Config {
        self.config
    }

    /// A float parameter by name (bias, layernorm scale/bias, …).
    pub fn f32(&self, name: &str) -> Option<Vec<f32>> {
        self.model.get(name).and_then(|it| it.to_f32().ok())
    }

    /// Output vocabulary size (number of tied-projection rows).
    pub fn output_vocab(&self) -> usize {
        self.trg_vocab
    }

    /// Full-vocabulary output logits `h · Wemb^T + bias`. Float table in the
    /// default build; full-vocab int8 GEMM (with the precomputed prepared bias)
    /// under `lean-embed`.
    pub fn full_logits(&self, h: &[f32]) -> Vec<f32> {
        self.full_logits_batch(h, 1)
    }

    /// Batched tied output projection: `h` is `[m, dim]` (m decoder tops stacked
    /// row-major), the result is `[m, vocab]`. Projecting the whole minibatch in
    /// one GEMM streams the large vocab weight once per batch instead of once per
    /// row — the matrix×matrix reuse win at the output layer. Rows are
    /// independent, so per-row results match [`full_logits`] exactly.
    pub fn full_logits_batch(&self, h: &[f32], m: usize) -> Vec<f32> {
        let mut out = Vec::new();
        self.full_logits_batch_into(h, m, &mut out);
        out
    }

    /// [`full_logits_batch`] into a caller-owned buffer (resized to `[m, vocab]`),
    /// reused across decode steps to avoid a fresh full-vocab allocation each time.
    #[cfg(not(feature = "lean-embed"))]
    pub fn full_logits_batch_into(&self, h: &[f32], m: usize, out: &mut Vec<f32>) {
        let d = self.dim;
        let vocab = self.trg_vocab;
        let bias = self
            .f32("decoder_ff_logit_out_b")
            .unwrap_or_else(|| vec![0.0; vocab]);
        out.clear();
        out.resize(m * vocab, 0.0);
        for row in 0..m {
            let hr = &h[row * d..(row + 1) * d];
            for v in 0..vocab {
                let w = &self.trg_wemb[v * d..(v + 1) * d];
                let mut acc = 0.0f32;
                for c in 0..d {
                    acc += hr[c] * w[c];
                }
                out[row * vocab + v] = acc + bias[v];
            }
        }
    }

    #[cfg(feature = "lean-embed")]
    pub fn full_logits_batch_into(&self, h: &[f32], m: usize, out: &mut Vec<f32>) {
        #[cfg(feature = "gemmology")]
        {
            // `proj_pb` is packed eagerly at load (see `Weights::new`); when present
            // it also backs embedding lookups, so the raw int8 copy is already gone.
            if let Some(pb) = self.proj_pb.borrow().as_ref() {
                let mut s = self.scratch.borrow_mut();
                ops::prepare_a_into(h, self.proj_qa, &mut s.a_u8);
                pb.matmul_into(&s.a_u8, m, self.proj_unquant, &self.proj_bias, out);
                return;
            }
        }
        let raw = self
            .model
            .get(self.trg_wemb_param)
            .expect("target embedding")
            .int8_transposed()
            .expect("int8 embedding");
        let a = ops::prepare_a(h, self.proj_qa);
        *out = ops::intgemm_affine(
            &a,
            m,
            self.dim,
            raw,
            self.trg_vocab,
            self.proj_unquant,
            &self.proj_bias,
        );
    }

    /// Layer-norm `scale` and optional `bias` for a sublayer base name (e.g.
    /// `encoder_l1_ffn`), borrowed from the load-time cache — no per-call decode.
    pub fn layer_norm(&self, base: &str) -> Option<(&[f32], Option<&[f32]>)> {
        self.layer_norms
            .get(base)
            .map(|(g, b)| (g.as_slice(), b.as_deref()))
    }

    /// Write one source (encoder) embedding row into `dst` (length `dim`): copied
    /// from the resident f32 table in the default build (shared vocab reuses the
    /// target table); dequantized on demand from the int8 tensor under
    /// `lean-embed`. No allocation — the caller owns `dst`.
    #[cfg(not(feature = "lean-embed"))]
    pub fn src_embed_row_into(&self, id: u32, dst: &mut [f32]) {
        let d = self.dim;
        let wemb = self.src_wemb.as_deref().unwrap_or(&self.trg_wemb);
        dst.copy_from_slice(&wemb[id as usize * d..(id as usize + 1) * d]);
    }

    #[cfg(feature = "lean-embed")]
    pub fn src_embed_row_into(&self, id: u32, dst: &mut [f32]) {
        self.dequant_row_into(self.src_wemb_param, self.src_inv_qmult, id, dst);
    }

    /// Write one target (decoder) embedding row into `dst` (length `dim`).
    #[cfg(not(feature = "lean-embed"))]
    pub fn trg_embed_row_into(&self, id: u32, dst: &mut [f32]) {
        let d = self.dim;
        dst.copy_from_slice(&self.trg_wemb[id as usize * d..(id as usize + 1) * d]);
    }

    #[cfg(feature = "lean-embed")]
    pub fn trg_embed_row_into(&self, id: u32, dst: &mut [f32]) {
        self.dequant_row_into(self.trg_wemb_param, self.trg_inv_qmult, id, dst);
    }

    /// Dequantize one embedding row from the int8 model tensor into `dst` (lean
    /// build) — `dst[c] = raw[c] · inv`, no allocation.
    #[cfg(feature = "lean-embed")]
    fn dequant_row_into(&self, param: &str, inv: f32, id: u32, dst: &mut [f32]) {
        let d = self.dim;
        // The target embedding is packed for the output projection and its raw int8
        // copy freed, so read the row back out of the packed buffer (a lossless
        // reblocking). The source embedding (split vocab) is never packed → raw.
        #[cfg(feature = "gemmology")]
        if param == self.trg_wemb_param {
            if let Some(pb) = self.proj_pb.borrow().as_ref() {
                let mut s = self.scratch.borrow_mut();
                s.wemb_row.resize(d, 0);
                pb.read_row(id as usize, &mut s.wemb_row);
                for (o, &b) in dst.iter_mut().zip(&s.wemb_row) {
                    *o = b as f32 * inv;
                }
                return;
            }
        }
        let raw = self
            .model
            .get(param)
            .expect("embedding param")
            .int8_transposed()
            .expect("int8 embedding");
        for (o, &b) in dst
            .iter_mut()
            .zip(&raw[id as usize * d..(id as usize + 1) * d])
        {
            *o = b as f32 * inv;
        }
    }

    /// Quant multiplier of the int8 target embedding, for the shortlist int8 output
    /// projection; `None` if the embedding shipped as float (no int8 projection).
    pub fn output_wemb_qmult(&self) -> Option<f32> {
        // Under lean-embed the raw copy may be freed once packed, so use the value
        // captured at load (`trg_inv_qmult = 1/qwemb`); lean-embed always loads int8.
        #[cfg(feature = "lean-embed")]
        {
            Some(1.0 / self.trg_inv_qmult)
        }
        #[cfg(not(feature = "lean-embed"))]
        {
            self.model.get(self.trg_wemb_param)?.quant_mult().ok()
        }
    }

    /// Gather one int8 target-embedding row (length `dim`) into `out`, for the
    /// shortlist projection's candidate weight matrix. Reads from the packed
    /// projection buffer when the raw copy has been freed (lean-embed + SIMD),
    /// else from the raw int8 tensor. Returns `false` if the embedding is float.
    pub fn output_wemb_int8_row(&self, id: u32, out: &mut [i8]) -> bool {
        #[cfg(all(feature = "lean-embed", feature = "gemmology"))]
        if let Some(pb) = self.proj_pb.borrow().as_ref() {
            pb.read_row(id as usize, out);
            return true;
        }
        let d = self.dim;
        match self
            .model
            .get(self.trg_wemb_param)
            .and_then(|it| it.int8_transposed().ok())
        {
            Some(raw) => {
                out.copy_from_slice(&raw[id as usize * d..(id as usize + 1) * d]);
                true
            }
            None => false,
        }
    }

    /// Activation quant-mult (qA) for the tied output projection
    /// ([`read_output_qa`]).
    pub fn output_qa(&self) -> f32 {
        read_output_qa(&self.model)
    }

    /// Run a shifted int8 affine `y = x·W + bias` from the weight's base name
    /// (e.g. `encoder_l0_self_Wq`). `x` is `[m, k]` row-major; the result is
    /// `[m, n]`. `bias_name` is the raw bias parameter, or `None` for the
    /// bias-less matmuls (SSRU's `W`), which use the fake-bias correction only.
    ///
    /// # Panics
    /// If the weight or its `*_QuantMultA` is missing, or shapes are inconsistent.
    pub fn affine(&self, base: &str, x: &[f32], m: usize, bias_name: Option<&str>) -> Vec<f32> {
        // Fast path: weight packed at load. Build the full prepared bias once
        // (correction + raw bias), then reuse the activation scratch and let the
        // shim reuse its own — a steady-state affine allocates only its output.
        #[cfg(feature = "gemmology")]
        {
            let mut cache = self.affine_cache.borrow_mut();
            if let Some(aw) = cache.get_mut(base) {
                if aw.bias.is_none() {
                    let mut bias = aw.correction.clone();
                    if let Some(bn) = bias_name {
                        if let Some(rb) = self.f32(bn) {
                            for (b, r) in bias.iter_mut().zip(rb.iter()) {
                                *b += *r;
                            }
                        }
                    }
                    aw.bias = Some(bias);
                }
                let mut s = self.scratch.borrow_mut();
                ops::prepare_a_into(x, aw.qa, &mut s.a_u8);
                let mut out = Vec::new();
                aw.pb
                    .matmul_into(&s.a_u8, m, aw.unquant, aw.bias.as_ref().unwrap(), &mut out);
                return out;
            }
        }

        // Scalar path: raw weight (gemmology off, or `k % 16 != 0` so it wasn't
        // packed and its raw bytes were kept).
        let w = self
            .model
            .get(base)
            .unwrap_or_else(|| panic!("missing weight {base}"));
        // Stored logical shape is [K, N]; data is transposed to [N, K].
        let k = w.shape[0] as usize;
        let n = w.shape[1] as usize;
        let b = w.int8_transposed().expect("int8 weight");
        debug_assert_eq!(b.len(), n * k);
        let qb = w.quant_mult().expect("weight quant mult");

        let qa = self
            .f32(&format!("{base}_QuantMultA"))
            .unwrap_or_else(|| panic!("missing {base}_QuantMultA"))[0];
        let unquant = 1.0 / (qa * qb);

        let raw_bias = match bias_name {
            Some(bn) => self.f32(bn).unwrap_or_else(|| panic!("missing bias {bn}")),
            None => vec![0.0; n],
        };
        let prepared = ops::prepare_bias(b, n, k, &raw_bias, unquant);

        let a = ops::prepare_a(x, qa);
        ops::intgemm_affine(&a, m, k, b, n, unquant, &prepared)
    }
}
