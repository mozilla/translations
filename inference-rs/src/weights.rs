//! Model-weights view for dynamic execution.
//!
//! Wraps [`crate::model::Model`] and resolves parameters by marian's naming
//! convention, exposing exactly what the transformer forward needs:
//! - [`Weights::affine`] runs a shifted int8 affine end to end (quantize the
//!   activation, prepare the bias, integer GEMM, unquantize) from a weight's
//!   base name — the `unquant = 1/(qA·qB)` multiplier is computed from the model's
//!   own quant multipliers, nothing from the trace.
//! - [`Weights::src_embed_row`] / [`Weights::trg_embed_row`] give one embedding
//!   row, and [`Weights::full_logits`] the tied output projection. Their backing
//!   representation (resident dequantized f32 tables vs. on-the-fly int8) is
//!   chosen by the `lean-embed` feature.
//! - [`Weights::f32`] returns float parameters (biases, layernorm scale/bias).
//! - [`Config`] holds the architecture dims parsed from `special:model.yml`.

use std::borrow::Cow;

use crate::model::Model;
use crate::ops;
#[cfg(not(feature = "lean-embed"))]
use crate::trace::DType;

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
///   perf is still to be measured, hence gated (see 06-memory-approach.md §7).
pub struct Weights {
    model: Model,
    config: Config,
    trg_vocab: usize,
    dim: usize,
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

    pub fn new(model: Model) -> Result<Weights, String> {
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

        #[cfg(not(feature = "lean-embed"))]
        {
            let trg_wemb = load_embedding(&model, trg_wemb_param)?;
            let src_wemb = if src_wemb_param == trg_wemb_param {
                None
            } else {
                Some(load_embedding(&model, src_wemb_param)?)
            };
            Ok(Weights {
                model,
                config,
                trg_vocab,
                dim,
                trg_wemb_param,
                trg_wemb,
                src_wemb,
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
            Ok(Weights {
                model,
                config,
                trg_vocab,
                dim,
                trg_wemb_param,
                src_wemb_param,
                src_inv_qmult,
                trg_inv_qmult: 1.0 / qwemb,
                proj_bias,
                proj_qa,
                proj_unquant,
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
    #[cfg(not(feature = "lean-embed"))]
    pub fn full_logits(&self, h: &[f32]) -> Vec<f32> {
        let d = self.dim;
        let vocab = self.trg_vocab;
        let bias = self
            .f32("decoder_ff_logit_out_b")
            .unwrap_or_else(|| vec![0.0; vocab]);
        let mut logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            let row = &self.trg_wemb[v * d..(v + 1) * d];
            let mut acc = 0.0f32;
            for c in 0..d {
                acc += h[c] * row[c];
            }
            logits[v] = acc + bias[v];
        }
        logits
    }

    #[cfg(feature = "lean-embed")]
    pub fn full_logits(&self, h: &[f32]) -> Vec<f32> {
        let raw = self
            .model
            .get(self.trg_wemb_param)
            .expect("target embedding")
            .int8_transposed()
            .expect("int8 embedding");
        let a = ops::prepare_a(h, self.proj_qa);
        ops::intgemm_affine(
            &a,
            1,
            self.dim,
            raw,
            self.trg_vocab,
            self.proj_unquant,
            &self.proj_bias,
        )
    }

    /// One source (encoder) embedding row. Borrowed from the resident f32 table
    /// in the default build (shared vocab reuses the target table); dequantized on
    /// demand from the int8 model tensor under `lean-embed`.
    #[cfg(not(feature = "lean-embed"))]
    pub fn src_embed_row(&self, id: u32) -> Cow<'_, [f32]> {
        let d = self.dim;
        let wemb = self.src_wemb.as_deref().unwrap_or(&self.trg_wemb);
        Cow::Borrowed(&wemb[id as usize * d..(id as usize + 1) * d])
    }

    #[cfg(feature = "lean-embed")]
    pub fn src_embed_row(&self, id: u32) -> Cow<'_, [f32]> {
        Cow::Owned(self.dequant_row(self.src_wemb_param, self.src_inv_qmult, id))
    }

    /// One target (decoder) embedding row.
    #[cfg(not(feature = "lean-embed"))]
    pub fn trg_embed_row(&self, id: u32) -> Cow<'_, [f32]> {
        let d = self.dim;
        Cow::Borrowed(&self.trg_wemb[id as usize * d..(id as usize + 1) * d])
    }

    #[cfg(feature = "lean-embed")]
    pub fn trg_embed_row(&self, id: u32) -> Cow<'_, [f32]> {
        Cow::Owned(self.dequant_row(self.trg_wemb_param, self.trg_inv_qmult, id))
    }

    /// Dequantize one embedding row from the int8 model tensor (lean build).
    #[cfg(feature = "lean-embed")]
    fn dequant_row(&self, param: &str, inv: f32, id: u32) -> Vec<f32> {
        let d = self.dim;
        let raw = self
            .model
            .get(param)
            .expect("embedding param")
            .int8_transposed()
            .expect("int8 embedding");
        raw[id as usize * d..(id as usize + 1) * d]
            .iter()
            .map(|&b| b as f32 * inv)
            .collect()
    }

    /// The raw int8 target embedding and its quant multiplier, for the int8
    /// output projection. Read from the model on demand (no separate copy kept);
    /// `None` if the embedding shipped as float.
    pub fn output_wemb_int8(&self) -> Option<(&[i8], f32)> {
        let it = self.model.get(self.trg_wemb_param)?;
        let raw = it.int8_transposed().ok()?;
        let qmult = it.quant_mult().ok()?;
        Some((raw, qmult))
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
