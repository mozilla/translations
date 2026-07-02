//! Model-weights view for dynamic execution.
//!
//! Wraps [`crate::model::Model`] and resolves parameters by marian's naming
//! convention, exposing exactly what the transformer forward needs:
//! - [`Weights::affine`] runs a shifted int8 affine end to end (quantize the
//!   activation, prepare the bias, integer GEMM, unquantize) from a weight's
//!   base name — the `unquant = 1/(qA·qB)` multiplier is computed from the model's
//!   own quant multipliers, nothing from the trace.
//! - [`Weights::embed_row`] / [`Weights::wemb`] give the dequantized embedding
//!   matrix (marian dequantizes `Wemb` at load, `integer_common.h:unquantizeWemb`).
//! - [`Weights::f32`] returns float parameters (biases, layernorm scale/bias).
//! - [`Config`] holds the architecture dims parsed from `special:model.yml`.

use crate::model::Model;
use crate::ops;
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
pub struct Weights {
    model: Model,
    config: Config,
    /// Source (encoder) embedding, dequantized `[src_vocab, dim]`, row-major.
    src_wemb: Vec<f32>,
    /// Target (decoder) embedding, dequantized `[trg_vocab, dim]`. Also the tied
    /// output-projection weight. For shared-vocab models this is the same data
    /// as `src_wemb`.
    trg_wemb: Vec<f32>,
    trg_vocab: usize,
    /// Raw int8 target embedding + quant multiplier, for the int8 output
    /// projection (shortlist path); `None` if it shipped as float.
    trg_wemb_i8: Option<Vec<i8>>,
    trg_qmult: f32,
    dim: usize,
}

/// A dequantized embedding matrix loaded from the model.
struct Embedding {
    data: Vec<f32>,
    raw_i8: Option<Vec<i8>>,
    qmult: f32,
    vocab: usize,
    dim: usize,
}

/// Load and dequantize an embedding parameter `[vocab, dim]` (int8/quantMult, or
/// float if it shipped dequantized).
fn load_embedding(model: &Model, name: &str) -> Result<Embedding, String> {
    let item = model
        .get(name)
        .ok_or_else(|| format!("model has no {name}"))?;
    let dim = *item.shape.last().ok_or("embedding has no shape")? as usize;
    let vocab = item.num_elements() / dim;
    let qmult = item.quant_mult().unwrap_or(1.0);
    let inv = 1.0 / qmult;
    let (data, raw_i8) = match item.dtype {
        DType::Float32 => (item.to_f32().map_err(|e| e.to_string())?, None),
        _ => {
            let raw = item.int8_transposed().map_err(|e| e.to_string())?;
            (
                raw.iter().map(|&b| b as f32 * inv).collect(),
                Some(raw.to_vec()),
            )
        }
    };
    Ok(Embedding {
        data,
        raw_i8,
        qmult,
        vocab,
        dim,
    })
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
        // tied output projection).
        let (src, trg) = if model.get("Wemb").is_some() {
            let shared = load_embedding(&model, "Wemb")?;
            let src = Embedding {
                data: shared.data.clone(),
                raw_i8: shared.raw_i8.clone(),
                qmult: shared.qmult,
                vocab: shared.vocab,
                dim: shared.dim,
            };
            (src, shared)
        } else {
            (
                load_embedding(&model, "encoder_Wemb")?,
                load_embedding(&model, "decoder_Wemb")?,
            )
        };
        let dim = trg.dim;
        if config.vocab == 0 {
            config.vocab = trg.vocab;
        }

        Ok(Weights {
            model,
            config,
            src_wemb: src.data,
            trg_wemb: trg.data,
            trg_vocab: trg.vocab,
            trg_wemb_i8: trg.raw_i8,
            trg_qmult: trg.qmult,
            dim,
        })
    }

    pub fn config(&self) -> Config {
        self.config
    }

    /// A float parameter by name (bias, layernorm scale/bias, …).
    pub fn f32(&self, name: &str) -> Option<Vec<f32>> {
        self.model.get(name).and_then(|it| it.to_f32().ok())
    }

    /// The target embedding `[trg_vocab, dim]` — the tied output-projection
    /// weight — and its dims.
    pub fn output_wemb(&self) -> (&[f32], usize, usize) {
        (&self.trg_wemb, self.trg_vocab, self.dim)
    }

    /// One source (encoder) embedding row.
    pub fn src_embed_row(&self, id: u32) -> &[f32] {
        let d = self.dim;
        &self.src_wemb[id as usize * d..(id as usize + 1) * d]
    }

    /// One target (decoder) embedding row.
    pub fn trg_embed_row(&self, id: u32) -> &[f32] {
        let d = self.dim;
        &self.trg_wemb[id as usize * d..(id as usize + 1) * d]
    }

    /// The raw int8 target embedding and its quant multiplier, for the int8
    /// output projection. `None` if it shipped as float.
    pub fn output_wemb_int8(&self) -> Option<(&[i8], f32)> {
        self.trg_wemb_i8.as_deref().map(|w| (w, self.trg_qmult))
    }

    /// Activation quant-mult (qA) for the tied output projection. Shared-vocab
    /// models name that node "none" (`none_QuantMultA`, a plain float32 scalar).
    /// Split-vocab (CJK) models name it `decoder_Wemb_QuantMultA` and store it as
    /// an `intgemm8` scalar: a single int8 value (127) plus an appended quant
    /// multiplier, whose *dequantized* value (`127 / quant_mult`) is the alpha.
    pub fn output_qa(&self) -> f32 {
        if let Some(v) = self.f32("none_QuantMultA") {
            return v[0];
        }
        if let Some(it) = self.model.get("decoder_Wemb_QuantMultA") {
            let raw = it.int8_transposed().expect("intgemm8 alpha scalar")[0] as f32;
            let qmult = it.quant_mult().expect("intgemm8 alpha quant mult");
            return raw / qmult;
        }
        panic!("model has no output-projection QuantMultA");
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
