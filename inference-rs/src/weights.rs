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
    /// Dequantized embedding matrix `[vocab, dim]`, row-major.
    wemb: Vec<f32>,
    /// Raw int8 embedding matrix `[vocab, dim]` and its quant multiplier, when
    /// `Wemb` shipped quantized — used for the int8 (tied) output projection.
    wemb_i8: Option<Vec<i8>>,
    wemb_qmult: f32,
    vocab: usize,
    dim: usize,
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

        // Dequantize Wemb: on disk it is row-major int8 [vocab, dim] with the
        // quant multiplier appended; float = int8 / quantMult.
        let wemb_item = model.get("Wemb").ok_or("model has no Wemb")?;
        let dim = *wemb_item.shape.last().ok_or("Wemb has no shape")? as usize;
        let vocab = wemb_item.num_elements() / dim;
        let q = wemb_item.quant_mult().unwrap_or(1.0);
        let inv = 1.0 / q;
        let (wemb, wemb_i8) = match wemb_item.dtype {
            DType::Float32 => (wemb_item.to_f32().map_err(|e| e.to_string())?, None),
            _ => {
                let raw = wemb_item.int8_transposed().map_err(|e| e.to_string())?;
                let f = raw.iter().map(|&b| b as f32 * inv).collect();
                (f, Some(raw.to_vec()))
            }
        };
        if config.vocab == 0 {
            config.vocab = vocab;
        }

        Ok(Weights {
            model,
            config,
            wemb,
            wemb_i8,
            wemb_qmult: q,
            vocab,
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

    /// The dequantized embedding matrix `[vocab, dim]` and its dims.
    pub fn wemb(&self) -> (&[f32], usize, usize) {
        (&self.wemb, self.vocab, self.dim)
    }

    /// One embedding row (per-token lookup).
    pub fn embed_row(&self, id: u32) -> &[f32] {
        let d = self.dim;
        &self.wemb[id as usize * d..(id as usize + 1) * d]
    }

    /// The raw int8 embedding matrix `[vocab, dim]` and its quant multiplier,
    /// for the int8 tied output projection. `None` if `Wemb` shipped as float.
    pub fn wemb_int8(&self) -> Option<(&[i8], f32)> {
        self.wemb_i8.as_deref().map(|w| (w, self.wemb_qmult))
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
