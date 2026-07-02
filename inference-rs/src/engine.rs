//! Dynamic transformer execution and greedy decoding (finalize-plan.md §2–§7).
//!
//! This assembles the ops into the Bergamot architecture (see
//! [fx-model-architecture.md](../fx-model-architecture.md)) for arbitrary input —
//! no trace involved. A 6-layer transformer **encoder** (bidirectional
//! self-attention + FFN) runs once; a 4-layer **SSRU decoder** (recurrent cell +
//! cross-attention + FFN) runs per step, carrying one cell-state vector per
//! layer. Greedy decoding argmaxes the tied output projection each step until
//! EOS.
//!
//! All GEMMs go through the shifted int8 affine ([`Weights::affine`]); the clean
//! float parts (layernorm, softmax, attention, elementwise) use [`crate::ops`].
//! Single sentence, batch 1 — so there is no padding and no attention mask.

use std::f32::consts::FRAC_PI_2;

use crate::ops;
use crate::spm::SpmVocab;
use crate::weights::{Config, Weights};

/// Layer-norm epsilon (layers/generic.h:463).
const EPS: f32 = 1e-6;

/// The translation engine: model weights + source/target vocabularies.
pub struct Engine {
    weights: Weights,
    src_vocab: SpmVocab,
    trg_vocab: SpmVocab,
    config: Config,
    /// Sinusoidal positional-encoding frequencies and offsets, length `dim`.
    pe_freq: Vec<f32>,
    pe_offs: Vec<f32>,
}

impl Engine {
    /// Build from a model file and the source/target `.spm` vocabularies. For
    /// most pairs the two vocab paths are identical; for CJK they differ.
    pub fn load(
        model_path: impl AsRef<std::path::Path>,
        src_vocab_path: impl AsRef<std::path::Path>,
        trg_vocab_path: impl AsRef<std::path::Path>,
    ) -> Result<Engine, String> {
        let weights = Weights::load(model_path)?;
        let src_vocab = SpmVocab::load(src_vocab_path).map_err(|e| e.to_string())?;
        let trg_vocab = SpmVocab::load(trg_vocab_path).map_err(|e| e.to_string())?;
        Ok(Engine::new(weights, src_vocab, trg_vocab))
    }

    pub fn new(weights: Weights, src_vocab: SpmVocab, trg_vocab: SpmVocab) -> Engine {
        let config = weights.config();
        let d = config.dim_emb;
        let t = d / 2;
        // PE(pos)[c] = sin(pos*freq[c] + offs[c]); rotor form (transformer.h:95).
        let mut pe_freq = vec![0.0f32; d];
        let mut pe_offs = vec![0.0f32; d];
        for c in 0..d {
            pe_freq[c] = 1e-4f32.powf((c % t) as f32 / (t as f32 - 1.0));
            pe_offs[c] = (c / t) as f32 * FRAC_PI_2;
        }
        Engine {
            weights,
            src_vocab,
            trg_vocab,
            config,
            pe_freq,
            pe_offs,
        }
    }

    /// Tokenize `text`, run greedy decoding, and detokenize the result.
    pub fn translate(&self, text: &str) -> String {
        let src_ids = self.src_vocab.encode_with_eos(text);
        let out_ids = self.greedy(&src_ids);
        self.trg_vocab.decode(&out_ids)
    }

    /// Greedy decode: encode the source, then argmax the tied projection each
    /// step, carrying SSRU cell state, until EOS or the length cap.
    pub fn greedy(&self, src_ids: &[u32]) -> Vec<u32> {
        let d = self.config.dim_emb;
        let seq = src_ids.len();
        let context = self.encode(src_ids);
        let eos = self.trg_vocab.eos_id();

        // One SSRU cell-state vector per decoder layer (1-based index; slot 0 unused).
        let mut cells = vec![vec![0.0f32; d]; self.config.dec_depth + 1];

        // max output length = ceil(factor * src_len), capped (config max-length-factor 2.0).
        let max_len = ((2.0 * seq as f32).ceil() as usize + 4).min(256);

        let mut out = Vec::new();
        let mut prev = eos; // decoder is seeded with EOS
        for step in 0..max_len {
            let top = self.decode_step(prev, step, &context, seq, &mut cells);
            let logits = self.project(&top);
            let next = argmax(&logits);
            if next == eos {
                break;
            }
            out.push(next);
            prev = next;
        }
        out
    }

    // --- encoder -------------------------------------------------------------

    /// Run the encoder over the source ids, returning the context `[seq, dim]`.
    pub fn encode(&self, src_ids: &[u32]) -> Vec<f32> {
        let seq = src_ids.len();
        let mut x = self.embed(src_ids, 0);
        for layer in 1..=self.config.enc_depth {
            x = self.encoder_layer(layer, &x, seq);
        }
        x
    }

    fn encoder_layer(&self, layer: usize, x: &[f32], seq: usize) -> Vec<f32> {
        let p = format!("encoder_l{layer}");
        // Self-attention sublayer: LayerNorm(x + SelfAttn(x)).
        let attn = self.multihead(&format!("{p}_self"), x, x, seq, seq);
        let x = self.postnorm(&attn, x, seq, &format!("{p}_self_Wo"));
        // FFN sublayer.
        self.ffn(&format!("{p}_ffn"), &x, seq)
    }

    // --- decoder -------------------------------------------------------------

    /// One decoder step for the token `prev_id` at output position `pos`.
    /// Updates the per-layer SSRU cell states and returns the top output `[dim]`.
    pub fn decode_step(
        &self,
        prev_id: u32,
        pos: usize,
        context: &[f32],
        seq: usize,
        cells: &mut [Vec<f32>],
    ) -> Vec<f32> {
        let mut x = self.embed(&[prev_id], pos);
        for layer in 1..=self.config.dec_depth {
            let p = format!("decoder_l{layer}");
            // SSRU autoregressive sublayer.
            let cand = self.weights.affine(&format!("{p}_rnn_W"), &x, 1, None); // x̃ = u·W
            let gate = self
                .weights
                .affine(&format!("{p}_rnn_Wf"), &x, 1, Some(&format!("{p}_rnn_bf"))); // f = u·Wf + bf
            // c = σ(f)·c_prev + (1−σ(f))·x̃ ; h = ReLU(c)
            let c = ops::highway(&cells[layer], &cand, &gate);
            cells[layer] = c.clone();
            let h = ops::relu(&c);
            let x_self = self.postnorm(&h, &x, 1, &format!("{p}_rnn_ffn"));
            // Cross-attention to the encoder context.
            let attn = self.multihead(&format!("{p}_context"), &x_self, context, 1, seq);
            let x_ctx = self.postnorm(&attn, &x_self, 1, &format!("{p}_context_Wo"));
            // FFN.
            x = self.ffn(&format!("{p}_ffn"), &x_ctx, 1);
        }
        x
    }

    /// Tied output projection: `logits[v] = h · Wemb[v] + b_out[v]` over the full
    /// vocabulary (full-vocab float path; the reference restricts to a shortlist,
    /// but greedy argmax over the full vocab matches on the happy path).
    pub fn project(&self, h: &[f32]) -> Vec<f32> {
        let (wemb, vocab, d) = self.weights.wemb();
        let bias = self
            .weights
            .f32("decoder_ff_logit_out_b")
            .unwrap_or_else(|| vec![0.0; vocab]);
        let mut logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            let row = &wemb[v * d..(v + 1) * d];
            let mut acc = 0.0f32;
            for c in 0..d {
                acc += h[c] * row[c];
            }
            logits[v] = acc + bias[v];
        }
        logits
    }

    // --- shared sublayers ----------------------------------------------------

    /// Multi-head attention. `q_in` is `[q_len, dim]`, `kv_in` is `[kv_len, dim]`.
    /// `prefix` supplies `{prefix}_W{q,k,v,o}` and `{prefix}_b{q,k,v,o}`.
    fn multihead(&self, prefix: &str, q_in: &[f32], kv_in: &[f32], q_len: usize, kv_len: usize) -> Vec<f32> {
        let d = self.config.dim_emb;
        let h = self.config.heads;
        let dk = d / h;
        let scale = 1.0 / (dk as f32).sqrt();

        let q = self.weights.affine(&format!("{prefix}_Wq"), q_in, q_len, Some(&format!("{prefix}_bq")));
        let k = self.weights.affine(&format!("{prefix}_Wk"), kv_in, kv_len, Some(&format!("{prefix}_bk")));
        let v = self.weights.affine(&format!("{prefix}_Wv"), kv_in, kv_len, Some(&format!("{prefix}_bv")));

        let mut joined = vec![0.0f32; q_len * d];
        let mut scores = vec![0.0f32; kv_len];
        for head in 0..h {
            let off = head * dk;
            for i in 0..q_len {
                let qh = &q[i * d + off..i * d + off + dk];
                // scaled dot-product scores over all kv positions
                for j in 0..kv_len {
                    let kh = &k[j * d + off..j * d + off + dk];
                    let dot: f32 = qh.iter().zip(kh).map(|(&a, &b)| a * b).sum();
                    scores[j] = dot * scale;
                }
                let weights = ops::softmax(&scores, 1, kv_len);
                // weighted sum of values
                let out = &mut joined[i * d + off..i * d + off + dk];
                for (j, &w) in weights.iter().enumerate() {
                    let vh = &v[j * d + off..j * d + off + dk];
                    for c in 0..dk {
                        out[c] += w * vh[c];
                    }
                }
            }
        }
        // output projection
        self.weights.affine(&format!("{prefix}_Wo"), &joined, q_len, Some(&format!("{prefix}_bo")))
    }

    /// FFN sublayer: `LayerNorm(x + W2·ReLU(W1·x))`. `prefix` e.g. `encoder_l1_ffn`.
    fn ffn(&self, prefix: &str, x: &[f32], seq: usize) -> Vec<f32> {
        let hidden = self.weights.affine(&format!("{prefix}_W1"), x, seq, Some(&format!("{prefix}_b1")));
        let hidden = ops::relu(&hidden);
        let inner = self.config.dim_ffn;
        let rows = hidden.len() / inner;
        debug_assert_eq!(rows, seq);
        let out = self.weights.affine(&format!("{prefix}_W2"), &hidden, seq, Some(&format!("{prefix}_b2")));
        self.postnorm(&out, x, seq, &format!("{prefix}_ffn"))
    }

    /// Post-norm residual: `LayerNorm(branch + residual)` with the `{ln}_ln_*`
    /// scale/bias params.
    fn postnorm(&self, branch: &[f32], residual: &[f32], rows: usize, ln: &str) -> Vec<f32> {
        let d = self.config.dim_emb;
        let sum: Vec<f32> = branch.iter().zip(residual).map(|(&a, &b)| a + b).collect();
        let gamma = self
            .weights
            .f32(&format!("{ln}_ln_scale"))
            .unwrap_or_else(|| panic!("missing {ln}_ln_scale"));
        let beta = self.weights.f32(&format!("{ln}_ln_bias"));
        ops::layer_normalization(&sum, &gamma, beta.as_deref(), rows, d, EPS)
    }

    // --- embeddings ----------------------------------------------------------

    /// Embed a run of token ids at consecutive positions starting at `start`:
    /// `x_t = √d · Wemb[id_t] + PE(start + t)`.
    fn embed(&self, ids: &[u32], start: usize) -> Vec<f32> {
        let d = self.config.dim_emb;
        let scale = (d as f32).sqrt();
        let mut out = vec![0.0f32; ids.len() * d];
        for (t, &id) in ids.iter().enumerate() {
            let row = self.weights.embed_row(id);
            let pos = (start + t) as f32;
            let dst = &mut out[t * d..(t + 1) * d];
            for c in 0..d {
                dst[c] = scale * row[c] + (pos * self.pe_freq[c] + self.pe_offs[c]).sin();
            }
        }
        out
    }
}

/// Index of the maximum element (first on ties).
fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}
