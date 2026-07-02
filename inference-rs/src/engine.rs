//! Dynamic transformer execution and greedy decoding.
//!
//! Assembles the ops into the Bergamot architecture for arbitrary input: a
//! 6-layer transformer **encoder** (bidirectional self-attention + FFN) runs
//! once; a 4-layer **SSRU decoder** (recurrent cell + cross-attention + FFN)
//! runs per step, carrying one cell-state vector per layer. Greedy decoding
//! argmaxes the tied output projection each step until EOS.
//!
//! All GEMMs go through the shifted int8 affine ([`Weights::affine`]); the clean
//! float parts (layernorm, softmax, attention, elementwise) use [`crate::ops`].
//! Single sentence, batch 1 — so there is no padding and no attention mask.

use std::f32::consts::FRAC_PI_2;

use crate::ops;
use crate::shortlist::Shortlist;
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
    /// Optional lexical shortlist restricting the output vocabulary per sentence.
    shortlist: Option<Shortlist>,
    /// Whether source and target share a vocabulary (affects shortlist candidates).
    shared_vocab: bool,
}

impl Engine {
    /// Build from a model file and the source/target `.spm` vocabularies. For
    /// most pairs the two vocab paths are identical; for CJK they differ.
    pub fn load(
        model_path: impl AsRef<std::path::Path>,
        src_vocab_path: impl AsRef<std::path::Path>,
        trg_vocab_path: impl AsRef<std::path::Path>,
    ) -> Result<Engine, String> {
        let shared = src_vocab_path.as_ref() == trg_vocab_path.as_ref();
        let weights = Weights::load(model_path)?;
        let src_vocab = SpmVocab::load(src_vocab_path).map_err(|e| e.to_string())?;
        let trg_vocab = SpmVocab::load(trg_vocab_path).map_err(|e| e.to_string())?;
        let mut engine = Engine::new(weights, src_vocab, trg_vocab);
        engine.shared_vocab = shared;
        Ok(engine)
    }

    /// Attach a lexical shortlist so decoding restricts the output vocabulary to
    /// the per-sentence candidate set (required for exact reference parity).
    pub fn with_shortlist(mut self, shortlist: Shortlist) -> Engine {
        self.shortlist = Some(shortlist);
        self
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
            shortlist: None,
            shared_vocab: true,
        }
    }

    /// Expose source tokenization for debugging.
    pub fn src_ids(&self, text: &str) -> Vec<u32> {
        self.src_vocab.encode_with_eos(text)
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

        // The shortlist candidate set is per-sentence — computed once. Split
        // vocabs (CJK) pass `shared = false`, so source token ids are not copied
        // into the target candidate set; the lexical translations still are.
        let candidates = self
            .shortlist
            .as_ref()
            .map(|s| s.candidates(src_ids, self.shared_vocab));

        let mut out = Vec::new();
        let mut prev = eos; // decoder is seeded with EOS
        for step in 0..max_len {
            let top = self.decode_step(prev, step, &context, seq, &mut cells);
            let next = self.project_argmax(&top, candidates.as_deref());
            if next == eos {
                break;
            }
            out.push(next);
            prev = next;
        }
        out
    }

    /// Project the decoder top and return the argmax token id. With a shortlist
    /// and a quantized `Wemb`, this runs the reference's int8 projection over the
    /// candidate columns (the `SelectColumnsB` path) for exact parity; otherwise
    /// it falls back to the full-vocab float projection.
    fn project_argmax(&self, h: &[f32], candidates: Option<&[u32]>) -> u32 {
        match (candidates, self.weights.output_wemb_int8()) {
            (Some(cands), Some((wemb_i8, qwemb))) => self.project_int8(h, cands, wemb_i8, qwemb),
            (Some(cands), None) => argmax_restricted(&self.project(h), cands),
            (None, _) => argmax(&self.project(h)),
        }
    }

    /// The int8 tied output projection restricted to `candidates`, matching the
    /// reference's `intgemmSelectColumnsB` + affine. Returns the best candidate's
    /// full-vocabulary id.
    fn project_int8(&self, h: &[f32], candidates: &[u32], wemb_i8: &[i8], qwemb: f32) -> u32 {
        let d = self.config.dim_emb;
        let n = candidates.len();
        // qA for the decoder-top activation feeding the tied projection.
        let qa = self.weights.output_qa();
        let unquant = 1.0 / (qa * qwemb);

        // Gather candidate embedding rows as the [N, K] weight, and their biases.
        let bias_full = self
            .weights
            .f32("decoder_ff_logit_out_b")
            .unwrap_or_else(|| vec![0.0; self.weights.output_vocab()]);
        let mut b_transposed = vec![0i8; n * d];
        let mut raw_bias = vec![0.0f32; n];
        for (j, &c) in candidates.iter().enumerate() {
            let row = &wemb_i8[c as usize * d..(c as usize + 1) * d];
            b_transposed[j * d..(j + 1) * d].copy_from_slice(row);
            raw_bias[j] = bias_full[c as usize];
        }
        let prepared = ops::prepare_bias(&b_transposed, n, d, &raw_bias, unquant);
        let a = ops::prepare_a(h, qa);
        let logits = ops::intgemm_affine(&a, 1, d, &b_transposed, n, unquant, &prepared);

        // argmax over candidates -> map back to the vocabulary id.
        let mut best = 0usize;
        for j in 1..n {
            if logits[j] > logits[best] {
                best = j;
            }
        }
        candidates[best]
    }

    // --- encoder -------------------------------------------------------------

    /// Run the encoder over the source ids, returning the context `[seq, dim]`.
    pub fn encode(&self, src_ids: &[u32]) -> Vec<f32> {
        let seq = src_ids.len();
        let mut x = self.embed(src_ids, 0, Side::Source);
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
        let mut x = self.embed(&[prev_id], pos, Side::Target);
        for layer in 1..=self.config.dec_depth {
            let p = format!("decoder_l{layer}");
            // SSRU autoregressive sublayer.
            let cand = self.weights.affine(&format!("{p}_rnn_W"), &x, 1, None); // x̃ = u·W
            let gate =
                self.weights
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

    /// Tied output projection over the full target vocabulary,
    /// `logits[v] = h · Wemb[v] + b_out[v]`. Delegates to
    /// [`Weights::full_logits`], whose representation (resident f32 table vs.
    /// on-the-fly int8) is chosen by the `lean-embed` feature.
    pub fn project(&self, h: &[f32]) -> Vec<f32> {
        self.weights.full_logits(h)
    }

    // --- shared sublayers ----------------------------------------------------

    /// Multi-head attention. `q_in` is `[q_len, dim]`, `kv_in` is `[kv_len, dim]`.
    /// `prefix` supplies `{prefix}_W{q,k,v,o}` and `{prefix}_b{q,k,v,o}`.
    fn multihead(
        &self,
        prefix: &str,
        q_in: &[f32],
        kv_in: &[f32],
        q_len: usize,
        kv_len: usize,
    ) -> Vec<f32> {
        let d = self.config.dim_emb;
        let h = self.config.heads;
        let dk = d / h;
        let scale = 1.0 / (dk as f32).sqrt();

        let q = self.weights.affine(
            &format!("{prefix}_Wq"),
            q_in,
            q_len,
            Some(&format!("{prefix}_bq")),
        );
        let k = self.weights.affine(
            &format!("{prefix}_Wk"),
            kv_in,
            kv_len,
            Some(&format!("{prefix}_bk")),
        );
        let v = self.weights.affine(
            &format!("{prefix}_Wv"),
            kv_in,
            kv_len,
            Some(&format!("{prefix}_bv")),
        );

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
        self.weights.affine(
            &format!("{prefix}_Wo"),
            &joined,
            q_len,
            Some(&format!("{prefix}_bo")),
        )
    }

    /// FFN sublayer: `LayerNorm(x + W2·ReLU(W1·x))`. `prefix` e.g. `encoder_l1_ffn`.
    fn ffn(&self, prefix: &str, x: &[f32], seq: usize) -> Vec<f32> {
        let hidden = self.weights.affine(
            &format!("{prefix}_W1"),
            x,
            seq,
            Some(&format!("{prefix}_b1")),
        );
        let hidden = ops::relu(&hidden);
        let inner = self.config.dim_ffn;
        let rows = hidden.len() / inner;
        debug_assert_eq!(rows, seq);
        let out = self.weights.affine(
            &format!("{prefix}_W2"),
            &hidden,
            seq,
            Some(&format!("{prefix}_b2")),
        );
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
    /// `x_t = √d · Wemb[id_t] + PE(start + t)`. `side` selects the source
    /// (encoder) or target (decoder) embedding — the same matrix for shared-vocab
    /// models, distinct for split-vocab (CJK) ones.
    fn embed(&self, ids: &[u32], start: usize, side: Side) -> Vec<f32> {
        let d = self.config.dim_emb;
        let scale = (d as f32).sqrt();
        let mut out = vec![0.0f32; ids.len() * d];
        for (t, &id) in ids.iter().enumerate() {
            let row = match side {
                Side::Source => self.weights.src_embed_row(id),
                Side::Target => self.weights.trg_embed_row(id),
            };
            let pos = (start + t) as f32;
            let dst = &mut out[t * d..(t + 1) * d];
            for c in 0..d {
                dst[c] = scale * row[c] + (pos * self.pe_freq[c] + self.pe_offs[c]).sin();
            }
        }
        out
    }
}

/// Which embedding matrix a lookup uses.
#[derive(Clone, Copy)]
enum Side {
    Source,
    Target,
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

/// Argmax over only the candidate ids (the shortlist restriction), returning a
/// full-vocabulary id.
fn argmax_restricted(logits: &[f32], candidates: &[u32]) -> u32 {
    let mut best = candidates[0];
    let mut best_val = logits[best as usize];
    for &c in &candidates[1..] {
        let val = logits[c as usize];
        if val > best_val {
            best_val = val;
            best = c;
        }
    }
    best
}
