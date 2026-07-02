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
//! Greedy decoding is single-sentence; [`Engine::encode_batch`] adds a padded,
//! mask-attention batched encoder for translating a block of sentences together.

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

/// Per-sentence wall-clock timing from [`Engine::translate_timed`], in the spans
/// the perf harness reports: encode, time-to-first-token, and full decode.
pub struct Timing {
    /// Source tokenization + encoder pass.
    pub encode_ms: f64,
    /// Decode time to the first emitted token (excludes encode). TTFT is
    /// `encode_ms + first_token_ms`.
    pub first_token_ms: f64,
    /// Total greedy-loop time.
    pub decode_ms: f64,
    /// Tokens generated (excludes the terminal EOS).
    pub out_tokens: usize,
}

/// Per-block wall-clock timing from [`Engine::translate_batch_timed`].
pub struct BlockTiming {
    /// Batched encode of the whole block.
    pub encode_ms: f64,
    /// Decode time to the block's first token (excludes encode).
    pub first_token_ms: f64,
    /// Total batched decode-loop time for the block.
    pub decode_ms: f64,
    /// Sentences in the block.
    pub sentences: usize,
    /// Total tokens generated across the block (excludes EOS).
    pub tokens: usize,
}

/// Encoder output for a batch of sentences: `[batch, seq, dim]` row-major, padded
/// to `seq` = the batch's max source length. `lens[b]` is sentence `b`'s true
/// length, so callers ignore the pad rows.
pub struct BatchedContext {
    pub data: Vec<f32>,
    pub batch: usize,
    pub seq: usize,
    pub dim: usize,
    pub lens: Vec<usize>,
}

impl BatchedContext {
    /// Valid (unpadded) encoder rows for sentence `b`: `[lens[b], dim]`.
    pub fn sentence(&self, b: usize) -> &[f32] {
        let stride = self.seq * self.dim;
        &self.data[b * stride..b * stride + self.lens[b] * self.dim]
    }
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

    /// Like [`translate`], but returns wall-clock [`Timing`] for the perf harness
    /// (`--timing`). Mirrors [`greedy`] with `Instant` markers around encode and
    /// the decode loop; the small duplication keeps timing out of the hot path.
    pub fn translate_timed(&self, text: &str) -> (String, Timing) {
        use std::time::Instant;
        let d = self.config.dim_emb;
        let src_ids = self.src_vocab.encode_with_eos(text);
        let seq = src_ids.len();

        let t_enc = Instant::now();
        let context = self.encode(&src_ids);
        let encode_ms = t_enc.elapsed().as_secs_f64() * 1e3;

        let eos = self.trg_vocab.eos_id();
        let mut cells = vec![vec![0.0f32; d]; self.config.dec_depth + 1];
        let max_len = ((2.0 * seq as f32).ceil() as usize + 4).min(256);
        let candidates = self
            .shortlist
            .as_ref()
            .map(|s| s.candidates(&src_ids, self.shared_vocab));

        let t_dec = Instant::now();
        let mut first_token_ms = 0.0;
        let mut out = Vec::new();
        let mut prev = eos;
        for step in 0..max_len {
            let top = self.decode_step(prev, step, &context, seq, &mut cells);
            let next = self.project_argmax(&top, candidates.as_deref());
            if step == 0 {
                first_token_ms = t_dec.elapsed().as_secs_f64() * 1e3;
            }
            if next == eos {
                break;
            }
            out.push(next);
            prev = next;
        }
        let decode_ms = t_dec.elapsed().as_secs_f64() * 1e3;
        let timing = Timing {
            encode_ms,
            first_token_ms,
            decode_ms,
            out_tokens: out.len(),
        };
        (self.trg_vocab.decode(&out), timing)
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

    /// Batched encoder over a block of sentences (the production unit). Sentences
    /// are padded to the batch's max source length; padded key positions are
    /// masked out of self-attention, so each sentence's valid rows are computed
    /// exactly as if it were encoded alone (see [07-batched-inference.md]). The
    /// affines and FFN/layernorm are per-row and run over `batch·seq` rows
    /// unchanged; only attention needs the padding mask.
    pub fn encode_batch(&self, sentences: &[Vec<u32>]) -> BatchedContext {
        let d = self.config.dim_emb;
        let batch = sentences.len();
        let seq = sentences.iter().map(Vec::len).max().unwrap_or(0);
        let lens: Vec<usize> = sentences.iter().map(Vec::len).collect();

        // Embed into [batch, seq, dim]; pad rows stay zero (masked in attention).
        let mut x = vec![0.0f32; batch * seq * d];
        for (b, ids) in sentences.iter().enumerate() {
            let emb = self.embed(ids, 0, Side::Source);
            x[b * seq * d..b * seq * d + ids.len() * d].copy_from_slice(&emb);
        }
        for layer in 1..=self.config.enc_depth {
            x = self.encoder_layer_batched(layer, &x, batch, seq, &lens);
        }
        BatchedContext {
            data: x,
            batch,
            seq,
            dim: d,
            lens,
        }
    }

    fn encoder_layer_batched(
        &self,
        layer: usize,
        x: &[f32],
        batch: usize,
        seq: usize,
        lens: &[usize],
    ) -> Vec<f32> {
        let p = format!("encoder_l{layer}");
        let rows = batch * seq;
        let attn = self.multihead_batched(&format!("{p}_self"), x, x, batch, seq, seq, lens);
        let x = self.postnorm(&attn, x, rows, &format!("{p}_self_Wo"));
        self.ffn(&format!("{p}_ffn"), &x, rows)
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

    /// One batched decoder step. `prev[b]` is sentence `b`'s previous token; all
    /// active sentences step in lockstep at output position `pos`. Cross-attention
    /// attends to each sentence's own encoder context (masked to its source
    /// length). Updates the per-layer `[batch, dim]` SSRU cell state and returns
    /// the tops `[batch, dim]`. Decoder rows are independent (no decoder
    /// self-attention), so a row's output matches single-sentence decoding.
    fn decode_step_batch(
        &self,
        prev: &[u32],
        pos: usize,
        ctx: &BatchedContext,
        cells: &mut [Vec<f32>],
    ) -> Vec<f32> {
        let d = self.config.dim_emb;
        let batch = prev.len();
        let mut x = vec![0.0f32; batch * d];
        for (b, &id) in prev.iter().enumerate() {
            let emb = self.embed(&[id], pos, Side::Target);
            x[b * d..(b + 1) * d].copy_from_slice(&emb);
        }
        for layer in 1..=self.config.dec_depth {
            let p = format!("decoder_l{layer}");
            let cand = self.weights.affine(&format!("{p}_rnn_W"), &x, batch, None);
            let gate = self.weights.affine(
                &format!("{p}_rnn_Wf"),
                &x,
                batch,
                Some(&format!("{p}_rnn_bf")),
            );
            // Highway/ReLU are elementwise over the whole [batch, dim] cell state.
            let c = ops::highway(&cells[layer], &cand, &gate);
            cells[layer] = c.clone();
            let h = ops::relu(&c);
            let x_self = self.postnorm(&h, &x, batch, &format!("{p}_rnn_ffn"));
            // Cross-attention: one query per sentence over its own context.
            let attn = self.multihead_batched(
                &format!("{p}_context"),
                &x_self,
                &ctx.data,
                batch,
                1,
                ctx.seq,
                &ctx.lens,
            );
            let x_ctx = self.postnorm(&attn, &x_self, batch, &format!("{p}_context_Wo"));
            x = self.ffn(&format!("{p}_ffn"), &x_ctx, batch);
        }
        x
    }

    /// Greedy-decode a block of sentences together (batched encode + batched
    /// decode with per-row EOS). Returns each sentence's output token ids, in
    /// input order. Sentences that hit EOS or their length cap are kept in the
    /// batch and masked out (their tops ignored) until the whole block finishes.
    pub fn greedy_batch(&self, sentences: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let d = self.config.dim_emb;
        let batch = sentences.len();
        let ctx = self.encode_batch(sentences);
        let eos = self.trg_vocab.eos_id();

        let max_len: Vec<usize> = sentences
            .iter()
            .map(|s| ((2.0 * s.len() as f32).ceil() as usize + 4).min(256))
            .collect();
        let cap = max_len.iter().copied().max().unwrap_or(0);
        // Per-sentence shortlist candidate sets (None when no shortlist attached).
        let cands: Vec<Option<Vec<u32>>> = sentences
            .iter()
            .map(|s| {
                self.shortlist
                    .as_ref()
                    .map(|sl| sl.candidates(s, self.shared_vocab))
            })
            .collect();

        let mut cells = vec![vec![0.0f32; batch * d]; self.config.dec_depth + 1];
        let mut prev = vec![eos; batch];
        let mut out = vec![Vec::new(); batch];
        let mut done = vec![false; batch];

        for step in 0..cap {
            if done.iter().all(|&x| x) {
                break;
            }
            let tops = self.decode_step_batch(&prev, step, &ctx, &mut cells);
            for b in 0..batch {
                if done[b] || step >= max_len[b] {
                    done[b] = true;
                    continue;
                }
                let next = self.project_argmax(&tops[b * d..(b + 1) * d], cands[b].as_deref());
                if next == eos {
                    done[b] = true;
                } else {
                    out[b].push(next);
                    prev[b] = next;
                }
            }
        }
        out
    }

    /// Tokenize a block of sentences, batch-translate, and detokenize each.
    pub fn translate_batch(&self, texts: &[&str]) -> Vec<String> {
        let ids: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| self.src_vocab.encode_with_eos(t))
            .collect();
        self.greedy_batch(&ids)
            .iter()
            .map(|o| self.trg_vocab.decode(o))
            .collect()
    }

    /// Like [`translate_batch`], but returns per-block [`BlockTiming`] for the
    /// block benchmark. Mirrors [`greedy_batch`] with `Instant` markers around the
    /// batched encode and decode loop.
    pub fn translate_batch_timed(&self, texts: &[&str]) -> (Vec<String>, BlockTiming) {
        use std::time::Instant;
        let d = self.config.dim_emb;
        let sentences: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| self.src_vocab.encode_with_eos(t))
            .collect();
        let batch = sentences.len();

        let t_enc = Instant::now();
        let ctx = self.encode_batch(&sentences);
        let encode_ms = t_enc.elapsed().as_secs_f64() * 1e3;

        let eos = self.trg_vocab.eos_id();
        let max_len: Vec<usize> = sentences
            .iter()
            .map(|s| ((2.0 * s.len() as f32).ceil() as usize + 4).min(256))
            .collect();
        let cap = max_len.iter().copied().max().unwrap_or(0);
        let cands: Vec<Option<Vec<u32>>> = sentences
            .iter()
            .map(|s| {
                self.shortlist
                    .as_ref()
                    .map(|sl| sl.candidates(s, self.shared_vocab))
            })
            .collect();

        let mut cells = vec![vec![0.0f32; batch * d]; self.config.dec_depth + 1];
        let mut prev = vec![eos; batch];
        let mut out = vec![Vec::new(); batch];
        let mut done = vec![false; batch];

        let t_dec = Instant::now();
        let mut first_token_ms = 0.0;
        for step in 0..cap {
            if done.iter().all(|&x| x) {
                break;
            }
            let tops = self.decode_step_batch(&prev, step, &ctx, &mut cells);
            if step == 0 {
                first_token_ms = t_dec.elapsed().as_secs_f64() * 1e3;
            }
            for b in 0..batch {
                if done[b] || step >= max_len[b] {
                    done[b] = true;
                    continue;
                }
                let next = self.project_argmax(&tops[b * d..(b + 1) * d], cands[b].as_deref());
                if next == eos {
                    done[b] = true;
                } else {
                    out[b].push(next);
                    prev[b] = next;
                }
            }
        }
        let decode_ms = t_dec.elapsed().as_secs_f64() * 1e3;
        let timing = BlockTiming {
            encode_ms,
            first_token_ms,
            decode_ms,
            sentences: batch,
            tokens: out.iter().map(Vec::len).sum(),
        };
        (
            out.iter().map(|o| self.trg_vocab.decode(o)).collect(),
            timing,
        )
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

    /// Batched multi-head attention over `[batch, q_len, dim]` / `[batch, kv_len,
    /// dim]`. `kv_lens[b]` is sentence `b`'s valid key count; keys at positions
    /// `>= kv_lens[b]` are masked (scored −∞ → zero weight), so a query never
    /// attends to padding. Q/K/V/O affines are per-row and run over the whole
    /// `batch·*_len`; only the scaled-dot-product is per-(batch, head).
    #[allow(clippy::too_many_arguments)]
    fn multihead_batched(
        &self,
        prefix: &str,
        q_in: &[f32],
        kv_in: &[f32],
        batch: usize,
        q_len: usize,
        kv_len: usize,
        kv_lens: &[usize],
    ) -> Vec<f32> {
        let d = self.config.dim_emb;
        let h = self.config.heads;
        let dk = d / h;
        let scale = 1.0 / (dk as f32).sqrt();
        let rows_q = batch * q_len;
        let rows_kv = batch * kv_len;

        let q = self.weights.affine(
            &format!("{prefix}_Wq"),
            q_in,
            rows_q,
            Some(&format!("{prefix}_bq")),
        );
        let k = self.weights.affine(
            &format!("{prefix}_Wk"),
            kv_in,
            rows_kv,
            Some(&format!("{prefix}_bk")),
        );
        let v = self.weights.affine(
            &format!("{prefix}_Wv"),
            kv_in,
            rows_kv,
            Some(&format!("{prefix}_bv")),
        );

        let mut joined = vec![0.0f32; rows_q * d];
        let mut scores = vec![0.0f32; kv_len];
        for b in 0..batch {
            let klen = kv_lens[b];
            for head in 0..h {
                let off = head * dk;
                for i in 0..q_len {
                    let qh = &q[(b * q_len + i) * d + off..(b * q_len + i) * d + off + dk];
                    for j in 0..kv_len {
                        if j < klen {
                            let kh =
                                &k[(b * kv_len + j) * d + off..(b * kv_len + j) * d + off + dk];
                            scores[j] =
                                qh.iter().zip(kh).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                        } else {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }
                    let weights = ops::softmax(&scores, 1, kv_len);
                    let out =
                        &mut joined[(b * q_len + i) * d + off..(b * q_len + i) * d + off + dk];
                    for (j, &w) in weights.iter().enumerate() {
                        if w == 0.0 {
                            continue;
                        }
                        let vh = &v[(b * kv_len + j) * d + off..(b * kv_len + j) * d + off + dk];
                        for c in 0..dk {
                            out[c] += w * vh[c];
                        }
                    }
                }
            }
        }
        self.weights.affine(
            &format!("{prefix}_Wo"),
            &joined,
            rows_q,
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
