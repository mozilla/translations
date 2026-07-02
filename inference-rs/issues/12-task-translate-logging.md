# Logging in the translate task.


```
➤ task inference-rs:translate -- en es --text "Hello world! This is a translation of some text."
task: [inference-rs:translate] python3 /Users/greg/dev/translations/inference-rs/scripts/translate.py en es --text 'Hello world! This is a translation of some text.'
[run] cargo run --quiet --manifest-path /Users/greg/dev/translations/inference-rs/Cargo.toml -- translate data/models/enes/model.enes.intgemm.alphas.bin data/models/enes/vocab.enes.spm data/models/enes/vocab.enes.spm
Hola mundo! Esta es una traducción de algún texto.
```

This task takes a long time to run with long time to first token. I want better logging here to understand and report on timing. I'm assuming there are some long oeprations we can report on with timing.

Do some design work and specify here in this issue what the output design should look like before diving into the implementation.

## Where the time actually goes (measure before designing)

Prime suspects, roughly in order — the design should confirm these with real numbers, not
assume:

1. **`cargo run` build/link** — the first invocation compiles. This is *tooling* latency,
   not engine latency; the timing report must not conflate it with translation. Report the
   binary build separately, or require a pre-built binary for timing runs.
2. **Model load + `Wemb` dequant** — reading the 31 MB `.bin` and expanding `Wemb` to f32
   (see [16-wemb-dequant-memory.md](./16-wemb-dequant-memory.md)). One-time per process.
3. **Encode** (whole source through 6 encoder layers) — this is most of time-to-first-token.
4. **Per-token decode** — SSRU + cross-attn + FFN, then the **full-vocab projection**
   (32000×384 scalar mults/token) which is likely the per-token hot spot.

## Proposed output design

Timing goes to **stderr** (stdout stays clean = just translations, so piping is unaffected),
behind a `--timing` flag on `translate.py` → passed to the CLI. Phased, human-readable:

```
[timing] model load        412 ms
[timing] vocab load          38 ms
[timing] encode (8 tok)      21 ms
[timing] decode step 0        9 ms   <- time to first token: 30 ms (encode + step 0)
[timing] decode 1..12        88 ms   (12 tokens, 7.3 ms/tok)
[timing] total              559 ms   | 13 tokens, 23.2 tok/s
```

Rules:
- **TTFT** = encode + first decode step, called out explicitly (it's the metric you care about).
- Build time is either excluded (pre-built binary) or on its own clearly-labelled line — never folded into "total".
- Aggregate the steady-state decode steps into one line (count + ms/tok); don't spam one line per token.
- Keep it dependency-free: `std::time::Instant` in the engine, printed at phase boundaries.

## Decided

- **`--timing`, off by default.** Clean stdout stays the default; timing is opt-in on
  `translate.py` → CLI.
- **Human-readable lines only** here (the format above). The perf harness
  ([10-perf-harness.md](./10-perf-harness.md)) parses these `[timing]` lines or instruments
  separately — no structured/JSON format to maintain in the engine.
