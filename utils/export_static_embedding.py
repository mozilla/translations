#!/usr/bin/env python3
"""
Export a static token embedding table from Mozilla/Bergamot/Marian translation model URLs.

Inputs:
  --model-url: URL to the Marian/Bergamot model file. May be a raw .bin file or a .zst file.
  --vocab-url: URL to the SentencePiece vocab/model file. May be a raw .spm file or a .zst file.

Outputs:
  - tokenizer.jsonl: token id -> tokenizer metadata
  - embeddings.npy: dense float32 matrix [vocab_size, embedding_dim], permanently scaled by normalized/clipped IDF
  - inverse_frequency_weights.npy: per-token normalized/clipped IDF scale factors
  - token_index_to_embedding.jsonl: token id/piece -> embedding vector (optional; can be large)
  - token_frequencies.csv: token id/piece -> SentencePiece score and normalized probability proxy
  - metadata.json: export metadata
  - model2vec_model/: optional Model2Vec StaticModel directory, loadable with StaticModel.from_pretrained()

Notes:
  Marian/Bergamot packages usually do not store raw corpus word-frequency counts.
  SentencePiece stores scores/log-probabilities, so this script exports those and a normalized
  exp(score) proxy. Raw counts are emitted as blank/null unless you provide a corpus separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import sentencepiece as spm

ALIGNMENT = 16384
CHUNK_SIZE = 1024 * 1024


def die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def download_url(url: str, dst: Path) -> None:
    """Download url to dst using only the Python standard library."""
    req = Request(url, headers={"User-Agent": "static-embedding-exporter/1.0"})
    try:
        with urlopen(req) as response, dst.open("wb") as out:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:
        die(f"Failed to download {url!r}: {exc}")


def filename_from_url(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name
    return name or fallback


def decompress_zst(src: Path, dst: Path) -> None:
    zstd = shutil.which("zstd")
    if zstd:
        with dst.open("wb") as out:
            subprocess.run([zstd, "-q", "-d", "-c", str(src)], check=True, stdout=out)
        return

    try:
        import zstandard as zstd_mod  # type: ignore
    except Exception:
        die("Need either the zstd CLI or the Python zstandard package to decompress .zst files")

    dctx = zstd_mod.ZstdDecompressor()
    with src.open("rb") as inp, dst.open("wb") as out:
        dctx.copy_stream(inp, out)


def download_and_maybe_decompress(url: str, work_dir: Path, fallback_name: str, output_suffix: str) -> Path:
    """Download a URL and return a local file path. If it is .zst, return the decompressed file."""
    downloaded = work_dir / filename_from_url(url, fallback_name)
    download_url(url, downloaded)

    if downloaded.suffix == ".zst":
        decompressed_name = downloaded.with_suffix("").name or fallback_name
        if not Path(decompressed_name).suffix:
            decompressed_name += output_suffix
        decompressed = work_dir / decompressed_name
        decompress_zst(downloaded, decompressed)
        return decompressed

    return downloaded


def parse_marian_binary(model_bin: Path) -> Tuple[List[Dict[str, int]], int]:
    """Parse enough of a Marian binary model to locate tensors.

    Layout used by compact Bergamot/Marian binaries:
      uint64 version
      uint64 tensor_count
      tensor_count records of 4x uint64
      null-terminated tensor-name table
      padding to 16 KiB
      tensor payloads in record order

    Record fields are not fully documented here. We use field 4 as byte size and assign names
    from the name table in order, which is sufficient for exporting Wemb.
    """
    with model_bin.open("rb") as f:
        magic = f.read(16)
        if len(magic) != 16:
            die("Model binary is too small")
        _version, count = struct.unpack("<QQ", magic)
        if count <= 0 or count > 100000:
            die(f"Unexpected tensor count in Marian binary: {count}")

        raw_records = f.read(count * 32)
        if len(raw_records) != count * 32:
            die("Truncated tensor metadata")

        records = []
        for i in range(count):
            a, b, c, nbytes = struct.unpack("<QQQQ", raw_records[i * 32:(i + 1) * 32])
            records.append({"index": i, "field0": a, "field1": b, "field2": c, "nbytes": nbytes})

        names_blob_len = ALIGNMENT - (16 + count * 32)
        if names_blob_len <= 0:
            die("Tensor metadata is larger than the expected 16 KiB header block")
        names_blob = f.read(names_blob_len)
        names = [x.decode("utf-8", errors="replace") for x in names_blob.split(b"\0") if x]
        if len(names) < count:
            die(f"Found only {len(names)} tensor names for {count} tensors")

    offset = ALIGNMENT
    for rec, name in zip(records, names):
        rec["name"] = name  # type: ignore[assignment]
        rec["offset"] = offset
        offset += int(rec["nbytes"])
    return records, ALIGNMENT


def infer_wemb_shape(nbytes: int, vocab_size: int) -> Tuple[int, int]:
    # Quantized int8 Wemb may include a small alignment pad at the end.
    dim = nbytes // vocab_size
    while dim > 0 and nbytes - vocab_size * dim > 4096:
        dim -= 1
    if dim <= 0:
        die(f"Could not infer embedding dimension from {nbytes} bytes and vocab size {vocab_size}")
    pad = nbytes - vocab_size * dim
    if pad < 0 or pad > 4096:
        die(f"Suspicious Wemb padding: {pad} bytes")
    return vocab_size, dim


def sp_type_name(sp: spm.SentencePieceProcessor, idx: int) -> str:
    if sp.is_unknown(idx):
        return "unknown"
    if sp.is_control(idx):
        return "control"
    if sp.is_unused(idx):
        return "unused"
    if sp.is_byte(idx):
        return "byte"
    return "normal"


def write_model2vec_static_model(
    output_dir: Path,
    dirname: str,
    embeddings: np.ndarray,
    sp: spm.SentencePieceProcessor,
    model_url: str,
    vocab_url: str,
    idf_weights: np.ndarray,
    model_name: str,
    base_model_name: str,
    language: List[str],
) -> Path:
    """Write a Model2Vec StaticModel using Model2Vec's own save_pretrained API.

    This requires optional dependencies:
      pip install model2vec tokenizers
    """
    try:
        from model2vec import StaticModel  # type: ignore
        from tokenizers.implementations import SentencePieceUnigramTokenizer  # type: ignore
    except Exception as exc:
        die(
            "--write-model2vec requires optional dependencies. Install them with "
            "`pip install model2vec tokenizers` and rerun. "
            f"Original import error: {exc}"
        )

    vocab = [(sp.id_to_piece(i), float(sp.get_score(i))) for i in range(sp.get_piece_size())]
    tokenizer = SentencePieceUnigramTokenizer(
        vocab=vocab,
        replacement="▁",
        add_prefix_space=True,
    )

    config = {
        "normalize": True,
        "embedding_source": "bergamot_marian_wemb",
        "model_url": model_url,
        "vocab_url": vocab_url,
        "embedding_dtype": str(embeddings.dtype),
        "idf_scaling": {
            "enabled": True,
            "probability_source": "normalized exp(SentencePiece score)",
            "formula": "clip(-log(max(probability, epsilon)), clip_min, clip_max) / mean(clipped_idf)",
            "epsilon": 1e-12,
            "clip_min": 1.0,
            "clip_max": 15.0,
            "normalized_mean": 1.0,
            "weight_min": float(idf_weights.min()),
            "weight_max": float(idf_weights.max()),
            "weight_mean": float(idf_weights.mean()),
        },
    }

    static_model = StaticModel(
        vectors=embeddings.astype(np.float32, copy=False),
        tokenizer=tokenizer,
        config=config,
        normalize=True,
        base_model_name=base_model_name,
        language=language,
    )

    model2vec_dir = output_dir / dirname
    static_model.save_pretrained(model2vec_dir, model_name=model_name)
    return model2vec_dir


def export(
    model_url: str,
    vocab_url: str,
    output_dir: Path,
    write_jsonl_embeddings: bool,
    write_model2vec: bool,
    model2vec_dirname: str,
    model2vec_model_name: str,
    model2vec_base_model_name: str,
    model2vec_language: List[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="static_emb_export_") as tmp_s:
        tmp = Path(tmp_s)
        model_bin = download_and_maybe_decompress(
            model_url,
            tmp,
            fallback_name="model.bin",
            output_suffix=".bin",
        )
        vocab_file = download_and_maybe_decompress(
            vocab_url,
            tmp,
            fallback_name="vocab.spm",
            output_suffix=".spm",
        )

        try:
            sp = spm.SentencePieceProcessor(model_file=str(vocab_file))
        except Exception as exc:
            die(f"Failed to load SentencePiece vocab from {vocab_file}: {exc}")

        vocab_size = sp.get_piece_size()
        records, _ = parse_marian_binary(model_bin)
        wemb = next((r for r in records if r.get("name") == "Wemb"), None)
        if wemb is None:
            tensor_names = ", ".join(str(r.get("name")) for r in records[:20])
            die(f"No tensor named Wemb found. First tensors: {tensor_names}")

        rows, dim = infer_wemb_shape(int(wemb["nbytes"]), vocab_size)
        with model_bin.open("rb") as f:
            f.seek(int(wemb["offset"]))
            raw = f.read(rows * dim)
        if len(raw) != rows * dim:
            die("Could not read the full Wemb tensor payload")

        embeddings = (
            np.frombuffer(raw, dtype=np.int8)
            .reshape(rows, dim)
            .astype(np.float32)
        )

        scores = np.array([sp.get_score(i) for i in range(vocab_size)], dtype=np.float64)
        finite = np.isfinite(scores)
        probs = np.zeros_like(scores)
        if finite.any():
            max_score = scores[finite].max()
            probs[finite] = np.exp(scores[finite] - max_score)
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs /= prob_sum

        safe_probs = np.maximum(probs, 1e-12)
        idf_weights = -np.log(safe_probs)
        idf_weights = np.clip(idf_weights, 1.0, 15.0)
        idf_weights = idf_weights / idf_weights.mean()
        idf_weights = idf_weights.astype(np.float32)

        embeddings = embeddings * idf_weights[:, None]
        np.save(output_dir / "embeddings.npy", embeddings)
        np.save(output_dir / "inverse_frequency_weights.npy", idf_weights)

        metadata = {
            "model_url": model_url,
            "vocab_url": vocab_url,
            "vocab_size": vocab_size,
            "embedding_dim": dim,
            "embedding_dtype": "float32_scaled_from_int8_quantized_marian_Wemb",
            "original_embedding_dtype": "int8_quantized_marian_Wemb",
            "wemb_tensor_bytes": int(wemb["nbytes"]),
            "local_download_note": "Inputs were downloaded to a temporary directory and removed after export.",
            "idf_scaling": {
                "enabled": True,
                "probability_source": "normalized exp(SentencePiece score)",
                "formula": "clip(-log(max(probability, epsilon)), clip_min, clip_max) / mean(clipped_idf)",
                "epsilon": 1e-12,
                "clip_min": 1.0,
                "clip_max": 15.0,
                "normalized_mean": 1.0,
                "weights_file": "inverse_frequency_weights.npy",
            },
            "note": "Raw corpus frequency counts are not stored in this package; embeddings.npy is permanently scaled using normalized/clipped IDF derived from SentencePiece scores.",
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        with (output_dir / "tokenizer.jsonl").open("w", encoding="utf-8") as out:
            for i in range(vocab_size):
                rec = {
                    "id": i,
                    "piece": sp.id_to_piece(i),
                    "score": float(sp.get_score(i)),
                    "type": sp_type_name(sp, i),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with (output_dir / "token_frequencies.csv").open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(
                out,
                fieldnames=[
                    "id",
                    "piece",
                    "raw_count",
                    "sentencepiece_score",
                    "normalized_probability_proxy",
                    "idf_weight",
                ],
            )
            writer.writeheader()
            for i in range(vocab_size):
                writer.writerow({
                    "id": i,
                    "piece": sp.id_to_piece(i),
                    "raw_count": "",
                    "sentencepiece_score": float(sp.get_score(i)),
                    "normalized_probability_proxy": float(probs[i]),
                    "idf_weight": float(idf_weights[i]),
                })

        if write_jsonl_embeddings:
            with (output_dir / "token_index_to_embedding.jsonl").open("w", encoding="utf-8") as out:
                for i in range(vocab_size):
                    out.write(json.dumps({
                        "id": i,
                        "piece": sp.id_to_piece(i),
                        "embedding": embeddings[i].astype(float).tolist(),
                    }, ensure_ascii=False) + "\n")

        if write_model2vec:
            model2vec_dir = write_model2vec_static_model(
                output_dir=output_dir,
                dirname=model2vec_dirname,
                embeddings=embeddings,
                sp=sp,
                model_url=model_url,
                vocab_url=vocab_url,
                idf_weights=idf_weights,
                model_name=model2vec_model_name,
                base_model_name=model2vec_base_model_name,
                language=model2vec_language,
            )
            metadata["model2vec"] = {
                "enabled": True,
                "directory": str(model2vec_dir.relative_to(output_dir)),
                "load_example": f"StaticModel.from_pretrained({str(model2vec_dir)!r})",
                "note": "Created using model2vec.StaticModel.save_pretrained().",
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(json.dumps(metadata, indent=2))
        print(f"Wrote files to: {output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export static token embeddings from model and SentencePiece vocab URLs."
    )
    ap.add_argument("--model-url", required=True, help="URL to model .bin or .zst file")
    ap.add_argument("--vocab-url", required=True, help="URL to SentencePiece .spm or .zst file")
    ap.add_argument("--output-dir", type=Path, default=Path("static_embedding_export"))
    ap.add_argument(
        "--write-jsonl-embeddings",
        action="store_true",
        help="Also write token_index_to_embedding.jsonl; embeddings.npy is always written",
    )
    ap.add_argument(
        "--write-model2vec",
        action="store_true",
        help="Also write a Model2Vec StaticModel directory using StaticModel.save_pretrained(). Requires: pip install model2vec tokenizers",
    )
    ap.add_argument(
        "--model2vec-dirname",
        default="model2vec_model",
        help="Subdirectory name under --output-dir for the Model2Vec model",
    )
    ap.add_argument(
        "--model2vec-model-name",
        default="bergamot-marian-static-embedding",
        help="Model name used in the generated Model2Vec model card/metadata",
    )
    ap.add_argument(
        "--model2vec-base-model-name",
        default="bergamot-marian",
        help="Base model name stored in Model2Vec metadata",
    )
    ap.add_argument(
        "--model2vec-language",
        default="fr,en",
        help="Comma-separated language codes stored in Model2Vec metadata, e.g. 'fr,en'",
    )
    args = ap.parse_args()
    language = [x.strip() for x in args.model2vec_language.split(",") if x.strip()]
    export(
        args.model_url,
        args.vocab_url,
        args.output_dir,
        args.write_jsonl_embeddings,
        args.write_model2vec,
        args.model2vec_dirname,
        args.model2vec_model_name,
        args.model2vec_base_model_name,
        language,
    )


if __name__ == "__main__":
    main()
