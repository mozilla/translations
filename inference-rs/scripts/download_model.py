#!/usr/bin/env python3
"""
Download a Firefox Translations production model from Remote Settings by language pair.

For the given source/target languages this:
  1. Fetches the model records from the production Remote Settings collection.
  2. Picks the latest "model", "vocab", and "lex" records for the pair.
  3. Downloads each zstd-compressed attachment from the CDN and decompresses it.
  4. Verifies the decompressed file's sha256 against the record's decompressedHash.
  5. Writes a bergamot decode config (config.{src}{trg}.yml) for translator-cli.

Run via poetry so the pipeline helpers are importable:
    PYTHONPATH=$(pwd) poetry run python -W ignore inference-rs/scripts/download_model.py en es

Or through the task wrapper:
    task inference-rs:download-model -- en es
"""

import argparse
import hashlib
import tempfile
from pathlib import Path

import requests
from zstandard import ZstdDecompressor

from pipeline.common.downloads import stream_download_to_file
from utils.common.remote_settings import get_prod_records_url

CDN_ROOT = "https://firefox-settings-attachments.cdn.mozilla.net"
# NOTE: production is the "-v2" collection. utils/common/remote_settings.py still points
# `models_collection` at the older "translations-models"; this uses v2 on purpose.
DEFAULT_COLLECTION = "translations-models-v2"


def fetch_records(collection: str) -> list[dict]:
    url = get_prod_records_url(collection)
    print(f"[rs] Fetching records: {url}")
    response = requests.get(url)
    response.raise_for_status()
    return response.json()["data"]


def version_key(record: dict) -> tuple:
    try:
        return tuple(int(p) for p in str(record.get("version", "0")).split("."))
    except ValueError:
        return (0,)


def pick_record(records: list[dict], file_type: str, src: str, trg: str):
    matches = [
        r
        for r in records
        if r.get("fileType") == file_type
        and r.get("sourceLanguage") == src
        and r.get("targetLanguage") == trg
    ]
    if not matches:
        return None
    matches.sort(key=version_key, reverse=True)
    if len(matches) > 1:
        versions = ", ".join(str(r.get("version")) for r in matches)
        print(
            f"[rs] {len(matches)} '{file_type}' records for {src}-{trg} ({versions}); using latest"
        )
    return matches[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_decompress(record: dict, dest_dir: Path) -> Path:
    attachment = record["attachment"]
    out_path = dest_dir / record["name"]
    url = f"{CDN_ROOT}/{attachment['location']}"

    print(f"[dl] {record['name']} v{record.get('version')} <- {url}")
    # stream_download_to_file refuses to overwrite, so download to a fresh temp path.
    with tempfile.TemporaryDirectory() as tmp:
        zst_path = Path(tmp) / attachment["filename"]
        stream_download_to_file(url, zst_path)
        with open(zst_path, "rb") as ifh, open(out_path, "wb") as ofh:
            ZstdDecompressor().copy_stream(ifh, ofh)

    expected = record.get("decompressedHash")
    if expected:
        got = sha256(out_path)
        if got != expected:
            raise SystemExit(
                f"[verify] HASH MISMATCH for {out_path.name}\n"
                f"  expected {expected}\n  got      {got}"
            )
        print(f"[verify] OK {out_path.name}")
    else:
        print(f"[verify] no decompressedHash on record for {out_path.name}; skipping check")
    return out_path


def write_config(
    dest_dir: Path,
    src: str,
    trg: str,
    model: str,
    src_vocab: str,
    trg_vocab: str,
    lex: str | None,
) -> Path:
    langs = f"{src}{trg}"
    config_path = dest_dir / f"config.{langs}.yml"
    lines = [
        "relative-paths: true",
        f"models: [{model}]",
        f"vocabs: [{src_vocab}, {trg_vocab}]",
    ]
    # Split-vocab pairs (CJK) ship no shortlist, and the reference shortlist is
    # incorrect for them anyway; only emit one when a lex was found.
    if lex is not None:
        lines.append(f"shortlist: [{lex}, false]")
    lines += [
        "beam-size: 1",
        "normalize: 1.0",
        "word-penalty: 0",
        "max-length-break: 128",
        "mini-batch-words: 1024",
        "max-length-factor: 2.0",
        "skip-cost: true",
        "gemm-precision: int8shiftAlphaAll",
        "alignment: soft",
        "quiet: true",
        "quiet-translation: true",
    ]
    config_path.write_text("\n".join(lines) + "\n")
    print(f"[config] wrote {config_path}")
    return config_path


def require(records: list[dict], file_type: str, src: str, trg: str, dest_dir: Path) -> str:
    record = pick_record(records, file_type, src, trg)
    if not record:
        raise SystemExit(f"[rs] No '{file_type}' record found for {src}-{trg}")
    return download_and_decompress(record, dest_dir).name


def optional(records: list[dict], file_type: str, src: str, trg: str, dest_dir: Path):
    record = pick_record(records, file_type, src, trg)
    return download_and_decompress(record, dest_dir).name if record else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=str, help="Source language code, e.g. en")
    parser.add_argument("target", type=str, help="Target language code, e.g. es")
    parser.add_argument(
        "--models-dir",
        default="data/models",
        help="Root directory to download into (default: data/models)",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Remote Settings collection (default: {DEFAULT_COLLECTION})",
    )
    args = parser.parse_args()

    src, trg = args.source.lower(), args.target.lower()
    dest_dir = Path(args.models_dir) / f"{src}{trg}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    records = fetch_records(args.collection)

    model = require(records, "model", src, trg, dest_dir)
    # Shared vocab ships one `vocab`; split vocab (CJK) ships `srcvocab`/`trgvocab`.
    shared = optional(records, "vocab", src, trg, dest_dir)
    if shared is not None:
        src_vocab = trg_vocab = shared
    else:
        print("[rs] no shared 'vocab'; fetching split srcvocab/trgvocab")
        src_vocab = require(records, "srcvocab", src, trg, dest_dir)
        trg_vocab = require(records, "trgvocab", src, trg, dest_dir)
    lex = optional(records, "lex", src, trg, dest_dir)

    config_path = write_config(dest_dir, src, trg, model, src_vocab, trg_vocab, lex)
    print(
        "\nDone. To translate with the reference C++ engine:\n"
        f'  echo "Hello" | inference/build/src/app/translator-cli '
        f"--model-config-paths {config_path} --cpu-threads 4"
    )


if __name__ == "__main__":
    main()
