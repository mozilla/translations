from dataclasses import dataclass
from pathlib import Path

import pytest

from remote_settings import gcs

from .common import ATTACHMENTS_SOURCE_DIR
from .gcs_fixtures import FakeGcs, build_gcs_fixture


@dataclass
class GcsArgs:
    """Minimal arguments for invoking the GCS downloader."""

    test: bool
    lang_pair: str
    architecture: str
    use_cached: bool = False


@pytest.fixture(autouse=True)
def gcs_output(monkeypatch):
    """Capture GCS output to avoid printing during tests."""
    messages = {"info": [], "error": [], "help": []}

    def capture_info(*args, **kwargs):
        """Collect info output from the GCS helper."""
        messages["info"].append(" ".join(str(arg) for arg in args))

    def capture_error(*args, **kwargs):
        """Collect error output from the GCS helper."""
        messages["error"].append(" ".join(str(arg) for arg in args))

    def capture_help(*args, **kwargs):
        """Collect help output from the GCS helper."""
        messages["help"].append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr(gcs, "print_info", capture_info)
    monkeypatch.setattr(gcs, "print_error", capture_error)
    monkeypatch.setattr(gcs, "print_help", capture_help)

    return messages


def test_gcs_download_populates_models(tmp_path, monkeypatch):
    """Ensure GCS downloads populate and decompress model artifacts."""
    fixture = build_gcs_fixture()
    fake_gcs = FakeGcs(fixture)

    monkeypatch.setenv("REMOTE_SETTINGS_TEST_GCS", "1")
    monkeypatch.setattr(gcs, "_list_objects", fake_gcs.list_objects)
    monkeypatch.setattr(gcs, "_download_object", fake_gcs.download_object)
    monkeypatch.setattr(gcs, "_fetch_metadata", fake_gcs.fetch_metadata)

    spec = next(spec for spec in fixture.specs if spec.architecture == "tiny")
    args = GcsArgs(test=True, lang_pair=spec.lang_pair, architecture=spec.architecture)
    gcs.ensure_model_files(args, tmp_path.as_posix())

    target_dir = Path(tmp_path) / spec.architecture / spec.lang_pair
    assert (target_dir / "metadata.json").exists()
    assert (target_dir / "model.esen.intgemm8.bin").exists()
    assert not (target_dir / "model.esen.intgemm8.bin.gz").exists()
    assert (target_dir / "lex.50.50.esen.s2t.bin").exists()
    assert not (target_dir / "lex.50.50.esen.s2t.bin.zst").exists()


def test_gcs_download_all_architectures(tmp_path, monkeypatch):
    """Download all architectures and confirm gzip and zstd artifacts are handled."""
    fixture = build_gcs_fixture()
    fake_gcs = FakeGcs(fixture)

    monkeypatch.setenv("REMOTE_SETTINGS_TEST_GCS", "1")
    monkeypatch.setattr(gcs, "_list_objects", fake_gcs.list_objects)
    monkeypatch.setattr(gcs, "_download_object", fake_gcs.download_object)
    monkeypatch.setattr(gcs, "_fetch_metadata", fake_gcs.fetch_metadata)

    for spec in fixture.specs:
        args = GcsArgs(test=True, lang_pair=spec.lang_pair, architecture=spec.architecture)
        gcs.ensure_model_files(args, tmp_path.as_posix())

        target_dir = Path(tmp_path) / spec.architecture / spec.lang_pair
        assert (target_dir / "metadata.json").exists()
        assert list(target_dir.glob("*.bin")) or list(target_dir.glob("*.spm"))
        assert not list(target_dir.glob("*.zst"))
        assert not list(target_dir.glob("*.gz"))

        for filename in spec.gzip_files:
            assert (target_dir / filename).exists()
            assert not (target_dir / f"{filename}.gz").exists()


def test_gcs_download_missing_architecture(tmp_path, monkeypatch):
    """Fail when no matching architecture exists in metadata."""
    fixture = build_gcs_fixture()
    fake_gcs = FakeGcs(fixture)

    monkeypatch.setenv("REMOTE_SETTINGS_TEST_GCS", "1")
    monkeypatch.setattr(gcs, "_list_objects", fake_gcs.list_objects)
    monkeypatch.setattr(gcs, "_download_object", fake_gcs.download_object)
    monkeypatch.setattr(gcs, "_fetch_metadata", fake_gcs.fetch_metadata)

    args = GcsArgs(test=True, lang_pair="esen", architecture="unknown-arch")
    with pytest.raises(SystemExit):
        gcs.ensure_model_files(args, tmp_path.as_posix())


def test_gcs_download_missing_lang_pair(tmp_path, monkeypatch):
    """Fail when no models are available for a language pair."""

    def empty_list_objects(bucket, prefix):
        """Return an empty object listing for any prefix."""
        return []

    monkeypatch.setenv("REMOTE_SETTINGS_TEST_GCS", "1")
    monkeypatch.setattr(gcs, "_list_objects", empty_list_objects)

    args = GcsArgs(test=True, lang_pair="zzzz", architecture="base")
    with pytest.raises(SystemExit):
        gcs.ensure_model_files(args, tmp_path.as_posix())


def test_gcs_download_redownloads_by_default(tmp_path, monkeypatch):
    """Redownload when cached models exist without --use-cached."""
    fixture = build_gcs_fixture()
    fake_gcs = FakeGcs(fixture)

    monkeypatch.setenv("REMOTE_SETTINGS_TEST_GCS", "1")
    monkeypatch.setattr(gcs, "_list_objects", fake_gcs.list_objects)
    monkeypatch.setattr(gcs, "_download_object", fake_gcs.download_object)
    monkeypatch.setattr(gcs, "_fetch_metadata", fake_gcs.fetch_metadata)

    spec = next(spec for spec in fixture.specs if spec.lang_pair == "esen")
    args = GcsArgs(test=True, lang_pair=spec.lang_pair, architecture=spec.architecture)
    gcs.ensure_model_files(args, tmp_path.as_posix())

    target_dir = Path(tmp_path) / spec.architecture / spec.lang_pair
    model_path = target_dir / "model.esen.intgemm8.bin"
    model_path.write_bytes(b"cached")

    gcs.ensure_model_files(args, tmp_path.as_posix())

    expected = (ATTACHMENTS_SOURCE_DIR / "tiny" / "esen" / model_path.name).read_bytes()
    assert model_path.read_bytes() == expected


def test_gcs_download_clears_target_dir(tmp_path, monkeypatch):
    """Remove any existing target directory contents before redownloading."""
    fixture = build_gcs_fixture()
    fake_gcs = FakeGcs(fixture)

    monkeypatch.setenv("REMOTE_SETTINGS_TEST_GCS", "1")
    monkeypatch.setattr(gcs, "_list_objects", fake_gcs.list_objects)
    monkeypatch.setattr(gcs, "_download_object", fake_gcs.download_object)
    monkeypatch.setattr(gcs, "_fetch_metadata", fake_gcs.fetch_metadata)

    spec = next(spec for spec in fixture.specs if spec.lang_pair == "esen")
    args = GcsArgs(test=True, lang_pair=spec.lang_pair, architecture=spec.architecture)
    gcs.ensure_model_files(args, tmp_path.as_posix())

    target_dir = Path(tmp_path) / spec.architecture / spec.lang_pair
    stale_path = target_dir / "stale.txt"
    stale_path.write_text("stale", encoding="utf-8")

    gcs.ensure_model_files(args, tmp_path.as_posix())

    assert not stale_path.exists()
    assert (target_dir / "model.esen.intgemm8.bin").exists()


def test_gcs_download_uses_cache_when_requested(tmp_path, monkeypatch):
    """Skip redownloads when --use-cached is provided."""
    fixture = build_gcs_fixture()
    fake_gcs = FakeGcs(fixture)

    monkeypatch.setenv("REMOTE_SETTINGS_TEST_GCS", "1")
    monkeypatch.setattr(gcs, "_list_objects", fake_gcs.list_objects)
    monkeypatch.setattr(gcs, "_download_object", fake_gcs.download_object)
    monkeypatch.setattr(gcs, "_fetch_metadata", fake_gcs.fetch_metadata)

    spec = next(spec for spec in fixture.specs if spec.lang_pair == "esen")
    args = GcsArgs(test=True, lang_pair=spec.lang_pair, architecture=spec.architecture)
    gcs.ensure_model_files(args, tmp_path.as_posix())

    target_dir = Path(tmp_path) / spec.architecture / spec.lang_pair
    model_path = target_dir / "model.esen.intgemm8.bin"
    model_path.write_bytes(b"cached")

    cached_args = GcsArgs(
        test=True,
        lang_pair=spec.lang_pair,
        architecture=spec.architecture,
        use_cached=True,
    )
    gcs.ensure_model_files(cached_args, tmp_path.as_posix())

    assert model_path.read_bytes() == b"cached"


def test_gcs_download_use_cached_missing_fails(tmp_path, monkeypatch):
    """Fail when --use-cached is set but no models are available locally."""

    def fail_list_objects(bucket, prefix):
        """Ensure GCS is not queried when cache is required."""
        raise AssertionError("GCS listing should not be invoked")

    monkeypatch.setenv("REMOTE_SETTINGS_TEST_GCS", "1")
    monkeypatch.setattr(gcs, "_list_objects", fail_list_objects)

    args = GcsArgs(test=True, lang_pair="esen", architecture="tiny", use_cached=True)
    with pytest.raises(SystemExit):
        gcs.ensure_model_files(args, tmp_path.as_posix())
