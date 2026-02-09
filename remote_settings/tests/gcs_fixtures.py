import gzip, json, shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .common import ATTACHMENTS_SOURCE_DIR


@dataclass
class ModelSpec:
    """Describe a model fixture for fake GCS listings."""

    architecture: str
    lang_pair: str
    model_name: str
    gzip_files: set[str]


@dataclass
class GcsFixture:
    """Container for fake GCS listings and downloads."""

    objects: list[dict]
    downloads: dict[str, dict]
    metadata: dict[tuple[str, str, str], dict]
    specs: list[ModelSpec]


class FakeGcs:
    """Fake GCS backend for offline tests."""

    def __init__(self, fixture: GcsFixture):
        """Initialize the fake backend with fixture data."""
        self._objects = fixture.objects
        self._downloads = fixture.downloads
        self._metadata = fixture.metadata

    def list_objects(self, bucket, prefix):
        """Return objects matching the requested prefix."""
        return [item for item in self._objects if item["name"].startswith(prefix)]

    def download_object(self, bucket, object_name, dest_path):
        """Write the fixture object to the destination path."""
        entry = self._downloads.get(object_name)
        if entry is None:
            raise KeyError(f"Missing fixture for {object_name}")

        source = entry["source"]
        compression = entry["compression"]
        dest_path = Path(dest_path)

        if compression == "gzip":
            with open(source, "rb") as source_file, gzip.open(dest_path, "wb") as dest_file:
                shutil.copyfileobj(source_file, dest_file)
            return

        shutil.copyfile(source, dest_path)

    def fetch_metadata(self, bucket, src, trg, model_name):
        """Return metadata for a model from fixture data."""
        return self._metadata.get((src, trg, model_name))


def build_gcs_fixture():
    """Build a fake GCS fixture from test attachments."""
    specs = [
        ModelSpec(
            architecture="base",
            lang_pair="encs",
            model_name="test-base",
            gzip_files={"vocab.encs.spm"},
        ),
        ModelSpec(
            architecture="base-memory",
            lang_pair="enes",
            model_name="test-base-memory",
            gzip_files={"lex.enes.s2t.bin"},
        ),
        ModelSpec(
            architecture="tiny",
            lang_pair="esen",
            model_name="test-tiny",
            gzip_files={"model.esen.intgemm8.bin"},
        ),
        ModelSpec(
            architecture="tiny",
            lang_pair="inha",
            model_name="test-tiny-inha",
            gzip_files=set(),
        ),
    ]

    updated = datetime(2024, 1, 1).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    objects = []
    downloads = {}
    metadata = {}

    for spec in specs:
        src = spec.lang_pair[:2]
        trg = spec.lang_pair[2:]
        model_dir = ATTACHMENTS_SOURCE_DIR / spec.architecture / spec.lang_pair
        prefix = f"models/{src}-{trg}/{spec.model_name}/exported/"

        metadata_path = model_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata[(src, trg, spec.model_name)] = json.load(metadata_file)

        object_name = f"{prefix}metadata.json"
        objects.append({"name": object_name, "updated": updated})
        downloads[object_name] = {"source": metadata_path, "compression": None}

        for path in model_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix not in {".bin", ".spm"}:
                continue

            if path.name in spec.gzip_files:
                object_name = f"{prefix}{path.name}.gz"
                objects.append({"name": object_name, "updated": updated})
                downloads[object_name] = {"source": path, "compression": "gzip"}
                continue

            zst_path = path.with_suffix(path.suffix + ".zst")
            if zst_path.exists():
                object_name = f"{prefix}{zst_path.name}"
                objects.append({"name": object_name, "updated": updated})
                downloads[object_name] = {"source": zst_path, "compression": None}
                continue

            object_name = f"{prefix}{path.name}"
            objects.append({"name": object_name, "updated": updated})
            downloads[object_name] = {"source": path, "compression": None}

    return GcsFixture(
        objects=objects,
        downloads=downloads,
        metadata=metadata,
        specs=specs,
    )
