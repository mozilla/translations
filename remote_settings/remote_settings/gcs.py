import os, json, gzip, shutil, sys, requests
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import zstandard as zstd

from remote_settings.format import print_error, print_help, print_info

GCS_BUCKET_ENV = "REMOTE_SETTINGS_GCS_BUCKET"
GCS_PREFIX_ENV = "REMOTE_SETTINGS_GCS_PREFIX"
TEST_GCS_ENV = "REMOTE_SETTINGS_TEST_GCS"
DEFAULT_BUCKET = "moz-fx-translations-data--303e-prod-translations-data"
DEFAULT_PREFIX = "models"
EXPORTED_DIR = "exported"


class GcsModel:
    """Represents a model entry discovered in GCS."""

    def __init__(self, src, trg, name, updated):
        """Initialize a GCS model record."""
        self.src = src
        self.trg = trg
        self.name = name
        self.updated = updated

    def __hash__(self):
        """Return a stable hash for set/dedup usage."""
        return hash(self.src + self.trg + self.name)

    def __eq__(self, other):
        """Compare two model records for equality."""
        return self.src == other.src and self.trg == other.trg and self.name == other.name

    @staticmethod
    def from_item(item):
        """Create a model record from a GCS list item."""
        gcs_path = item["name"]
        parts = gcs_path.split("/")
        src, trg = parts[1].split("-")
        updated = datetime.strptime(item["updated"], "%Y-%m-%dT%H:%M:%S.%fZ")
        return GcsModel(src, trg, parts[2], updated)


def gcs_bucket():
    """Resolve the GCS bucket used for model downloads."""
    return os.environ.get(GCS_BUCKET_ENV, DEFAULT_BUCKET)


def gcs_prefix():
    """Resolve the GCS prefix used for model downloads."""
    return os.environ.get(GCS_PREFIX_ENV, DEFAULT_PREFIX)


def allow_test_downloads():
    """Return True when test-mode GCS downloads are enabled."""
    return os.environ.get(TEST_GCS_ENV, "").lower() in {"1", "true", "yes"}


def ensure_model_files(args, base_dir):
    """Fetch model files from GCS when they are missing locally.

    Args:
        args (argparse.Namespace): The arguments passed through the CLI
        base_dir (str): The base directory for record attachments
    """
    if (args.test and not allow_test_downloads()) or not args.lang_pair:
        return

    target_dir = Path(base_dir) / args.architecture / args.lang_pair
    use_cached = getattr(args, "use_cached", False)
    has_cache = _has_model_files(target_dir)
    if use_cached:
        if has_cache:
            return
        print_error(f"No cached models found in {target_dir}")
        print_help("Re-run without --use-cached to download the models.")
        sys.exit(1)
    if target_dir.exists() and not use_cached:
        shutil.rmtree(target_dir)

    bucket = gcs_bucket()
    src = args.lang_pair[:2]
    trg = args.lang_pair[2:]

    model_name = _select_model(bucket, src, trg, args.architecture)
    print_info(f"Downloading {src}-{trg} ({args.architecture}) from {bucket}: {model_name}")
    _download_model_files(bucket, src, trg, model_name, target_dir)

    if not _has_model_files(target_dir):
        print_error(f"No model files downloaded for {args.lang_pair} in {target_dir}")
        sys.exit(1)


def _has_model_files(path):
    """Check whether a local directory includes required model assets."""
    if not path.exists():
        return False

    has_attachments = False
    has_metadata = False

    for entry in path.iterdir():
        if not entry.is_file():
            continue
        if entry.name.endswith(".bin") or entry.name.endswith(".spm"):
            has_attachments = True
        if entry.name == "metadata.json":
            has_metadata = True

    return has_attachments and has_metadata


def _select_model(bucket, src, trg, architecture):
    """Select the latest exported model matching the requested architecture."""
    models = _list_models(bucket, src, trg)
    if not models:
        print_error(f"No models found in GCS for {src}-{trg}")
        print_help(f"Bucket: {bucket}")
        sys.exit(1)

    available_architectures = set()
    for model in sorted(models, key=lambda m: m.updated, reverse=True):
        metadata = _fetch_metadata(bucket, src, trg, model.name)
        if not metadata:
            continue
        metadata_arch = metadata.get("architecture")
        if metadata_arch:
            available_architectures.add(metadata_arch)
        if metadata_arch == architecture:
            return model.name

    if available_architectures:
        print_error(f"No models found for architecture '{architecture}' in {bucket}")
        print_help(f"Available architectures: {', '.join(sorted(available_architectures))}")
    else:
        print_error(f"No metadata.json found for {src}-{trg} models in {bucket}")
    sys.exit(1)


def _fetch_metadata(bucket, src, trg, model_name):
    """Fetch metadata.json for a model export from GCS."""
    object_name = f"{gcs_prefix()}/{src}-{trg}/{model_name}/{EXPORTED_DIR}/metadata.json"
    url = _gcs_download_url(bucket, object_name)
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to fetch metadata.json: {e}")
        print_help(f"Bucket: {bucket}")
        sys.exit(1)

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        print_error(f"Failed to fetch metadata.json: {response.status_code}")
        print_help(f"Bucket: {bucket}")
        sys.exit(1)

    try:
        return response.json()
    except json.JSONDecodeError:
        print_error("metadata.json could not be parsed")
        print_help(f"Bucket: {bucket}")
        sys.exit(1)


def _list_models(bucket, src, trg):
    """List distinct model names for a language pair from GCS."""
    prefix = f"{gcs_prefix()}/{src}-{trg}"
    items = _list_objects(bucket, prefix)
    models_by_name = {}
    for item in items:
        name = item.get("name")
        if not name or f"/{EXPORTED_DIR}/" not in name:
            continue
        model = GcsModel.from_item(item)
        existing = models_by_name.get(model.name)
        if not existing or model.updated > existing.updated:
            models_by_name[model.name] = model
    return list(models_by_name.values())


def _download_model_files(bucket, src, trg, model_name, dest_dir):
    """Download exported model files from GCS into the target directory."""
    prefix = f"{gcs_prefix()}/{src}-{trg}/{model_name}/{EXPORTED_DIR}/"
    items = _list_objects(bucket, prefix)
    if not items:
        print_error(f"No exported artifacts found for {src}-{trg}: {model_name}")
        print_help(f"Bucket: {bucket}")
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        object_name = item.get("name")
        if not object_name:
            continue
        filename = object_name.split("/")[-1]
        if not filename or not _should_download(filename):
            continue
        dest_path = dest_dir / filename
        if dest_path.exists() and dest_path.stat().st_size > 0:
            continue
        _download_object(bucket, object_name, dest_path)
        _maybe_decompress(dest_path)


def _should_download(filename):
    """Return True for supported model artifact filenames."""
    if filename == "metadata.json":
        return True
    if filename.endswith(".bin") or filename.endswith(".spm"):
        return True
    if filename.endswith(".bin.gz") or filename.endswith(".spm.gz"):
        return True
    if filename.endswith(".bin.zst") or filename.endswith(".spm.zst"):
        return True
    return False


def _list_objects(bucket, prefix):
    """List objects from a GCS bucket using the JSON API."""
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    items = []
    page_token = None

    while True:
        params = {"prefix": prefix}
        if page_token:
            params["pageToken"] = page_token

        try:
            response = requests.get(url, params=params)
        except requests.exceptions.RequestException as e:
            print_error(f"Failed to list GCS objects: {e}")
            print_help(f"Bucket: {bucket}")
            sys.exit(1)

        if response.status_code != 200:
            print_error(f"Failed to list GCS objects: {response.status_code}")
            print_help(f"Bucket: {bucket}")
            sys.exit(1)

        try:
            data = response.json()
        except json.JSONDecodeError:
            print_error("Failed to parse GCS response")
            print_help(f"Bucket: {bucket}")
            sys.exit(1)

        if "error" in data:
            print_error(data["error"].get("message", "Failed to list GCS objects"))
            print_help(f"Bucket: {bucket}")
            sys.exit(1)

        items.extend(data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return items


def _download_object(bucket, object_name, dest_path):
    """Download a single GCS object to a local path."""
    url = _gcs_download_url(bucket, object_name)
    try:
        response = requests.get(url, stream=True)
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to download {object_name}: {e}")
        print_help(f"Bucket: {bucket}")
        sys.exit(1)

    if response.status_code != 200:
        print_error(f"Failed to download {object_name}: {response.status_code}")
        print_help(f"Bucket: {bucket}")
        sys.exit(1)

    with open(dest_path, "wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)


def _maybe_decompress(path):
    """Decompress a supported archive and remove the original."""
    if path.suffix == ".gz":
        _decompress_gzip(path)
        return
    if path.suffix == ".zst":
        _decompress_zstd(path)


def _decompress_gzip(path):
    """Decompress a gzip file to its original name."""
    output_path = path.with_suffix("")
    if output_path.exists():
        path.unlink()
        return
    with gzip.open(path, "rb") as source, open(output_path, "wb") as output:
        shutil.copyfileobj(source, output)
    path.unlink()


def _decompress_zstd(path):
    """Decompress a zstd file to its original name."""
    output_path = path.with_suffix("")
    if output_path.exists():
        path.unlink()
        return
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as source, open(output_path, "wb") as output:
        dctx.copy_stream(source, output)
    path.unlink()


def _gcs_download_url(bucket, object_name):
    """Build a direct download URL for a GCS object."""
    encoded = quote(object_name, safe="/")
    return f"https://storage.googleapis.com/{bucket}/{encoded}"
