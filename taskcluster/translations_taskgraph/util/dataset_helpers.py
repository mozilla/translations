from pathlib import Path
from urllib.parse import urlparse
import hashlib


# We keep this relatively short because these datasets end up in task labels,
# which end up in task cache routes, which need to be <= 256 characters.
DATASET_NAME_MAX_LENGTH = 50


# Important! Keep in sync with `Dataset._escape` in pipeline/common/datasets.py.
def sanitize_dataset_name(dataset: str) -> str:
    # URLs can be too large when used as Taskcluster labels. Create a nice identifier for them.
    # See https://github.com/mozilla/translations/issues/527
    if dataset.startswith("https://") or dataset.startswith("http://"):
        url = urlparse(dataset)

        hostname = url.hostname
        if hostname == "storage.googleapis.com":
            hostname = "gcp"

        # Get the name of the file from theh path without the extension.
        file = Path(url.path).stem
        file = file.replace(".[LANG]", "").replace("[LANG]", "")

        # Compute a hash to avoid any name collisions.
        md5 = hashlib.md5()
        md5.update(dataset.encode("utf-8"))
        hash = md5.hexdigest()[:6]

        dataset = f"{hostname}_{file}_{hash}"
    # Even non-URL datasets can be too long, for example:
    # mtdata_ELRC-convention_against_torture_other_cruel_inhuman_or_degrading_treatment_or_punishment_united_nations-1-ell-eng
    # We need to truncate and hash any that are over a certain length
    elif len(dataset) > DATASET_NAME_MAX_LENGTH:
        md5 = hashlib.md5()
        md5.update(dataset.encode("utf-8"))
        hash = md5.hexdigest()[:6]

        truncated = dataset[:DATASET_NAME_MAX_LENGTH]
        dataset = f"{truncated}_{hash}"

    return (
        dataset.replace("://", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("@", "_")
    )
