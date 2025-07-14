from collections.abc import Iterable
import hashlib
import json
from logging import Logger
import os
import subprocess
import tempfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from random import Random
from typing import Callable, Iterator, Literal, Optional, Set, Union, Dict
from urllib.parse import urlparse
import unicodedata

# We keep this relatively short because these datasets end up in task labels,
# which end up in task cache routes, which need to be <= 256 characters.
DATASET_NAME_MAX_LENGTH = 50


class Dataset:
    """
    Convert a dataset key into a structured format.

    e.g.

    dataset.key               "opus_CCAligned/v1"
    dataset.importer:         "opus"
    dataset.name:             "CCAligned/v1"
    dataset.file_safe_key():  "opus_CCAligned_v1"
    dataset.file_safe_name(): "CCAligned_v1"
    """

    def __init__(self, dataset_key: str) -> None:
        key_parts = dataset_key.split("_")

        self.key = dataset_key
        self.importer = key_parts[0]
        self.name = "_".join(key_parts[1:])

        if not self.importer:
            raise Exception(f"Could not find the importer in the dataset key {dataset_key}")

        if not self.name:
            raise Exception(f"Could not find the name in the dataset key {dataset_key}")

    # Important! Keep in sync with dataset_helpers.py.
    def _escape(dataset: str) -> str:
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
        )

    def file_safe_key(self) -> str:
        return Dataset._escape(self.key)

    def file_safe_name(self) -> str:
        return Dataset._escape(self.name)


def shuffle_with_max_lines(
    line_stream: Iterator[str],
    seed: int | float | str | bytes | bytearray | None,
    max_lines: int,
    total_byte_size: Optional[int] = None,
    estimate_total_byte_size: Optional[Callable[[float], int]] = None,
) -> list[str]:
    """
    Shuffle a line stream, but only retain up to a maximum number of lines in memory.
    Note that the final ordering is determined by the seed and the contents of the file. So
    running this multiple times on the same dataset will return the same result, but running
    it with the same seed and different content will create a different ordering.

    Only run for monolingual data or where the parallel sentences are in the same line and
    separated by a delimiter.

    The distribution should be even unless the initial content is not representative of the
    general size of the sentences, in this case the distribution will be slightly biased. See
    the test cases for more in-depth examples.

    These options are mutually exclusive, and one must be provided:
    - total_byte_size - The byte size of the lines.
    - estimate_total_byte_size - An estimate of the size of the corpus after max_lines have been
                                 filled. The average bytes per line is provided
    """
    lines: list[str] = []

    random = Random(seed)  # Make this deterministic based on dataset key.

    total_bytes = 0

    if total_byte_size is None:
        assert (
            estimate_total_byte_size
        ), "Either total_byte_size or estimate_total_byte_size must be provided"

    # Fill up the lines up until the max, and measure the total bytes.
    for line in line_stream:
        # Encoding returns the underlying byte representation which is then measured.
        total_bytes = total_bytes + len(line.encode("utf-8"))

        lines.append(line)

        if len(lines) == max_lines:
            break

    if total_byte_size is None:
        total_byte_size = estimate_total_byte_size(float(total_bytes) / float(max_lines))

    line_index = len(lines)
    random.shuffle(lines)

    # Consume the rest of the line stream, but sample based on the probability that adding
    # something to the collection will be representative.

    for i, line in enumerate(line_stream):
        # Continuously adjust this estimation in case the first sampled data is not representative.
        total_bytes = total_bytes + len(line.encode("utf-8"))
        average_bytes_per_line = total_bytes / (max_lines + i + 1)
        estimated_lines = total_byte_size / average_bytes_per_line
        line_sampling_probability = max_lines / estimated_lines

        if random.random() < line_sampling_probability:
            if len(lines) == max_lines:
                # Treat the `lines` list as a ring buffer since we've reached the max lines. As new
                # lines are randomly sampled, old randomly sampled lines roll out of the buffer.
                lines[line_index % max_lines] = line
                line_index += 1
            else:
                # Python throws "IndexError: list assignment index out of range" if you attempt
                # to assign outside the existing range, so use an append here.
                lines.append(line)

    # Do a final shuffle to ensure that the newly sampled lines are shuffled with the original
    # set of shuffled lines.
    random.shuffle(lines)

    return lines


def shuffle_in_temp_files(
    line_stream: Iterator[str],
    output: TextIOWrapper,
    seed: str,
    chunk_bytes: int,
    bucket_bytes: int,
    chunk_dir: Optional[str] = tempfile.gettempdir(),
    keep_chunks=False,
):
    """
    Shuffle large datasets by storing chunks to the file system. The ordering is guaranteed to be
    stable across two datasets as long as they are the same length. For instance it could be used
    to shuffle `dataset.en.zst` and `dataset.ca.zst` the same if the two are parallel sentences.

    Take in a stream of lines (from a download, or stdin) and split it out to chunks.

    tmpdir
    ├── chunk.1
    ├── chunk.2
    ├── chunk.3
    ├── chunk.4
    ├── ...
    └── chunk.100

    After the entire dataset is written to chunks, pick random chunks and put them into a
    bucket. Only one bucket is fully loaded into memory at a time, and the contents
    of the bucket is shuffled in memory.

    Bucket:
    ┌───────────┐
    │ chunk.85  │
    │ chunk.3   │
    │ chunk.52  │
    │ chunk.30  │
    │ chunk.12  │
    │ chunk.18  │
    └───────────┘

    • shuffle bucket lines
    • write to output

    At most 1 bucket will be held in memory. At most the dataset + 1 bucket of file space will be
    needed when running this algorithm.
    """
    random = Random(seed)

    chunk_index = 0
    chunk_file = open(os.path.join(chunk_dir, f"chunk.{chunk_index}"), "wt")

    # Write out the chunks to disk.
    bytes_written_to_chunk = 0
    for line in line_stream:
        line_bytes = len(line.encode("utf-8")) + 1

        if bytes_written_to_chunk + line_bytes > chunk_bytes:
            # Start a new chunk.
            chunk_file.close()
            chunk_index += 1
            chunk_file = open(os.path.join(chunk_dir, f"chunk.{chunk_index}"), "wt")
            bytes_written_to_chunk = 0

        chunk_file.write(line + "\n")
        bytes_written_to_chunk += line_bytes

    chunk_file.close()

    # Shuffle the chunk indexes
    chunk_count = chunk_index + 1

    shuffled_chunk_indexes = [*range(chunk_count)]
    random.shuffle(shuffled_chunk_indexes)

    # Load a single bucket into memory, discarding the chunks.
    bucket_count = 0
    bytes_in_bucket = 0
    bucket = []

    for chunk_index in shuffled_chunk_indexes:
        chunk_name = os.path.join(chunk_dir, f"chunk.{chunk_index}")

        # Read in the chunk line by line.
        with open(chunk_name, "r") as file:
            for line in file.readlines():
                bucket.append(line)
                bytes_in_bucket += len(line.encode("utf-8"))

                # If the bucket overflows, shuffle and write it out.
                if bytes_in_bucket > bucket_bytes:
                    random.shuffle(bucket)
                    for shuffled_line in bucket:
                        output.write(shuffled_line)

                    # Create the new bucket.
                    bucket = []
                    bytes_in_bucket = 0
                    bucket_count += 1

        if not keep_chunks:
            os.remove(chunk_name)

    if len(bucket) > 0:
        random.shuffle(bucket)
        for shuffled_line in bucket:
            output.write(shuffled_line)

    print(f"Shuffled with {bucket_count} buckets.")


class Statistics:
    """
    Base class for handling statistical data and JSON serialization in the pipeline. All
    public data attributes in the implementing class will be saved as JSON. This class
    standardizes how the JSON is generated, and where it is saved.

    You can derive data at JSON generation time by providing an update_derived_data method.

    For instance stats.save_json() for Statistics("nllb.en.zst") would produce "nllb.en.stats.json".
    """

    def __init__(self, dataset_path: Optional[Union[Path, str]] = None) -> None:
        self._dataset_path = Path(dataset_path) if dataset_path else None

    def save_json(self) -> Path:
        """
        Standardizes how the JSON is saved, based on the dataset.
        """
        if not self._dataset_path:
            raise Exception("A dataset_path is required when saving to JSON.")

        path = self._dataset_path.parent / f"{self._dataset_path.stem}.stats.json"
        obj = self.as_json()
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(obj, json_file, indent=2)
            json_file.write("\n")
        return path

    def _is_subclass(value: any):
        """
        Determine if a child object is a subclass or not.
        """
        try:
            return issubclass(value.__class__, Statistics)
        except AttributeError:
            return False

    def as_json(root: Union[int, str, float, list, "Statistics"]) -> Union[int, str, float, list]:
        """
        Recursively walk the data attributes of the statistics.
        """
        if Statistics._is_subclass(root):
            stats: Statistics = root
            stats.update_derived_data()
            obj = {}
            for key, value in stats.__dict__.items():
                if key.startswith("_"):
                    continue
                obj[key] = Statistics.as_json(value)

            return obj

        if isinstance(root, list):
            return [Statistics.as_json(item) for item in root]

        if isinstance(root, dict):
            root_dict: dict = root
            return {key: Statistics.as_json(value) for key, value in root_dict.items()}

        if isinstance(root, (float, int, str)):
            return root

        return str(root)

    def update_derived_data(self):
        """
        Update any derived data in the sub values. Override this method if anything
        needs to be derived.
        """
        pass


class FilteringStep(Statistics):
    """
    For each step for filtering, store how many were kept or filtered.
    """

    def __init__(
        self, description: str, filtered=0, kept=0, dataset_path: Optional[Path] = None
    ) -> None:
        super().__init__(dataset_path)
        self.description = description
        self.filtered = filtered
        self.kept = kept
        self.visited = 0

    def update_derived_data(self):
        super().update_derived_data()
        # Only two of the values need to be kept up to date, the last can be computed.
        if not self.visited:
            self.visited = self.filtered + self.kept
        elif self.filtered and not self.kept:
            self.kept = self.visited - self.filtered
            return
        elif self.kept and not self.filtered:
            self.filtered = self.visited - self.kept


@dataclass
class CountingStep(Statistics):
    """
    This is just a single value that is being counted.
    """

    value: int
    description: str

    def __init__(
        self,
        description: str,
        value=0,
        dataset_path: Optional[Path] = None,
    ) -> None:
        super().__init__(dataset_path)
        self.description = description
        self.value = value


class WeakStringSet(Set):
    """
    A Set that weakly holds on to strings by storing a hashed `int`. Using this class
    makes it easy to see if a string is duplicated across large datasets without holding
    the entire set of strings in memory.

    Usage:
        unique_strings = WeakStringSet()
        unique_strings.add("string a")
        unique_strings.add("string b")

        assert "string a" in unique_strings
        assert "string b" in unique_strings
        assert "string c" not in unique_strings
    """

    def __init__(self, iter: Optional[Iterable[str]] = None) -> None:
        if iter:
            super().__init__((WeakStringSet._hash_string(string) for string in iter))
        else:
            super().__init__()

    def __contains__(self, string: str) -> bool:
        return super().__contains__(WeakStringSet._hash_string(string))

    def add(self, string: str) -> None:
        """
        Add a string to the weak set. The strings are stored uniquely based on their
        contents with the whitespace surrounding them stripped.
        """
        super().add(WeakStringSet._hash_string(string))

    def update(self, iter: Iterable[str]):
        super().update((WeakStringSet._hash_string(string) for string in iter))

    def remove(self, string: str):
        super().remove(WeakStringSet._hash_string(string))

    def discard(self, string: str):
        super().discard(WeakStringSet._hash_string(string))

    def _hash_string(string: str) -> int:
        """
        Return a hash of a line. The line has its whitespace stripped and text representation
        normalized to ensure a consistent representation.
        """
        cleaned_line = unicodedata.normalize("NFC", string.strip())
        return hash(cleaned_line)


class WeakStringDict(Dict):
    """
    A Dict that weakly holds on to key strings by storing a hashed `int`. Using this class
    makes it easy to see if a string is duplicated across large datasets without holding
    the entire set of strings in memory.

    This is an alternate version of WeakStringSet that also stores a float value (score)
    associated to the string.

    Usage:
        unique_strings = WeakStringDict()
        unique_strings["string a"] = 0.78
        unique_strings["string b"] = 0.911

        assert "string a" in unique_strings
        assert "string b" in unique_strings
        assert "string c" not in unique_strings
    """

    def __init__(self, iter: Optional[Iterable[str]] = None) -> None:
        if iter:
            super().__init__((WeakStringDict._hash_string(string) for string in iter))
        else:
            super().__init__()

    def __contains__(self, string: str) -> bool:
        return super().__contains__(WeakStringDict._hash_string(string))

    def __setitem__(self, string: str, val: float) -> None:
        """
        Add/set a string the weak dict as key and its value associated.
        The strings are stored uniquely based on their
        contents with the whitespace surrounding them stripped.
        """
        super().__setitem__(WeakStringDict._hash_string(string), val)

    def __delitem__(self, string: str):
        super().__delitem__(WeakStringDict._hash_string(string))

    def __getitem__(self, string: str) -> float:
        return super().__getitem__(WeakStringDict._hash_string(string))

    def _hash_string(string: str) -> int:
        """
        Return a hash of a line. The line has its whitespace stripped and text representation
        normalized to ensure a consistent representation.
        """
        cleaned_line = unicodedata.normalize("NFC", string.strip())
        return hash(cleaned_line)


def decompress(
    source: Union[str, Path],
    destination: Optional[Union[Path, str]] = None,
    remove: bool = False,
    logger: Optional[Logger] = None,
) -> Path:
    """
    Decompresses a file using the appropriate command based on its file extension.

    Args:
    file_path: The path to the file to be decompressed
    remove: If set to `True`, the original compressed file will be removed after decompression.
    destination: Be default the file will be decompressed next to the original. This arguments
                 allows for overriding the destination.
    logger: Log information about the decompression
    """
    if isinstance(source, str):
        source = Path(source)
    if not destination:
        destination = source.parent / source.stem

    if logger:
        logger.info(f"[decompress] From: {source}")
        logger.info(f"[decompress] To: {destination}")

    if source.suffix == ".zst":
        command = ["zstdmt", "--decompress", "--force", "-o", destination, source]
        if remove:
            command.append("--rm")

        subprocess.check_call(command)
    elif source.suffix == ".gz":
        command = ["gzip", "-c", "-d", source]
        with open(destination, "wb") as out_file:
            subprocess.check_call(command, stdout=out_file)
        if remove:
            source.unlink()
    else:
        raise Exception(f"Unknown file type to decompress: {source}")

    if remove:
        logger.info(f"[decompress] Removed: {source}")

    return destination


def compress(
    source: Union[str, Path],
    destination: Optional[Union[Path, str]] = None,
    remove: bool = False,
    compression_type: Union[Literal["zst"], Literal["gz"]] = None,
    logger: Optional[Logger] = None,
) -> Path:
    """
    Compress a file using the appropriate command based on its file extension.

    Args:
    source:     The path to the file to be compressed
    destination: Be default the file will be compressed next to the original. This arguments
                 allows for overriding the destination.
    remove:      If set to `True`, the original decompressed file will be removed.
    type:        The type defaults to "zst", and is implied by the destination, however it can
                 be explicitly set.
    logger:      Log information about the compression
    """
    if isinstance(source, str):
        source = Path(source)
    if isinstance(destination, str):
        destination = Path(destination)

    # Ensure the compression type is valid and present
    if compression_type and destination:
        assert f".{type}" == destination.suffix, "The compression type and destination must match."

    if not compression_type:
        if destination:
            compression_type = destination.suffix[1:]
        else:
            compression_type = "zst"

    # Set default destination if not provided
    if not destination:
        destination = source.with_suffix(f"{source.suffix}.{compression_type}")

    if logger:
        logger.info(f"Compressing: {source}")
        logger.info(f"Destination: {destination}")

    if compression_type == "zst":
        command = ["zstdmt", "--compress", "--force", "--quiet", source, "-o", destination]
        if remove:
            command.append("--rm")
        subprocess.check_call(command)
    elif compression_type == "gz":
        with open(destination, "wb") as out_file:
            subprocess.check_call(["gzip", "-c", "--force", source], stdout=out_file)
        if remove:
            source.unlink()
    else:
        raise ValueError(f"Unsupported compression type: {compression_type}")

    if remove:
        logger.info(f"Removed {source}")

    return destination
