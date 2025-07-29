import gzip
import io
import json
import os
import shutil
import time
from contextlib import ExitStack, contextmanager
from io import BufferedReader
from pathlib import Path
from typing import Any, Callable, Generator, Literal, Optional, Union
from zipfile import ZipFile

import requests
from zstandard import ZstdCompressor, ZstdDecompressor

from pipeline.common import format_bytes
from pipeline.common.logging import get_logger

logger = get_logger(__file__)


class DownloadException(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


def stream_download_to_file(url: str, destination: Union[str, Path], timeout_sec=10.0) -> None:
    """
    Streams a download to a file, and retries several times if there are any failures. The
    destination file must not already exist.
    """
    if os.path.exists(destination):
        raise DownloadException(f"That file already exists: {destination}")

    logger.info(f"Destination: {destination}")

    # If this is mocked for a test, use the locally mocked path.
    mocked_location = get_mocked_downloads_file_path(url)
    if mocked_location:
        shutil.copy(mocked_location, destination)
        return

    try:
        with open(destination, "wb") as file, DownloadChunkStreamer(url, timeout_sec=timeout_sec) as chunk_streamer:
            for chunk in chunk_streamer.download_chunks():
                file.write(chunk)
    except DownloadException:
        Path(destination).unlink()
        raise


def get_mocked_downloads_file_path(url: str) -> Optional[str]:
    """If there is a mocked download, get the path to the file, otherwise return None"""
    mocked_downloads_str = os.environ.get("MOCKED_DOWNLOADS")
    if not mocked_downloads_str:
        return None

    mocked_downloads = json.loads(mocked_downloads_str)

    if not isinstance(mocked_downloads, dict):
        raise DownloadException(
            "Expected the mocked downloads to be a json object mapping the URL to file path"
        )

    source_file = mocked_downloads.get(url)
    if not source_file:
        print("MOCKED_DOWNLOADS:", json.dumps(mocked_downloads, indent=2))
        raise DownloadException(f"Received a URL that was not in MOCKED_DOWNLOADS {url}")

    if not os.path.exists(source_file):
        raise DownloadException(f"The source file specified did not exist {source_file}")

    logger.info("Mocking a download.")
    logger.info(f"   url: {url}")
    logger.info(f"  file: {source_file}")

    return source_file


def location_exists(location: str):
    """
    Checks if a location (url or file path) exists.
    """
    if get_mocked_downloads_file_path(location):
        return True

    if location.startswith("http://") or location.startswith("https://"):
        response = requests.head(location, allow_redirects=True)
        return response.ok
    return os.path.exists(location)


def attempt_mocked_request(url: str) -> Optional[BufferedReader]:
    """
    If there are mocked download, use that.
    """
    file_path = get_mocked_downloads_file_path(url)
    if file_path:
        return open(file_path, "rb")
    return None


def get_download_size(url: str) -> int:
    """Get the total bytes of a file to download."""
    mocked_file_path = get_mocked_downloads_file_path(url)
    if mocked_file_path:
        return os.path.getsize(mocked_file_path)

    response = requests.head(url, allow_redirects=True)
    size = response.headers.get("content-length", 0)
    return int(size)


class RemoteDecodingLineStreamer:
    """
    Base class to stream lines directly from a remote file.
    """

    def __init__(self, url: str) -> None:
        self.url = url

        self.decoding_stream = None
        self.byte_chunk_stream = None
        self.line_stream = None

    def __enter__(self):
        mocked_request = attempt_mocked_request(self.url)
        if mocked_request:
            # We are in a test.
            logger.info(f"Using a mocked download: {self.url}")
            self.byte_chunk_stream = mocked_request
            self.decoding_stream = self.decode(self.byte_chunk_stream)
        else:
            self.byte_chunk_stream = DownloadChunkStreamer(self.url).__enter__()
            self.decoding_stream = self.decode(self.byte_chunk_stream)

        self.line_stream = io.TextIOWrapper(self.decoding_stream, encoding="utf-8")

        return self.line_stream

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        if self.line_stream:
            self.line_stream.close()
        if self.decoding_stream:
            self.decoding_stream.close()
        if self.byte_chunk_stream:
            self.byte_chunk_stream.close()

    def decode(self, byte_stream: Any) -> Any:
        # This byte stream requires no decoding, so just pass it on through.
        return byte_stream


class RemoteGzipLineStreamer(RemoteDecodingLineStreamer):
    """
    Stream lines directly from a remote gzip file. The line includes the newlines separator.

    Usage:

        with RemoteGzipLineStreamer(url) as lines:
            for line in lines:
                print(line)
    """

    def decode(self, byte_stream):
        return gzip.GzipFile(fileobj=byte_stream)


class RemoteZstdLineStreamer(RemoteDecodingLineStreamer):
    """
    Stream lines directly from a remote zstd file. The line includes the newlines separator.

    Usage:

        with RemoteZstdLineStreamer(url) as lines:
            for line in lines:
                print(line)
    """

    def decode(self, byte_stream):
        return ZstdDecompressor().stream_reader(byte_stream)


class DownloadChunkStreamer(io.IOBase):
    """
    Streams a download as chunks, and retries several times if there are any failures. This
    clas implements io.IOBase so it can be used as a file reader.

    Iterator over chunks directly:

        with DownloadChunkStreamer(url) as chunk_streamer:
            for chunk in chunk_streamer.download_chunks():
                f.write(chunk)

    Or pass it as a file handle:

        with DownloadChunkStreamer(url) as f:
             gzip.GzipFile(fileobj=f)
    """

    def __init__(self, url: str, total_retries=3, timeout_sec=10.0, wait_before_retry_sec=60.0):
        self.url = url
        self.response = None

        # How many retry attempts should there be, and how long to wait between retries.
        self.total_retries = total_retries
        self.wait_before_retry_sec = wait_before_retry_sec

        # How long to wait for a response to timeout? This is the time that no new data is received.
        self.timeout_sec = timeout_sec

        self.report_every = 0.05  # What percentage of the download to report updates?
        self.next_report_percent = self.report_every  # The next report percentage.

        self.downloaded_bytes = 0
        self.chunk_bytes = 8 * 1024

        # The buffered `read` data.
        self.buffer = b""

        # The Generator result of _download_chunks.
        self.chunk_iter: Optional[Generator[bytes, None, None]] = None

    def __enter__(self):
        """
        On enter, kick off the download, and store the chunk iterator. This iterator
        handles the restarts for Requests.
        """
        self.chunk_iter = self.download_chunks()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Close out the response, and cancel any iterators.
        """
        if self.response:
            self.response.close()

        self.response = None
        self.chunk_iter = None

    def read(self, size=-1) -> bytes:
        """
        This method implements the io.IOBase read method. It buffers the chunks until the `size`
        requirement is fulfilled. It is backed by the chunks_iter created by the download_chunks
        method.
        """
        if not self.chunk_iter:
            # The chunk iterator was consumed. Return an empty byte object to indicate the download
            # is complete.
            return b""

        if size < 0:
            # Load everything into the buffer, and return it.
            for chunk in self.chunk_iter:
                self.buffer += chunk
            result = self.buffer
            self.buffer = b""
            return result

        # Load the buffer with requested amount of data to read. -1 indicates load everything.
        while len(self.buffer) < size:
            chunk = next(self.chunk_iter, None)
            if chunk:
                self.buffer += chunk
            else:
                # The stream ended.
                break

        # Return the requested read amount, and divide up the remaining buffer.
        result = self.buffer[:size]
        self.buffer = self.buffer[size:]

        return result

    def readable(self):
        return True

    def download_chunks(self) -> Generator[bytes, None, None]:
        """
        This method is the generator that is responsible for running the request, and retrying
        when there is a failure. It yields the fixed size byte chunks, and exposes a generator
        to be consumed. This generator can be used directly in a for loop, or the entire class
        can be passed in as a file handle.
        """
        next_report_percent = self.report_every
        total_bytes = 0
        exception = None

        for retry in range(self.total_retries):
            if retry > 0:
                logger.error(f"Remaining retries: {self.total_retries - retry}")

            try:
                headers = {}
                if self.downloaded_bytes > 0:
                    # Pick up the download from where it was before.
                    headers = {"Range": f"bytes={self.downloaded_bytes}-"}

                self.response = requests.get(
                    self.url, headers=headers, stream=True, timeout=self.timeout_sec
                )
                self.response.raise_for_status()

                # Report the download size.
                if not total_bytes and "content-length" in self.response.headers:
                    total_bytes = int(self.response.headers["content-length"])
                    logger.info(f"Download size: {total_bytes:,} bytes")

                for chunk in self.response.iter_content(chunk_size=self.chunk_bytes):
                    if not chunk:
                        continue

                    self.downloaded_bytes += len(chunk)

                    # Report the percentage downloaded every `report_every` percentage.
                    if total_bytes and self.downloaded_bytes >= next_report_percent * total_bytes:
                        logger.info(
                            f"{self.downloaded_bytes / total_bytes * 100.0:.0f}% downloaded "
                            f"({self.downloaded_bytes}/{total_bytes} bytes)"
                        )
                        next_report_percent += self.report_every

                    yield chunk

                # The download is complete.
                self.close()
                logger.info("100% downloaded - Download finished.")
                return

            except requests.exceptions.Timeout as error:
                logger.error(f"The connection timed out: {error}.")
                exception = error

            except requests.exceptions.RequestException as error:
                # The RequestException is the generic error that catches all classes of "requests"
                # errors. Don't attempt to be be smart about this, just attempt again until
                # the retries are done.
                logger.error(f"A download error occurred: {error}")
                exception = error

            # Close out the response on an error. It will be recreated when retrying.
            if self.response:
                self.response.close()
                self.response = None

            logger.info(f"Retrying in {self.wait_before_retry_sec} sec")
            time.sleep(self.wait_before_retry_sec)

        self.close()
        raise DownloadException("The download failed.") from exception

    def decode(self, byte_stream) -> Generator[bytes, None, None]:
        """Pass through the byte stream. This method can be specialized by child classes."""
        return byte_stream


@contextmanager
def _read_lines_multiple_files(
    files: list[Union[str, Path]],
    encoding: str,
    path_in_archive: Optional[str],
    on_enter_location: Optional[Callable[[str], None]] = None,
) -> Generator[Generator[str, None, None], None, None]:
    """
    Iterates through each line in multiple files, combining it into a single stream.
    """

    stack = None

    def iter(stack: ExitStack):
        for file_path in files:
            logger.info(f"Reading lines from: {file_path}")
            lines = stack.enter_context(
                read_lines(file_path, path_in_archive, on_enter_location, encoding=encoding)
            )
            yield from lines
            stack.close()

    try:
        stack = ExitStack()
        yield iter(stack)
    finally:
        if stack:
            stack.close()


@contextmanager
def _read_lines_single_file(
    location: Path | str,
    encoding: str,
    path_in_archive: Optional[str] = None,
    on_enter_location: Optional[Callable[[str], None]] = None,
) -> Generator[Generator[str, None, None], None, None]:
    """
    A smart function to efficiently stream lines from a local or remote file.
    The location can either be a URL or a local file system path.
    It handles gzip, zst, and plain text files.

    Args:
        location - URL or file path
        path_in_archive  - The path to a file in a zip archive
        on_enter_location - A lambda for when a new location is entered
    """
    location = str(location)
    if on_enter_location:
        on_enter_location(location)

    if location.startswith("http://") or location.startswith("https://"):
        # If this is mocked for a test, use the locally mocked path.
        mocked_location = get_mocked_downloads_file_path(location)
        if mocked_location:
            location = mocked_location

    stack = ExitStack()

    try:
        if location.startswith("http://") or location.startswith("https://"):
            # This is a remote file.

            response = requests.head(location, allow_redirects=True)
            content_type = response.headers.get("Content-Type")
            if content_type == "application/gzip":
                yield stack.enter_context(RemoteGzipLineStreamer(location))  # type: ignore[reportReturnType]

            elif content_type == "application/zstd":
                yield stack.enter_context(RemoteZstdLineStreamer(location))  # type: ignore[reportReturnType]

            elif content_type == "application/zip":
                raise DownloadException("Streaming a zip from a remote location is supported.")

            elif content_type == "text/plain":
                yield stack.enter_context(RemoteDecodingLineStreamer(location))  # type: ignore[reportReturnType]

            elif location.endswith(".gz") or location.endswith(".gzip"):
                yield stack.enter_context(RemoteGzipLineStreamer(location))  # type: ignore[reportReturnType]

            elif location.endswith(".zst"):
                yield stack.enter_context(RemoteZstdLineStreamer(location))  # type: ignore[reportReturnType]
            else:
                # Treat as plain text.
                yield stack.enter_context(RemoteDecodingLineStreamer(location))  # type: ignore[reportReturnType]

        else:  # noqa: PLR5501
            # This is a local file.
            if location.endswith(".gz") or location.endswith(".gzip"):
                yield stack.enter_context(gzip.open(location, "rt", encoding=encoding))  # type: ignore[reportReturnType]

            elif location.endswith(".zst"):
                input_file = stack.enter_context(open(location, "rb"))
                zst_reader = stack.enter_context(ZstdDecompressor().stream_reader(input_file))
                yield stack.enter_context(io.TextIOWrapper(zst_reader, encoding=encoding))  # type: ignore[reportReturnType]

            elif location.endswith(".zip"):
                if not path_in_archive:
                    raise DownloadException("Expected a path into the zip file.")
                zip = stack.enter_context(ZipFile(location, "r"))
                if path_in_archive not in zip.namelist():
                    raise DownloadException(
                        f"Path did not exist in the zip file: {path_in_archive}"
                    )
                file = stack.enter_context(zip.open(path_in_archive, "r"))
                yield stack.enter_context(io.TextIOWrapper(file, encoding=encoding))  # type: ignore[reportReturnType]
            else:
                # Treat as plain text.
                yield stack.enter_context(open(location, "rt", encoding=encoding))  # type: ignore[reportReturnType]
    finally:
        stack.close()


def read_lines(
    location_or_locations: Union[Path, str, list[Union[str, Path]]],
    path_in_archive: Optional[str] = None,
    on_enter_location: Optional[Callable[[str], None]] = None,
    encoding="utf-8",
):
    """
    A smart function to efficiently stream lines from a local or remote file.
    The location can either be a URL or a local file system path.
    It handles gzip, zst, and plain text files.
    It can also handle a list of files.

    Args:
        location_or_locations - A single URL or file path, or a list
        path_in_archive  - The path to a file in a zip archive

    Usage:
        with read_lines("output.txt.gz") as lines:
            for line in lines:
                print(line)

        paths = [
            "http://example.com/file.txt.gz",
            "path/to/file.txt.zst",
        ]
        with read_lines(paths) as lines:
            for line in lines:
                print(line)
    """

    if isinstance(location_or_locations, list):
        return _read_lines_multiple_files(
            location_or_locations, encoding, path_in_archive, on_enter_location
        )

    return _read_lines_single_file(
        location_or_locations, encoding, path_in_archive, on_enter_location
    )


@contextmanager
def write_lines(path: Path | str, encoding="utf-8"):
    """
    A smart function to create a context to write lines to a file. It works on .zst, .gz, and
    raw text files. It reads the extension to determine the file type. If writing out a raw
    text file, for instance a sample of a dataset that is just used for viewing, include a
    "byte order mark" so that the browser can properly detect the encoding.

    with write_lines("output.txt.gz") as output:
        output.write("writing a line\n")
        output.write("writing a second lines\n")
    """

    stack = None
    try:
        path = str(path)
        stack = ExitStack()

        if path.endswith(".zst"):
            file = stack.enter_context(open(path, "wb"))
            compressor = stack.enter_context(ZstdCompressor().stream_writer(file))
            yield stack.enter_context(io.TextIOWrapper(compressor, encoding=encoding))
        elif path.endswith(".gz"):
            yield stack.enter_context(gzip.open(path, "wt", encoding=encoding))
        else:
            yield stack.enter_context(open(path, "wt", encoding=encoding))

    finally:
        if stack:
            stack.close()


def count_lines(path: Path | str) -> int:
    """
    Similar to wc -l, this counts the lines in a file. However, this command does so regardless
    of the compression strategy used on the file.
    """
    with read_lines(path) as lines:
        return sum(1 for _ in lines)


def is_file_empty(path: Path | str) -> bool:
    """
    Attempts to read a line to determine if a file is empty or not. Works on local or remote files
    as well as compressed or uncompressed files.
    """
    with read_lines(path) as lines:
        try:
            next(lines)
            return False
        except StopIteration:
            return True


def get_file_size(location: Path | str) -> int:
    """Get the size of a file, whether it is remote or local."""
    if isinstance(location, str) and (
        location.startswith("http://") or location.startswith("https://")
    ):
        return get_download_size(location)
    return os.path.getsize(location)


def get_human_readable_file_size(location: Path | str) -> tuple[str, int]:
    """Get the size of a file in a human-readable string, and the numeric bytes."""
    bytes = get_file_size(location)
    return format_bytes(bytes), bytes


def compress_file(
    path: Path | str,
    keep_original: bool = True,
    compressed_path: Optional[Path | str] = None,
    compression: Literal["zst", "gz"] = "zst",
) -> Path:
    """
    Compresses a file to .zst or .gz format. It returns the path of the compressed file.
    "zst" is the preferred compression scheme.
    """
    path = Path(path)

    if compression == "zst":
        if not compressed_path:
            compressed_path = Path(str(path) + ".zst")
        cctx = ZstdCompressor()
        with open(path, "rb") as infile:
            with open(compressed_path, "wb") as outfile:
                outfile.write(cctx.compress(infile.read()))

    elif compression == "gz":
        if not compressed_path:
            compressed_path = Path(str(path) + ".gz")
        with open(path, "rb") as infile:
            with gzip.open(compressed_path, "wb") as outfile:
                outfile.write(infile.read())

    else:
        raise ValueError(f"Unsupported compression format: {compression}")

    if not keep_original:
        # Delete the original file
        path.unlink()

    return Path(compressed_path)


def decompress_file(
    path: Union[str, Path],
    keep_original: bool = True,
    decompressed_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Decompresses a .gz or .zst file. It returns the path of the decompressed file.
    """
    path = Path(path)

    if decompressed_path:
        decompressed_path = Path(decompressed_path)
    else:
        # Remove the original suffix
        decompressed_path = path.with_suffix("")

    with ExitStack() as stack:
        decompressed_file = stack.enter_context(decompressed_path.open("wb"))

        if path.suffix == ".gz":
            compressed_file = stack.enter_context(gzip.open(str(path), "rb"))
            decompressed_file.write(compressed_file.read())
            while True:
                # Write the data out in chunks so that all of the it doesn't need to be
                # into memory.
                chunk = compressed_file.read(10_240)
                if not chunk:
                    break
                decompressed_file.write(chunk)

        elif path.suffix == ".zst":
            compressed_file = stack.enter_context(open(path, "rb"))
            for chunk in ZstdDecompressor().read_to_iter(compressed_file):
                # Write the data out in chunks so that all of the it doesn't need to be
                # into memory.
                decompressed_file.write(chunk)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

    if not keep_original:
        # Delete the original file
        path.unlink()

    return decompressed_path
