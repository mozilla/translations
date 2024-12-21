import logging
from pathlib import Path
import subprocess
import threading
import time

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

STOP_BYTE_COUNT_LOGGER = False
STOP_GPU_LOGGER = False


def get_logger(name: str):
    """
    Get a logger using the __file__ name.

    For example in pipeline/bicleaner/download_pack.py

        logger = get_logger(__file__)
        logger.info("This is a log.")

    Will log:

        > [download_pack] This is a log.
    """

    logger = logging.getLogger(Path(name).stem)
    logger.setLevel(logging.INFO)
    return logger


def _log_gpu_stats(logger: logging.Logger, interval_seconds: int):
    # Only load gpustat when it's needed.
    import gpustat

    global STOP_GPU_LOGGER
    while True:
        time.sleep(interval_seconds)
        if STOP_GPU_LOGGER:
            STOP_GPU_LOGGER = False
            return
        try:
            logger.info("[gpu] Current GPU stats:")
            gpustat.print_gpustat()
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to retrieve GPU stats: {e}")


def stop_gpu_logging():
    global STOP_GPU_LOGGER
    STOP_GPU_LOGGER = True


def start_gpu_logging(logger: logging.Logger, interval_seconds: int):
    """Logs GPU stats on an interval using gpustat in a background thread."""
    assert not STOP_GPU_LOGGER, "A gpu logger should not already be running"

    thread = threading.Thread(
        target=_log_gpu_stats,
        # Set as a daemon thread so it automatically is closed on shutdown.
        daemon=True,
        args=(logger, interval_seconds),
    )
    thread.start()


def _log_byte_rate(logger: logging.Logger, interval_seconds: int, file_path: Path):
    global STOP_BYTE_COUNT_LOGGER
    previous_byte_count = 0
    previous_time = time.time()
    is_zst = file_path.suffix == ".zst"

    while True:
        time.sleep(interval_seconds)
        if STOP_BYTE_COUNT_LOGGER:
            STOP_BYTE_COUNT_LOGGER = False
            return

        try:
            if is_zst:
                # This takes ~1 second to run on 5 million sentences.
                current_byte_count = 0
                cmd = ["zstd", "-dc", str(file_path)]
                with subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ) as process:
                    for chunk in iter(lambda: process.stdout.read(8192), b""):
                        current_byte_count += len(chunk)
            else:
                # This is pretty much instantaneous.
                result = subprocess.run(
                    ["wc", "-c", str(file_path)], capture_output=True, text=True, check=True
                )
                current_byte_count = int(result.stdout.split()[0])

            bytes_added = current_byte_count - previous_byte_count

            current_secs = time.time()
            elapsed_secs = current_secs - previous_time
            byte_rate = bytes_added / elapsed_secs if bytes_added > 0 else 0

            logger.info(f"[bytes] Added: {bytes_added:,}")
            logger.info(f"[bytes] Total: {current_byte_count:,}")
            logger.info(f"[bytes] Rate: {byte_rate:,.2f} bytes/second")

            previous_byte_count = current_byte_count
            previous_time = time.time()
        except Exception as e:
            logger.error(f"Failed to monitor byte count: {e}")


def stop_byte_count_logger():
    global STOP_BYTE_COUNT_LOGGER
    STOP_BYTE_COUNT_LOGGER = True


def start_byte_count_logger(logger: logging.Logger, interval_seconds: int, file_path: Path):
    """
    Monitors the rate of bytes being added to a file, logging the number of bytes
    added per second over the interval.
    """

    assert not STOP_BYTE_COUNT_LOGGER, "A line count logger should not already be running"
    thread = threading.Thread(
        target=_log_byte_rate, args=(logger, interval_seconds, file_path), daemon=True
    )
    thread.start()
