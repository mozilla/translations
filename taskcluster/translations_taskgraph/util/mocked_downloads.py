"""
Allow tests to mock downloads.
"""

import os
from typing import Optional

mocked_downloads = None


def mock_taskcluster_downloads(downloads: Optional[dict[str, str]]):
    global mocked_downloads
    mocked_downloads = downloads


def get_mocked_downloads_file_path(url: str) -> Optional[str]:
    """If there is a mocked download, get the path to the file, otherwise return None"""
    if not mocked_downloads:
        return None

    # Taskcluster code generally suppressed output, but here we are in a test context
    # where stdout is helpful.
    import sys

    blocked_stdout = sys.stdout
    sys.stdout = sys.__stdout__

    source_file = mocked_downloads.get(url)
    if not source_file:
        print("mocked_downloads:", mocked_downloads)
        raise Exception(f"Received a URL that was not in MOCKED_DOWNLOADS {url}")

    if not os.path.exists(source_file):
        raise Exception(f"The source file specified did not exist {source_file}")

    print("Mocking a download.")
    print(f"   url: {url}")
    print(f"  file: {source_file}")

    sys.stdout = blocked_stdout

    return source_file
