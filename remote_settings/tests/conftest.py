import signal
import shutil
import subprocess
import time
from dataclasses import dataclass

import pytest
import requests

from remote_settings import gcs

from .common import ATTACHMENTS_SOURCE_DIR, MODELS_DIR
from .gcs_fixtures import FakeGcs, build_gcs_fixture


@pytest.fixture(scope="module", autouse=True)
def local_remote_settings():
    """Starts the localhost RemoteSettings server for end-to-end tests."""

    cmd = ["poetry", "run", "python", "-m", "remote_settings", "local-server"]
    print(f"🚀 Launching local-server with: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def _forward():
        """Forward local-server output to the test logs."""
        for line in iter(proc.stdout.readline, ""):
            print(f"🌐 local-server | {line.rstrip()}")

    import threading, atexit

    threading.Thread(target=_forward, daemon=True).start()
    atexit.register(proc.terminate)

    # This server takes an indeterminate amount of time to start up, so ping the heartbeat
    # until we receive a 200 response that the server is ready. If it is not ready within
    # 60 seconds, we will fail the test cases.
    heartbeat = "http://localhost:8888/__heartbeat__"
    for i in range(60):
        try:
            r = requests.get(heartbeat, timeout=0.5)
            print(f"✅  Heartbeat check {i}: {r.status_code}")
            if r.status_code == 200:
                break
        except Exception as e:
            print(f"⏱️  Heartbeat check {i} failed: {e}")
        time.sleep(1)
    else:
        stdout, stderr = proc.communicate(timeout=5)
        proc.terminate()
        raise RuntimeError(
            f"❌  Local Remote Settings server failed to start in time.\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    yield

    print("\n🧹  Tearing down server...")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


@dataclass
class GcsArgs:
    """Minimal arguments for invoking the GCS downloader."""

    test: bool
    lang_pair: str
    architecture: str
    use_cached: bool = False


@pytest.fixture(scope="function", autouse=True)
def gcs_models(monkeypatch):
    """Populate the test models directory using the mocked GCS downloader."""
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)

    fixture = build_gcs_fixture()
    fake_gcs = FakeGcs(fixture)

    monkeypatch.setenv("REMOTE_SETTINGS_TEST_GCS", "1")
    monkeypatch.setattr(gcs, "_list_objects", fake_gcs.list_objects)
    monkeypatch.setattr(gcs, "_download_object", fake_gcs.download_object)
    monkeypatch.setattr(gcs, "_fetch_metadata", fake_gcs.fetch_metadata)

    base_dir = MODELS_DIR.as_posix()

    for spec in fixture.specs:
        args = GcsArgs(test=True, lang_pair=spec.lang_pair, architecture=spec.architecture)
        gcs.ensure_model_files(args, base_dir)

    emty_source = ATTACHMENTS_SOURCE_DIR / "tiny" / "emty"
    emty_dest = MODELS_DIR / "tiny" / "emty"
    emty_dest.mkdir(parents=True, exist_ok=True)
    for path in emty_source.iterdir():
        if path.is_file():
            shutil.copyfile(path, emty_dest / path.name)

    monkeypatch.undo()

    yield

    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
