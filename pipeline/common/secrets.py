import json
import os
import tempfile

from pipeline.common.logging import get_logger

import taskcluster

logger = get_logger(__file__)


class Secrets:
    def __init__(self):
        root_url = os.environ.get("TASKCLUSTER_PROXY_URL")
        assert root_url, "When running in Taskcluster the TASKCLUSTER_PROXY_URL must be set."
        self.secrets = taskcluster.Secrets({"rootUrl": root_url})

    def prepare_key_hf(self):
        logger.info("Reading HF secret from Taskcluster")
        os.environ["HF_TOKEN"] = self.read_key("huggingface")["token"]

    def prepare_keys(self):
        logger.info("Reading secrets from Taskcluster")
        os.environ["HF_TOKEN"] = self.read_key("huggingface")["token"]
        os.environ["OPENAI_API_KEY"] = self.read_key("chatgpt")["token"]
        os.environ["AZURE_TRANSLATOR_KEY"] = self.read_key("azure-translate")["token"]
        google_key_file = tempfile.NamedTemporaryFile("w", delete=False)
        with google_key_file:
            json.dump(self.read_key("google-translate"), google_key_file)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_key_file.name

    def read_key(self, name: str) -> dict:
        try:
            response = self.secrets.get(f"project/translations/level-1/{name}")
            return response["secret"]
        except Exception as e:
            raise ValueError(f"Could not retrieve the secret key {name}: {e}")
