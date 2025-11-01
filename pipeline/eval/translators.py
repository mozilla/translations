import os
import subprocess
import tempfile
import time
import urllib.request
import uuid
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import requests
import toolz
import yaml
from mtdata import iso
from tqdm import tqdm

from pipeline.common.downloads import location_exists


class LanguagePairNotSupported(Exception):
    def __init__(self, src, trg, translator):
        self.src = src
        self.trg = trg
        self.translator = translator


class Translator(ABC):
    name = None

    def __init__(self, src, trg):
        self.src = src
        self.trg = trg
        self.model_name = None

    def list_models(self) -> list[str]:
        ...

    def prepare(self, model_name: str):
        ...

    def transalate(self, texts: list[str]) -> list[str]:
        ...


def hf_model_exists(model_id: str) -> bool:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    hf_api = HfApi()

    try:
        hf_api.model_info(model_id)
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            return False
        else:
            raise

    return True


class GoogleTranslator(Translator):
    name = "google"

    def __init__(self, src, trg):
        super().__init__(src, trg)

    def list_models(self) -> list[str]:
        return ["v2"]

    def prepare(self, model_name: str):
        from google.cloud import translate_v2

        self.model_name = model_name
        self.translate_client = translate_v2.Client()

    def translate(self, texts: list[str]) -> list[str]:
        from google.api_core.exceptions import ServiceUnavailable

        def do_translate(partition):
            try:
                return self.translate_client.translate(
                    partition, target_language=self.trg, source_language=self.src
                )
            except ServiceUnavailable:
                return None

        results = []
        # decrease partition size if hitting limit of max 204800 bytes per request
        for partition in tqdm(list(toolz.partition_all(77, texts))):
            for _ in range(7):
                response = do_translate(partition)
                if response is not None:
                    break

                time.sleep(60)

            results += [r["translatedText"] for r in response]

        return results


class MicrosoftTranslator(Translator):
    name = "microsoft"

    def __init__(self, src, trg):
        super().__init__(src, trg)

        if self.src == "tl":
            self.src = "fil"
        elif self.trg == "tl":
            self.trg = "fil"

    def list_models(self) -> list[str]:
        return ["3.0"]

    def prepare(self, model_name: str):
        subscription_key = os.environ["AZURE_TRANSLATOR_KEY"]
        location = os.getenv("AZURE_LOCATION", "global")
        self.url = "https://api.cognitive.microsofttranslator.com/translate"
        self.headers = {
            "Ocp-Apim-Subscription-Key": subscription_key,
            "Ocp-Apim-Subscription-Region": location,
            "Content-type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4()),
        }
        self.model_name = model_name

    def translate(self, texts: list[str]) -> list[str]:
        params = {"api-version": self.model_name, "from": self.src, "to": [self.trg]}

        results = []
        # decrease partition size if hitting limit of max 10000 characters per request
        for partition in tqdm(list(toolz.partition_all(20, texts))):
            body = [{"text": text} for text in partition]
            response = requests.post(self.url, params=params, headers=self.headers, json=body)

            if response.status_code != 200:
                raise ValueError(
                    f"Incorrect response. code: {response.status_code} body: {response.json()}"
                )

            results += [r["translations"][0]["text"] for r in response.json()]

        return results


class NllbTranslator(Translator):
    name = "nllb"
    LANG_CODE_MAP = {
        "ar": "arb_Arab",
        "az": "azj_Latn",
        "fa": "pes_Arab",
        "lv": "lvs_Latn",
        "ms": "zsm_Latn",
        "zh": "zho_Hans",
    }

    def list_models(self) -> list[str]:
        return ["nllb-200-distilled-600M"]

    def prepare(self, model_name: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"facebook/{self.model_name}", src_lang=self.src
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"facebook/{self.model_name}").to(
            self.device
        )

        if self.trg in self.LANG_CODE_MAP:
            lang_code = self.LANG_CODE_MAP[self.trg]
        else:
            lang_code = None
            for lang in self.tokenizer.additional_special_tokens:
                if lang.startswith(iso.iso3_code(self.trg)):
                    assert (
                        lang_code is None
                    ), "Multiple NLLB language codes found for the same language ID, need to disambiguate!"
                    lang_code = lang
            assert lang_code is not None, f"Lang code for {self.trg} was not found"

        self.forced_bos_token_id = self.tokenizer.lang_code_to_id[lang_code]

    def translate(self, texts: list[str]) -> list[str]:
        results = []

        for partition in tqdm(list(toolz.partition_all(10, texts))):
            tokenized_src = self.tokenizer(partition, return_tensors="pt", padding=True).to(
                self.device
            )
            generated_tokens = self.model.generate(
                **tokenized_src, forced_bos_token_id=self.forced_bos_token_id
            )
            results += self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return results


class OpusmtTranslator(Translator):
    name = "opusmt"

    def _get_old_opusmt_hf_model_name(self):
        # todo: there are newer models available like Tatoeba ones
        return f"opus-mt-{self.src}-{self.trg}"

    def list_models(self) -> list[str]:
        model_name = self._get_old_opusmt_hf_model_name()
        if hf_model_exists(f"Helsinki-NLP/{model_name}"):
            return [model_name]

        return []

    def prepare(self, model_name: str):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/{self.model_name}")
        self.model = MarianMTModel.from_pretrained(f"Helsinki-NLP/{self.model_name}").to(
            self.device
        )

    def translate(self, texts: list[str]) -> list[str]:
        results = []

        for partition in tqdm(list(toolz.partition_all(10, texts))):
            tokenized_src = self.tokenizer(partition, return_tensors="pt", padding=True).to(
                self.device
            )
            generated_tokens = self.model.generate(**tokenized_src)
            results += self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return results


class ArgosTranslator(Translator):
    name = "argos"

    def list_models(self) -> list[str]:
        from argostranslate import package

        package.update_package_index()
        lang_packages = [
            p
            for p in package.get_available_packages()
            if p.from_code == self.src and p.to_code == self.trg
        ]
        if not lang_packages:
            return []

        return [str(lang_packages[0].package_version)]

    def prepare(self, model_name: str):
        os.environ["ARGOS_DEVICE_TYPE"] = "cuda"
        from argostranslate import package
        from argostranslate import settings

        package_to_install = [
            p
            for p in package.get_available_packages()
            if str(p.package_version) == model_name
            and p.from_code == self.src
            and p.to_code == self.trg
        ][0]
        package.install_from_path(package_to_install.download())
        self.model_name = model_name
        assert settings.device == "cuda"

    def translate(self, texts: list[str]) -> list[str]:
        from argostranslate import translate

        return [translate.translate(text, self.src, self.trg) for text in tqdm(texts)]


@dataclass
class BergamotModel:
    src: str
    trg: str
    name: str

    @staticmethod
    def from_path(gcs_path):
        parts = gcs_path.split("/")
        src, trg = parts[1].split("-")
        return BergamotModel(src, trg, parts[2])


class BergamotTranslator(Translator):
    name = "bergamot"

    def __init__(self, src: str, trg: str, bucket: str, translator_cli_path: str):
        super().__init__(src, trg)
        self.bucket = bucket
        # download model from GCS
        self.translator_cli_path = translator_cli_path

    @staticmethod
    def list_all_models(bucket: str, src: str = None, trg: str = None) -> list[BergamotModel]:
        # look for objects like models/en-uk/spring-2024_J4QWVDPJQdOGbUB0xfzj1Q/exported/lex.50.50.enuk.s2t.bin.gz
        # spring-2024_J4QWVDPJQdOGbUB0xfzj1Q would be the name of the model
        prefix = f"models/{src}-{trg}" if src and trg else "models/"
        models = {
            BergamotModel.from_path(item["name"])
            for item in requests.get(
                f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={prefix}"
            ).json()["items"]
            if "/exported/" in item["name"]
        }

        return list(models)

    def list_models(self) -> list[str]:
        return [m.name for m in self.list_all_models(self.bucket, src=self.src, trg=self.trg)]

    def prepare(self, model_name: str):
        shortlist = f"lex.50.50.{self.src}{self.trg}.s2t.bin.gz"
        model = f"model.{self.src}{self.trg}.intgemm.alphas.bin.gz"
        joint_vocab = f"vocab.{self.src}{self.trg}.spm.gz"
        src_vocab = f"srcvocab.{self.src}{self.trg}.spm.gz"
        trg_vocab = f"trgvocab.{self.src}{self.trg}.spm.gz"

        gcs_path = f"https://storage.googleapis.com/{self.bucket}/models/{self.src}-{self.trg}/{model_name}/exported/"
        tmp_dir = tempfile.gettempdir()
        download_dir = Path(f"{tmp_dir}/models/{self.src}-{self.trg}/{model_name}")
        os.makedirs(download_dir, exist_ok=True)

        # download shortlist and model
        shortlist_path = download_dir / shortlist
        model_path = download_dir / model
        urllib.request.urlretrieve(gcs_path + shortlist, shortlist_path)
        urllib.request.urlretrieve(gcs_path + model, model_path)

        # download vocabs
        joint_vocab_url = gcs_path + joint_vocab
        if location_exists(joint_vocab_url):
            join_vocab_path = download_dir / joint_vocab
            urllib.request.urlretrieve(joint_vocab_url, join_vocab_path)
            vocabs = [join_vocab_path, join_vocab_path]
        else:
            src_vocab_path = download_dir / src_vocab
            urllib.request.urlretrieve(gcs_path + src_vocab, src_vocab_path)
            trg_vocab_path = download_dir / trg_vocab
            urllib.request.urlretrieve(gcs_path + trg_vocab, trg_vocab_path)
            vocabs = [src_vocab_path, trg_vocab_path]

        # the config should be the same as on inference
        yaml_config = {
            "bergamot-mode": "wasm",
            "models": [model_path],
            "vocabs": vocabs,
            "shortlist": [shortlist_path, False],
            "beam-size": 1,
            "normalize": 1.0,
            "word-penalty": 0,
            "max-length-break": 128,
            "mini-batch-words": 1024,
            "workspace": 128,
            "max-length-factor": 2.0,
            "skip-cost": True,
            "cpu-threads": 0,
            "quiet": True,
            "quiet-translation": True,
            "gemm-precision": "int8shiftAlphaAll",
            "alignment": "soft",
        }

        self.config_path = download_dir / "config.yaml"
        with open(self.config_path, "w") as f:
            yaml.dump(yaml_config, f)

    def translate(self, texts: list[str]) -> list[str]:
        cmd = [
            self.translator_cli_path,
            "--model-config-paths",
            self.config_path,
            "--log-level",
            "info",
        ]
        translations = subprocess.check_output(cmd, input="\n".join(texts)).decode("utf-8")
        return translations.split("\n")


# todo: we can test a few popular pairs directly
# class PivotBergamotTranslator:
