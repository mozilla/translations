from dataclasses import dataclass
from typing import Optional

from pipeline.eval.langs import (
    WMT24PP_DEFAULTS_MAP,
    FLORES_PLUS_DEFAULTS_MAP,
)
from datasets import load_dataset


@dataclass
class Segment:
    source_text: str
    ref_text: str
    domain: Optional[str]
    topic: Optional[str]


class Dataset:
    def __init__(self, src: str, trg: str):
        self.src = src
        self.trg = trg

    def download(self):
        ...

    def get_texts(self) -> list[Segment]:
        ...

    @staticmethod
    def supports_lang(src: str, trg: str) -> bool:
        ...


class Flores200Plus(Dataset):
    name = "flores200-plus"

    @staticmethod
    def supports_lang(src: str, trg: str) -> bool:
        return src in FLORES_PLUS_DEFAULTS_MAP and trg in FLORES_PLUS_DEFAULTS_MAP

    def download(self):
        self.src_ds = load_dataset(
            "openlanguagedata/flores_plus", FLORES_PLUS_DEFAULTS_MAP[self.src], split="test"
        )
        self.trg_ds = load_dataset(
            "openlanguagedata/flores_plus", FLORES_PLUS_DEFAULTS_MAP[self.trg], split="test"
        )

    def get_texts(self) -> list[Segment]:
        return [
            Segment(*item)
            for item in zip(
                self.src_ds["text"],
                self.trg_ds["text"],
                self.src_ds["domain"],
                self.src_ds["topic"],
            )
        ]


class Wmt24pp(Dataset):
    name = "wmt24pp"

    @staticmethod
    def supports_lang(src: str, trg: str) -> bool:
        return src in WMT24PP_DEFAULTS_MAP and trg in WMT24PP_DEFAULTS_MAP

    def download(self):
        lang = self.src if self.trg == "en" else self.trg
        lp = f"en-{WMT24PP_DEFAULTS_MAP[lang]}"
        # there is no separate test split
        ds = load_dataset("google/wmt24pp", lp, split="train")
        self.filtered = ds.filter(lambda ex: not ex["is_bad_source"] and ex["lp"] == lp)["train"]

    def get_texts(self) -> list[Segment]:
        source_texts, target_texts = (
            (self.filtered["source"], self.filtered["target"])
            if self.src == "en"
            else (self.filtered["target"], self.filtered["source"])
        )

        return [
            Segment(*item) for item in zip(source_texts, target_texts, self.filtered["domain"])
        ]


class Bouqet(Dataset):
    name = "bouqet"

    def download(self):
        import pandas as pd

        data = load_dataset("facebook/bouquet", "paragraph_level", split="test").to_pandas()

        self.df = pd.merge(
            data.loc[data["src_lang"].eq("spa_Latn")].drop(["tgt_lang", "tgt_text"], axis=1),
            data.loc[data["src_lang"].eq("rus_Cyrl"), ["src_lang", "src_text", "uniq_id"]].rename(
                {"src_lang": "tgt_lang", "src_text": "tgt_text"}, axis=1
            ),
            on="uniq_id",
        )

    def get_texts(self) -> list[Segment]:
        return [Segment(*item) for item in zip(self.df["src_text"], self.df["tgt_text"], "", "")]
