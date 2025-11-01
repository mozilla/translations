from abc import ABC
from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    source_text: str
    ref_text: str
    domain: Optional[str]
    topic: Optional[str]


class Dataset(ABC):
    def download(self, src: str, trg: str):
        ...

    def get_texts(self) -> list[Segment]:
        ...


class Flores200Plus(Dataset):
    name = "flores200-plus-test"

    def download(self, src: str, trg: str):
        pass

    def get_texts(self) -> list[Segment]:
        pass


class Wmt24pp(Dataset):
    name = "wmt24pp"

    def download(self, src: str, trg: str):
        pass

    def get_texts(self) -> list[Segment]:
        pass


class Bouqet(Dataset):
    name = "bouqet"

    def download(self, src: str, trg: str):
        pass

    def get_texts(self) -> list[Segment]:
        pass
