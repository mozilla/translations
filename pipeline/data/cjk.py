"""
Chinese, Japanese, Korean (CJK) specific data importing code
"""

from enum import Flag
from pathlib import Path
from typing import Optional, Union

import hanzidentifier
import opencc

from pipeline.common.datasets import Statistics
from pipeline.common.downloads import read_lines, write_lines


CJK_LANGS = ["zh", "ja", "ko"]


class ChineseType(Flag):
    none = 0
    simplified = 1
    traditional = 2


class ConversionStep(Statistics):
    """
    When converting data, count how many sentences were converted, and how many were visited.
    """

    def __init__(self, description: str, converted=0, dataset_path: Optional[Path] = None) -> None:
        super().__init__(dataset_path)
        self.description = description
        self.converted = converted
        self.visited = 0


class DatasetStatistics(Statistics):
    def __init__(self, dataset_path: Union[str, Path], script: ChineseType) -> None:
        super().__init__(dataset_path)
        self.script = script
        self.script_conversion = ConversionStep(
            f"How many sentences in the dataset were converted to {script.name}",
        )


class ChineseConverter:
    def __init__(self):
        self.s2t = opencc.OpenCC("s2t.json")
        self.t2s = opencc.OpenCC("t2s.json")

    def convert_file(
        self, input_path: Union[str, Path], output_path: Union[str, Path], to: ChineseType
    ) -> DatasetStatistics:
        stats = DatasetStatistics(output_path, to)
        with write_lines(output_path) as out_file, read_lines(input_path) as lines:
            for line in lines:
                stats.script_conversion.visited += 1
                ch_type = self._detect(line)
                if ch_type in (ch_type.none, to):
                    new_line = line
                else:
                    new_line = self._convert_line(line, to)
                    stats.script_conversion.converted += 1
                out_file.write(new_line)
        return stats

    @staticmethod
    def _detect(text) -> ChineseType:
        res = hanzidentifier.identify(text)
        if res == hanzidentifier.SIMPLIFIED:
            return ChineseType.simplified
        if res == hanzidentifier.TRADITIONAL:
            return ChineseType.traditional
        if res in (hanzidentifier.BOTH, hanzidentifier.MIXED):
            return ChineseType.traditional | ChineseType.simplified
        return ChineseType.none

    def _convert_line(self, text: str, to: ChineseType) -> str:
        if to == ChineseType.simplified:
            return self.t2s.convert(text)
        elif to == ChineseType.traditional:
            return self.s2t.convert(text)
        raise ValueError(f"Unsupported type: {to}")
