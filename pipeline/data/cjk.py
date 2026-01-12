"""
Chinese, Japanese, Korean (CJK) specific data importing code
"""
from enum import Flag
from pathlib import Path
from typing import Optional

import hanzidentifier
import opencc

from pipeline.common.datasets import Statistics
from pipeline.common.downloads import read_lines, write_lines
from pipeline.common.logging import get_logger

logger = get_logger(__file__)


class ChineseType(Flag):
    none = 0
    simplified = 1
    traditional = 2


class ConversionStep(Statistics):
    """
    When converting data, count how many sentences were converted, and how many were visited.
    """

    def __init__(
        self, description: str, converted=0, filtered=0, dataset_path: Optional[Path] = None
    ) -> None:
        super().__init__(dataset_path)
        self.description = description
        self.converted = converted
        self.filtered = filtered
        self.visited = 0


class DatasetStatistics(Statistics):
    def __init__(self, dataset_path: Path, script: ChineseType) -> None:
        super().__init__(dataset_path)
        self.script = script
        self.script_conversion = ConversionStep(
            f"How many sentences in the dataset were converted to {script.name} or filtered",
        )


class ChineseConverter:
    def __init__(self):
        self.s2t = opencc.OpenCC("s2t.json")
        self.t2s = opencc.OpenCC("t2s.json")

    def convert_file(
        self, input_path: Path, output_path: Path, to: ChineseType
    ) -> DatasetStatistics:
        """
        Convert all lines to one variant of Chinese
        """
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

    def filter_file(self, input_path: Path, output_path: Path, variant: ChineseType):
        """
        Filter everything except the specified variant of Chinese
        """
        stats = DatasetStatistics(output_path, variant)
        with write_lines(output_path) as out_file, read_lines(input_path) as lines:
            for line in lines:
                stats.script_conversion.visited += 1
                ch_type = self._detect(line)
                if ch_type == variant:
                    out_file.write(line)
                else:
                    stats.script_conversion.filtered += 1

        return stats

    def filter_parallel_corpus(
        self,
        zh_path: Path,
        other_path: Path,
        zh_output_path: Path,
        other_output_path: Path,
        variant: ChineseType,
    ):
        """
        Filter everything except the specified variant of Chinese in a parallel corpus
        """
        stats = DatasetStatistics(zh_output_path, variant)
        with (
            write_lines(zh_output_path) as zh_out_file,
            write_lines(other_output_path) as other_out_file,
            read_lines(zh_path) as zh_lines,
            read_lines(other_path) as other_lines,
        ):
            for zh_line, other_line in zip(zh_lines, other_lines):
                stats.script_conversion.visited += 1
                ch_type = self._detect(zh_line)
                if ch_type == variant:
                    zh_out_file.write(zh_line)
                    other_out_file.write(other_line)
                else:
                    stats.script_conversion.filtered += 1

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


def handle_chinese_mono(
    file_destination: Path, is_src: bool, language_code: str, variant: ChineseType = None
):
    if not language_code.startswith("zh"):
        raise ValueError("Run only for Chinese")
    if not variant:
        variant = ChineseType.traditional if language_code == "zh_hant" else ChineseType.simplified

    converted_path = file_destination.with_suffix(".converted.zst")
    chinese_converter = ChineseConverter()
    if is_src:
        logger.info(f"Converting the output file to {variant}")
        stats = chinese_converter.convert_file(file_destination, converted_path, variant)
    else:
        logger.info(f"Filtering out everything except {variant} in the output file")
        stats = chinese_converter.filter_file(file_destination, converted_path, variant)
    converted_path.replace(file_destination)
    print(
        f"Converted {stats.script_conversion.converted}, Filtered: {stats.script_conversion.filtered} Visited: {stats.script_conversion.visited}"
    )
    stats.save_json()


def handle_chinese_parallel(output_prefix: str, src: str, trg: str, variant: ChineseType = None):
    if not src.startswith("zh") and not trg.startswith("zh"):
        raise ValueError("Run only for Chinese")

    if not variant:
        zh_code = src if src.startswith("zh") else trg
        variant = ChineseType.traditional if zh_code == "zh_hant" else ChineseType.simplified

    chinese_converter = ChineseConverter()
    is_src = src.startswith("zh")
    if is_src:
        logger.info(f"Converting the output file to {variant}")
        input_path = Path(f"{output_prefix}.{src}.zst")
        converted_path = Path(f"{output_prefix}.converted.{src}.zst")
        stats = chinese_converter.convert_file(
            input_path=input_path,
            output_path=converted_path,
            to=variant,
        )
        converted_path.replace(input_path)
    else:
        logger.info(f"Filtering out everything except {variant} from a parallel corpus")
        trg_path = Path(f"{output_prefix}.{trg}.zst")
        src_path = Path(f"{output_prefix}.{src}.zst")
        trg_filtered_path = Path(f"{output_prefix}.filtered.{trg}.zst")
        src_filtered_path = Path(f"{output_prefix}.filtered.{src}.zst")
        stats = chinese_converter.filter_parallel_corpus(
            zh_path=trg_path,
            other_path=src_path,
            zh_output_path=trg_filtered_path,
            other_output_path=src_filtered_path,
            variant=variant,
        )
        src_filtered_path.replace(src_path)
        trg_filtered_path.replace(trg_path)
    print(
        f"Converted {stats.script_conversion.converted}, Filtered: {stats.script_conversion.filtered} Visited: {stats.script_conversion.visited}"
    )
    stats.save_json()
