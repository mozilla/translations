from enum import Flag

import hanzidentifier
import opencc

from pipeline.common.downloads import read_lines, write_lines


class ChineseType(Flag):
    none = 0
    simplified = 1
    traditional = 2


class ChineseConverter:
    def __init__(self):
        self.s2t = opencc.OpenCC("s2t.json")
        self.t2s = opencc.OpenCC("t2s.json")

    def convert_file(self, input_path: str, output_path: str, to: ChineseType) -> int:
        count = 0
        with write_lines(output_path) as out_file, read_lines(input_path) as lines:
            for line in lines:
                # remove new line
                ch_type = self._detect(line)
                if ch_type not in (ch_type.none, to):
                    new_line = self._convert_line(line, to)
                    count += 1
                else:
                    new_line = line
                out_file.write(new_line)
        return count

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
