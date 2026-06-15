"""
This file contains utilities for determining information about a language's script.
It will contain information about scripts we will probably never train, but it's at
least handy to have a fairly exhaustive list for future trainings. The information
was built by looking up scripts in CLDR data, and then manually documenting the
properties of that script from research on Wikipedia.

The main function is `get_script_info` which returns a `ScriptInfo`. This is backed by
getting the script from `icu`. `icu` is lazily loaded since it doesn't load on every
system.
"""

from enum import Enum
from typing import TypedDict


class ScriptType(Enum):
    # A script where letters represent both consonants and vowels, written in sequence.
    # Examples: Latin, Greek, Cyrillic, Armenian, Georgian
    ALPHABETIC = "alphabetic"

    # A script that primarily represents consonants. Vowels are typically omitted
    # and only optionally added via diacritical marks. For instance كِتاب (kitāb) with
    # diacritics vs كتاب without
    # Examples: Arabic, Hebrew
    ABJAD = "abjad"

    # A script where each character is a consonant-vowel unit. Vowel modifications
    # to the base unit are typically diacritical marks that modify a base consonant and
    # combine with it visually.
    #
    # For instance in Hindi क (ka) by itself implies /ka/, but adding the diacritic ि
    # it becomes कि = /ki/.
    #
    # Examples: Devanagari, Bengali, Tamil, Thai, Khmer, Ethiopic
    ABUGIDA = "abugida"

    # A script where each symbol represents an entire syllable.
    # Examples: Cherokee, Vai, Yi, Japanese Kana
    SYLLABARY = "syllabary"

    # A script where each symbol represents a word or morpheme, not individual sounds.
    # Examples: Chinese Han characters, Japanese Kanji
    LOGOGRAPHIC = "logographic"

    # A script where characters are built from shapes representing phonetic features.
    # Example: Hangul (Korean)
    FEATURAL = "featural"


class ScriptInfo(TypedDict):
    name: str
    type: ScriptType
    bicameral: bool


def is_script_phonemic(script_type: ScriptType):
    """
    Phonemic scripts are ones where the graphemes used, and their ordering directly
    relates to their the sounds they make (phonemes). This is in contrast to
    logographic scripts which use ideographs. One edge case is Hangul, which is featural.
    It is treated as logographic even though it's technically phonemic due to the way the
    characters are constructed from phonemic features (see #1101).
    """
    return script_type in {
        ScriptType.ALPHABETIC,
        ScriptType.ABJAD,
        ScriptType.ABUGIDA,
        ScriptType.SYLLABARY,
    }


def get_script_info(locale_str: str) -> ScriptInfo:
    """
    Take a BCP-47 locale code, and get the script info for it. This information is needed
    for knowing what type of augmentations can be performed on the data.
    """
    # If there is a defined default to bypass icu default return it
    if locale_str in default_scripts:
        return default_scripts[locale_str]

    # Load pyicu inline, as it only compiles on Linux.
    from icu import Locale  # type: ignore[reportAttributeAccessIssue]

    icu_locale = Locale(locale_str)
    maximized = Locale.addLikelySubtags(icu_locale)
    script_code = maximized.getScript()

    return scripts[script_code]


# This list was built from pulling CLDR data at an attempt to understand casing support.
# https://github.com/unicode-org/cldr/tree/main/common/casing
#
# There are many more scripts available, and if needed support can be added here.
# https://www.unicode.org/standard/supported.html
#
# To add a new script, look up information, generally from Wikipedia on the script
# type and whether it is bicameral (e.g. cased with an upper and lower case).
scripts: dict[str, ScriptInfo] = {
    "Adlm": {
        # 𞤀𞤤𞤳𞤵𞤤𞤫 𞤁𞤢𞤲𞤣𞤢𞤴𞤯𞤫 𞤂𞤫𞤻𞤮𞤤 𞤃𞤵𞤤𞤵𞤺𞤮𞤤
        "name": "Adlam",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Arab": {
        # العربية
        "name": "Arabic",
        "type": ScriptType.ABJAD,
        "bicameral": False,
    },
    "Armn": {
        # Հայերեն
        "name": "Armenian",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Beng": {
        # বাংলা
        "name": "Bengali",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Cans": {
        # ᓀᐦᐃᔭᐍᐏᐣ
        "name": "Canadian Aboriginal",
        "type": ScriptType.SYLLABARY,
        "bicameral": False,
    },
    "Cher": {
        # ᏣᎳᎩ
        "name": "Cherokee",
        "type": ScriptType.SYLLABARY,
        "bicameral": True,
    },
    "Cyrl": {
        # кириллица
        "name": "Cyrillic",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Deva": {
        # देवनागरी
        "name": "Devanagari",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Dsrt": {
        # 𐐓𐐲𐑌𐐮𐐻𐐰𐑉
        "name": "Deseret",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Ethi": {
        # አማርኛ
        "name": "Ethiopic",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Geor": {
        # ქართული
        "name": "Georgian",
        "type": ScriptType.ALPHABETIC,
        "bicameral": False,  # Generally unicameral, but has bicameral scripts.
    },
    "Grek": {
        # Ελληνικά
        "name": "Greek",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Gujr": {
        # ગુજરાતી
        "name": "Gujarati",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Guru": {
        # ਗੁਰਮੁਖੀ
        "name": "Gurmukhi",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Hang": {
        # 한국어 / 조선말
        "name": "Hangul",
        # Hangul is technically phonographic, as the character blocks encode phonemic
        # meaning by forming syllable blocks, rather than ideographic meaning, as in
        # Chinese.
        "type": ScriptType.FEATURAL,
        "bicameral": False,
    },
    "Hans": {
        # 简体字
        "name": "Han (Simplified)",
        "type": ScriptType.LOGOGRAPHIC,
        "bicameral": False,
    },
    "Hant": {
        # 繁體字
        "name": "Han (Traditional)",
        "type": ScriptType.LOGOGRAPHIC,
        "bicameral": False,
    },
    "Hebr": {
        # עברית
        "name": "Hebrew",
        "type": ScriptType.ABJAD,
        "bicameral": False,
    },
    "Jpan": {
        # 日本語
        "name": "Japanese",
        "type": ScriptType.LOGOGRAPHIC,
        "bicameral": False,
    },
    "Knda": {
        # ಕನ್ನಡ
        "name": "Kannada",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Khmr": {
        # ភាសាខ្មែរ
        "name": "Khmer",
        "type": ScriptType.ABUGIDA,
        # Khmr has subscript variations, but they encode semantic meanings, so probably
        # shouldn't be augmented for casing, as it would change the meaning of the text.
        "bicameral": False,
    },
    "Kore": {
        # 한국어 / 조선말
        "name": "Korean",
        # Kore is technically a mix of Hang and Hani
        "type": ScriptType.FEATURAL,
        "bicameral": False,
    },
    "Laoo": {
        # ລາວ
        "name": "Lao",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Latn": {
        # Latin
        "name": "Latin",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Mlym": {
        # മലയാളം
        "name": "Malayalam",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Mymr": {
        # မြန်မာစာ
        "name": "Myanmar",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Orya": {
        # ଓଡ଼ିଆ
        "name": "Odia",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Osge": {
        # 𐒰𐒻𐒼𐒰𐒿𐒷
        "name": "Osage",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Sinh": {
        # සිංහල අක්ෂර මාලාව
        "name": "Sinhala",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Taml": {
        # தமிழ்
        "name": "Tamil",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Telu": {
        # తెలుగు
        "name": "Telugu",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Tfng": {
        # ⵜⴰⵎⴰⵣⵉⵖⵜ
        "name": "Tifinagh",
        "type": ScriptType.ALPHABETIC,
        "bicameral": False,
    },
    "Thai": {
        # ภาษาไทย
        "name": "Thai",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Tibt": {
        # བོད་ཡིག
        "name": "Tibetan",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Vaii": {
        # ꕉꕜꕮ ꔔꘋ ꖸ ꔰ
        "name": "Vai",
        "type": ScriptType.SYLLABARY,
        "bicameral": False,
    },
    "Yiii": {
        # ꆈꌠꁱꂷ
        "name": "Yi",
        "type": ScriptType.SYLLABARY,
        "bicameral": False,
    },
}

default_scripts: dict[str, ScriptInfo] = {
    # ICU returns Kore by default for Korean, but Hang if kor_Hang is specified as input code
    # so here make it consistently return always Hangul
    "ko": scripts["Hang"],
    "kor": scripts["Hang"],
    "kor_Hang": scripts["Hang"],
}
