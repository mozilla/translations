"""
Utilities to convert between common language code schemes.

The universal language code we use in the training config is:
    <ISO 639-1>[_<ISO-15924 script or BCB-47 region>]
It should be parsable by ICU.

Examples:
    en
    ru
    nb (can be converted to no if any tool requires it)
    sq (same, ISO 639-3 als)
    zh (ISO 639-3 default cmn, script default Hans)
    zh_hant (ISO 639-3 macro language `zho`, ISO 639-3 individual default `cmn`)
    sr_cyrl (ICU `sr_Cyrl_RS`)
    pt_br (ICU locale `pt_Latn_BR`, ISO 639-3 `por`)
    es_valencia (ICU `es_Latn_ES_VALENCIA`)

Some tools and datasets require further convertion into their corresponding language code schemes.

ISO 639-1 two-letter codes (example: zh, ru, en)
ISO 639-3 three-letter codes (example: cmn, rus, eng)
ISO 639-3 + ISO-15924 script (example: cmn_Hant, rus_Cyrl, eng_Latn)
BCP-47 locales (example: ru-RU, zh-TW, en-CA, pt-BR, zh-Hans, es-ES-valencia)
ISO 639-3 + ISO-15924 + Glottocodes (example: por_Latn_braz1246 for Brazilian Portuguese)
"""
import json
from typing import Optional, Union, Container, Iterable

import icu

from pipeline.langs.maps import (
    ISO6393_DEFAULTS_MAP,
    ISO6393_DEFAULTS_REVERSED_MAP,
    PONTOON_DEFAULTS_BCP_MAP,
    PONTOON_LANGUAGES,
    FLORES_101_DEFAULTS_MAP,
    FLORES_PLUS_DEFAULTS_MAP,
    BICLEANER_AI_DEFAULTS_MAP,
    BOUQUET_DEFAULTS_MAP,
    GOOGLE_LANGS,
    NLLB_DEFAULTS_MAP,
    COMET22_SUPPORT,
    PIPELINE_SUPPORT,
    WMT24PP_LANGS,
    METRICX24_LANGS,
    COMMON_FALLBACKS,
    MICROSOFT_LANGS,
    FLORES_101_LANGUAGES,
)
from pipeline.langs.scripts import get_script_info, ScriptInfo, is_script_phonemic


def to_iso6391(lang: str, default_map: dict[str, str] = None) -> str:
    """
    Converts language in ISO-693-1<_optional_script> format to ISO-693-1

    For example, zh_hant -> zh
    """
    if default_map and lang in default_map:
        return default_map[lang]

    return icu.Locale(lang).getLanguage()


def icu_normalize(lang: str):
    """
    Normalizes ISO-693-1<_optional_script> to ICU/CLDR <iso_lang>[_<Script>][_<REGION>].

    For example: en -> en, zh -> zh, zh_hant -> zh_Hant, sr_cyrl -> sr_Cyrl, pr_br -> pr_BR
    """
    return str(icu.Locale(lang))


def to_iso6393(lang: str, default_map: dict[str, str] = None) -> str:
    """
    Converts language in ISO-693-1<_optional_script> format to ISO-693-3

    For example, zh_hant -> zho, ru -> rus
    """
    if default_map and lang in default_map:
        return default_map[lang]

    return icu.Locale(lang).getISO3Language()


def to_iso6393_individual_and_script(lang: str) -> str:
    """
    Converts language in ISO-693-1<_optional_script> format to the format <ISO-693-3>_<ISO-15924(script)>
        with an individual language code (unlike the macro language one), e.g. `cmn` instead of `zho`.

    For example, ru -> rus_Cyrl, zh -> cmn_Hans, zh_hant -> cmn_Hant
    """

    if lang in ISO6393_DEFAULTS_MAP:
        return ISO6393_DEFAULTS_MAP[lang]

    locale = icu.Locale(lang)
    # add default script
    locale = icu.Locale.addLikelySubtags(locale)
    local_and_script = f"{locale.getISO3Language()}_{locale.getScript()}"

    return local_and_script


def to_locale(lang: str) -> str:
    """
    Converts language in ISO-693-1<_optional_script> format to BCP-47 region specific locale in Unicode format

    For example, zh_hant -> zh_TW, zh_hans -> zh_CN. The default region is added based on ICU/CLDR rules.
    """
    locale = icu.Locale(lang).addLikelySubtags()
    return f"{locale.getLanguage()}_{locale.getCountry()}"


def iso6393_and_script_to_lang_id(lang: str) -> str:
    """
    Converts language in ISO-693-3 3-letter format and script to the pipeline language identifier

    For example, cmn_Hans -> zh, cmn_Hant -> zh_hant, eng -> en, zho -> zh
    """
    # ICU does not convert macro language to an individual one
    if lang in ISO6393_DEFAULTS_REVERSED_MAP:
        return ISO6393_DEFAULTS_REVERSED_MAP[lang]

    # ICU normalizes the locale to ISO-639-1 where there are default rules
    return icu.Locale(lang).getLanguage()


class LanguageNotSupported(Exception):
    def __init__(self, lang):
        self.lang = lang


class LangCode(str):
    """
    Training pipeline language code label.

    Format: <ISO 639-1>[_<ISO-15924 script or BCB-47 region>]
    """

    def __new__(cls, value: str):
        if not value.islower():
            raise ValueError(f"LangCode must be lowercase: '{value}'")
        if not all(c.isalnum() or c == "_" for c in value):
            raise ValueError(
                f"LangCode must contain only letters, numbers, and underscores: '{value}'"
            )
        return super().__new__(cls, value)

    def script(self) -> Optional[ScriptInfo]:
        return get_script_info(self)

    def is_script_phonemic(self) -> bool:
        script = self.script()
        if not script:
            raise LanguageNotSupported(self)

        return is_script_phonemic(script["type"])

    def is_script_bicameral(self) -> bool:
        script = self.script()
        if not script:
            raise LanguageNotSupported(self)

        return script["bicameral"]

    def is_cjk(self) -> bool:
        return to_iso6391(self) in {"zh", "ja", "ko"}

    def is_chinese(self) -> bool:
        return self.startswith("zh")

    def is_chinese_traditional(self) -> bool:
        return str(self) == "zh_hant"

    def _find_code(self, supported_langs: Union[Container, Iterable], check_script=False):
        lang = str(self)

        if lang in supported_langs:
            return lang

        if lang in COMMON_FALLBACKS:
            for fallback in COMMON_FALLBACKS[lang]:
                if fallback in supported_langs:
                    return fallback

        iso6393_is = to_iso6393_individual_and_script(self)
        iso6393 = to_iso6393(self)
        iso6391 = to_iso6391(self)

        # from more specific to less specific
        for code in (iso6393_is, iso6393, iso6391):
            if code in supported_langs:
                if not check_script or get_script_info(code) == self.script():
                    return code

        raise LanguageNotSupported(self)

    # Datasets

    def opus(self) -> str:
        # zh_hant -> zh
        return to_iso6391(self)

    def mtdata(self) -> str:
        # zh_hant -> zho
        return to_iso6393(self)

    def sacrebleu(self) -> str:
        # zh_hant -> zh
        return to_iso6391(self)

    def flores101(self) -> str:
        # zh_hant -> zho_trad
        lang = str(self)
        if lang in FLORES_101_DEFAULTS_MAP:
            return FLORES_101_DEFAULTS_MAP[lang]
        return self._find_code(FLORES_101_LANGUAGES, check_script=True)

    def pontoon(self) -> str:
        # zh_hant -> zh-TW
        lang = str(self)
        if lang in PONTOON_DEFAULTS_BCP_MAP:
            return PONTOON_DEFAULTS_BCP_MAP[lang]

        return self._find_code(PONTOON_LANGUAGES)

    def hplt(self) -> str:
        # zh_hant -> cmn_Hant
        return to_iso6393_individual_and_script(self)

    def newscrawl(self) -> str:
        # zh_hant -> zh
        return to_iso6391(self)

    # Cleaning

    def opuscleaner(self) -> str:
        # zh_hant -> zh_Hant
        return icu_normalize(self)

    def bicleaner(self) -> str:
        # zh_hant -> zh, sr_cyrl -> hbs
        return to_iso6391(self, BICLEANER_AI_DEFAULTS_MAP)

    def monocleaner(self) -> str:
        # zh_hant -> zh
        return to_iso6391(self)

    def fasttext(self):
        # zh_hant -> cmn_Hant for openlid/nllb models
        return to_iso6393_individual_and_script(self)

    @staticmethod
    def from_fasttext(lang: str):
        # cmn_Hant -> zh_hant, cmn_Hans -> zh, eng_Latn -> en etc.
        return LangCode(iso6393_and_script_to_lang_id(lang))

    # Metrics

    def comet22(self):
        # zh_hant -> zh
        return self._find_code(COMET22_SUPPORT)

    def metricx24(self):
        # zh_hant -> zh
        return self._find_code(METRICX24_LANGS)

    # Final evals datasets

    def flores200plus(self):
        # zh_hant -> cmn_Hant
        lang = str(self)
        if lang in FLORES_PLUS_DEFAULTS_MAP:
            return FLORES_PLUS_DEFAULTS_MAP[lang]

        return self._find_code(list(FLORES_PLUS_DEFAULTS_MAP.values()))

    def bouquet(self):
        # pt -> por_Latn_braz1246
        return self._find_code(list(BOUQUET_DEFAULTS_MAP.values()), check_script=True)

    def wmt24pp(self):
        # zh_hant -> zh_TW
        locale = to_locale(self)
        if locale in WMT24PP_LANGS:
            return locale

        return self._find_code(WMT24PP_LANGS)

    # Models and APIs

    def nllb(self):
        # zh_hant -> zh_TW
        lang = str(self)
        if lang in NLLB_DEFAULTS_MAP:
            return NLLB_DEFAULTS_MAP[lang]

        iso6391 = to_iso6391(lang)
        if iso6391 in FLORES_PLUS_DEFAULTS_MAP:
            return FLORES_PLUS_DEFAULTS_MAP[iso6391]

        return self._find_code(list(FLORES_PLUS_DEFAULTS_MAP.values()))

    def google(self):
        # zh_hant -> zh-TW
        return self._find_code(GOOGLE_LANGS)

    def microsoft(self):
        # zh_hant -> zh_TW
        return self._find_code(MICROSOFT_LANGS)


def generate_all(save_path: str = None) -> dict[str, dict[str, str]]:
    """
    Generate a JSON with all language codes mappings for various tools we use in the pipeline.
    """
    all = {}
    # Use Flores200-plus as a starting point to generate the mapping
    for lang in sorted(PIPELINE_SUPPORT):
        lang_code = LangCode(lang)

        def wrap(func):
            try:
                return func()
            except LanguageNotSupported:
                return "not supported"

        all[lang] = {
            "name": icu.Locale(lang_code).getDisplayName(),
            "script": lang_code.script()["name"] if lang_code.script() else "not supported",
            "opus": wrap(lambda: lang_code.opus()),
            "mtdata": wrap(lambda: lang_code.mtdata()),
            "sacrebleu": wrap(lambda: lang_code.sacrebleu()),
            "flores101": wrap(lambda: lang_code.flores101()),
            "pontoon": wrap(lambda: lang_code.pontoon()),
            "hplt": wrap(lambda: lang_code.hplt()),
            "newscrawl": wrap(lambda: lang_code.newscrawl()),
            "opuscleaner": wrap(lambda: lang_code.opuscleaner()),
            "bicleaner": wrap(lambda: lang_code.bicleaner()),
            "monocleaner": wrap(lambda: lang_code.monocleaner()),
            "fasttext": wrap(lambda: lang_code.fasttext()),
            "comet22": wrap(lambda: lang_code.comet22()),
            "metricx24": wrap(lambda: lang_code.metricx24()),
            "flores200-plus": wrap(lambda: lang_code.flores200plus()),
            "bouquet": wrap(lambda: lang_code.bouquet()),
            "wmt24pp": wrap(lambda: lang_code.wmt24pp()),
            "nllb": wrap(lambda: lang_code.nllb()),
            "google": wrap(lambda: lang_code.google()),
            "microsoft": wrap(lambda: lang_code.microsoft()),
        }

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all, f, indent=4)
            print(f"The mapping is saved to {save_path}")

    return all
