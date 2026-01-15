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

# TODO: implement convertions required for various tools and defaults for backward compatibility with ISO-639-1
# TODO: see https://github.com/mozilla/translations/issues/1139

import icu

from pipeline.langs.maps import ISO6393_DEFAULTS_MAP, ISO6393_DEFAULTS_REVERSED_MAP


def to_iso6391(lang: str) -> str:
    """
    Converts language in ISO-693-1<_optional_script> format to ISO-693-1

    For example, zh_hant -> zh
    """
    return icu.Locale(lang).getLanguage()


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
