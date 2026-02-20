# Based on XLM-R
# from the repo + codes https://github.com/Unbabel/COMET/tree/master?tab=readme-ov-file#languages-covered
COMET22_SUPPORT = [
    # Afrikaans
    "af",
    # Amharic
    "am",
    # Arabic
    "ar",
    # Assamese
    "as",
    # Azerbaijani
    "az",
    # Belarusian
    "be",
    # Bulgarian
    "bg",
    # Bengali
    "bn",
    # Bengali Romanized
    "bn",
    # Breton
    "br",
    # Bosnian
    "bs",
    # Catalan
    "ca",
    # Czech
    "cs",
    # Welsh
    "cy",
    # Danish
    "da",
    # German
    "de",
    # Greek
    "el",
    # English
    "en",
    # Esperanto
    "eo",
    # Spanish
    "es",
    # Estonian
    "et",
    # Basque
    "eu",
    # Persian
    "fa",
    # Finnish
    "fi",
    # Filipino
    "fil",
    # French
    "fr",
    # Western Frisian
    "fy",
    # Irish
    "ga",
    # Scottish Gaelic
    "gd",
    # Galician
    "gl",
    # Gujarati
    "gu",
    # Hausa
    "ha",
    # Serbo-Croatian, a macro code not explictly supported
    # but almos all of its individuals are supported (bos, hrv, srp)
    "hbs",
    # Hebrew
    "he",
    # Hindi
    "hi",
    # Hindi Romanized
    "hi",
    # Croatian
    "hr",
    # Hungarian
    "hu",
    # Armenian
    "hy",
    # Indonesian
    "id",
    # Icelandic
    "is",
    # Italian
    "it",
    # Japanese
    "ja",
    # Javanese
    "jv",
    # Georgian
    "ka",
    # Kazakh
    "kk",
    # Khmer
    "km",
    # Kurdish (Kurmanji)
    "kmr",
    # Kannada
    "kn",
    # Korean
    "ko",
    # Kyrgyz
    "ky",
    # Latin
    "la",
    # Lao
    "lo",
    # Lithuanian
    "lt",
    # Latvian
    "lv",
    # Malagasy
    "mg",
    # Macedonian
    "mk",
    # Malayalam
    "ml",
    # Mongolian
    "mn",
    # Marathi
    "mr",
    # Malay
    "ms",
    # Burmese
    "my",
    # Nepali
    "ne",
    # Dutch
    "nl",
    # Norwegian
    "no",
    # Oromo
    "om",
    # Oriya (Odia)
    "or",
    # Punjabi
    "pa",
    # Polish
    "pl",
    # Pashto
    "ps",
    # Portuguese
    "pt",
    # Romanian
    "ro",
    # Russian
    "ru",
    # Sanskrit
    "sa",
    # Sindhi
    "sd",
    # Sinhala
    "si",
    # Slovak
    "sk",
    # Slovenian
    "sl",
    # Somali
    "so",
    # Albanian
    "sq",
    # Serbian
    "sr",
    # Sundanese
    "su",
    # Swedish
    "sv",
    # Swahili
    "sw",
    # Tamil
    "ta",
    # Tamil Romanized
    "ta",
    # Telugu
    "te",
    # Telugu Romanized
    "te",
    # Thai
    "th",
    # Turkish
    "tr",
    # Uyghur
    "ug",
    # Ukrainian
    "uk",
    # Urdu
    "ur",
    # Urdu Romanized
    "ur",
    # Uzbek
    "uz",
    # Vietnamese
    "vi",
    # Xhosa
    "xh",
    # Yiddish
    "yi",
    # Chinese
    "zh",
]

# Based on T5
# from https://github.com/google-research/multilingual-t5?tab=readme-ov-file#languages-covered
# https://huggingface.co/datasets/allenai/c4
METRICX24_LANGS = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bg-Latn": "Bulgarian (Latin)",
    "bn": "Bangla",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "co": "Corsican",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "el-Latn": "Greek (Latin)",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "hi": "Hindi",
    "hi-Latn": "Hindi (Latin script)",
    "hmn": "Hmong, Mong",
    "ht": "Haitian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "iw": "former Hebrew",
    "ja": "Japanese",
    "ja-Latn": "Japanese (Latin)",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kyrgyz",
    "la": "Latin",
    "lb": "Luxembourgish",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "ny": "Nyanja",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "ru-Latn": "Russian (Latin)",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sm": "Samoan",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "st": "Southern Sotho",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
    "zh-Latn": "Chinese (Latin)",
    "zu": "Zulu",
}

METRICX_DEFAULTS_MAP = {"bs": "sr", "hr": "sr", "hbs": "sr"}

# https://huggingface.co/datasets/google/wmt24pp
WMT24PP_LANGS = {
    "en_US": {"lang": "English", "country": "United States"},
    "ar_EG": {"lang": "Arabic", "country": "Egypt"},
    "ar_SA": {"lang": "Arabic", "country": "Saudi Arabia"},
    "bg_BG": {"lang": "Bulgarian", "country": "Bulgaria"},
    "bn_IN": {"lang": "Bengali", "country": "India"},
    "ca_ES": {"lang": "Catalan", "country": "Spain"},
    "cs_CZ": {"lang": "Czech", "country": "Czechia"},
    "da_DK": {"lang": "Danish", "country": "Denmark"},
    "de_DE": {"lang": "German", "country": "Germany"},
    "el_GR": {"lang": "Greek", "country": "Greece"},
    "es_MX": {"lang": "Spanish", "country": "Mexico"},
    "et_EE": {"lang": "Estonian", "country": "Estonia"},
    "fa_IR": {"lang": "Farsi", "country": "Iran"},
    "fi_FI": {"lang": "Finnish", "country": "Finland"},
    "fil_PH": {"lang": "Filipino", "country": "Philippines"},
    "fr_CA": {"lang": "French", "country": "Canada"},
    "fr_FR": {"lang": "French", "country": "France"},
    "gu_IN": {"lang": "Gujarati", "country": "India"},
    "he_IL": {"lang": "Hebrew", "country": "Israel"},
    "hi_IN": {"lang": "Hindi", "country": "India"},
    "hr_HR": {"lang": "Croatian", "country": "Croatia"},
    "hu_HU": {"lang": "Hungarian", "country": "Hungary"},
    "id_ID": {"lang": "Indonesian", "country": "Indonesia"},
    "it_IT": {"lang": "Italian", "country": "Italy"},
    "ja_JP": {"lang": "Japanese", "country": "Japan"},
    "kn_IN": {"lang": "Kannada", "country": "India"},
    "ko_KR": {"lang": "Korean", "country": "South Korea"},
    "lt_LT": {"lang": "Lithuanian", "country": "Lithuania"},
    "lv_LV": {"lang": "Latvian", "country": "Latvia"},
    "ml_IN": {"lang": "Malayalam", "country": "India"},
    "mr_IN": {"lang": "Marathi", "country": "India"},
    "nl_NL": {"lang": "Dutch", "country": "Netherlands"},
    "no_NO": {"lang": "Norwegian", "country": "Norway"},
    "pa_IN": {"lang": "Punjabi", "country": "India"},
    "pl_PL": {"lang": "Polish", "country": "Poland"},
    "pt_BR": {"lang": "Portuguese", "country": "Brazil"},
    "pt_PT": {"lang": "Portuguese", "country": "Portugal"},
    "ro_RO": {"lang": "Romanian", "country": "Romania"},
    "ru_RU": {"lang": "Russian", "country": "Russia"},
    "sk_SK": {"lang": "Slovak", "country": "Slovakia"},
    "sl_SI": {"lang": "Slovenian", "country": "Slovenia"},
    "sr_RS": {"lang": "Serbian", "country": "Serbia"},
    "sv_SE": {"lang": "Swedish", "country": "Sweden"},
    "sw_KE": {"lang": "Swahili", "country": "Kenya"},
    "sw_TZ": {"lang": "Swahili", "country": "Tanzania"},
    "ta_IN": {"lang": "Tamil", "country": "India"},
    "te_IN": {"lang": "Telugu", "country": "India"},
    "th_TH": {"lang": "Thai", "country": "Thailand"},
    "tr_TR": {"lang": "Turkish", "country": "Turkey"},
    "uk_UA": {"lang": "Ukrainian", "country": "Ukraine"},
    "ur_PK": {"lang": "Urdu", "country": "Pakistan"},
    "vi_VN": {"lang": "Vietnamese", "country": "Vietnam"},
    "zh_CN": {"lang": "Mandarin", "country": "China"},
    "zh_TW": {"lang": "Mandarin", "country": "Taiwan"},
    "zu_ZA": {"lang": "Zulu", "country": "South Africa"},
}

WMT24PP_DEFAULTS_MAP = {
    # WMT24pp always has "en" on one side
    "en": "en",
    # Only MX is available for Spanish
    "es": "es_MX",
    "hbs": "hr_HR",
}

# https://huggingface.co/datasets/openlanguagedata/flores_plus#language-coverage
# pick ISO-639-3 default (generated by ChatGPT)
FLORES_PLUS_DEFAULTS_MAP = {
    # A
    "af": "afr_Latn",
    "am": "amh_Ethi",
    "ar": "arb_Arab",  # Arabic → default to MSA
    "as": "asm_Beng",
    "ay": "ayr_Latn",
    "az": "azj_Latn",  # Azerbaijani → default to North (Latin)
    # B
    "ba": "bak_Cyrl",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bm": "bam_Latn",
    "bn": "ben_Beng",
    "bo": "bod_Tibt",
    "bs": "bos_Latn",
    # C
    "ca": "cat_Latn",
    "ca_valencia": "cat_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    # D
    "da": "dan_Latn",
    "de": "deu_Latn",
    "dz": "dzo_Tibt",
    # E
    "ee": "ewe_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "eo": "epo_Latn",
    "es": "spa_Latn",
    "et": "ekk_Latn",  # ekk (Standard Estonian) used instead of est macrolanguage
    "eu": "eus_Latn",
    # F
    "fa": "pes_Arab",  # Persian → default to Iranian Persian
    "ff": "fuv_Latn",  # Fulah → default to Nigerian Fulfulde
    "fi": "fin_Latn",
    "fj": "fij_Latn",
    "fo": "fao_Latn",
    "fr": "fra_Latn",
    # G
    "ga": "gle_Latn",
    "gd": "gla_Latn",
    "gl": "glg_Latn",
    "gn": "gug_Latn",  # gug (Paraguayan Guarani) used, grn macrolanguage not available
    "gu": "guj_Gujr",
    # H
    "ha": "hau_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "ht": "hat_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    # I
    "id": "ind_Latn",
    "ig": "ibo_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    # J
    "ja": "jpn_Jpan",
    "jv": "jav_Latn",
    # K
    "ka": "kat_Geor",
    "ki": "kik_Latn",
    "kk": "kaz_Cyrl",  # default to Cyrillic
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Hang",
    "ks": "kas_Arab",  # default to Arabic script
    "ku": "kmr_Latn",  # Kurdish → default to Kurmanji (Latin)
    "ky": "kir_Cyrl",
    # L
    "lb": "ltz_Latn",
    "lg": "lug_Latn",
    "li": "lim_Latn",
    "ln": "lin_Latn",
    "lo": "lao_Laoo",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    # M
    "mg": "plt_Latn",  # Malagasy → default to Plateau Malagasy
    "mi": "mri_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",  # Mongolian → default to Khalkha (Cyrillic)
    "mr": "mar_Deva",
    "ms": "zsm_Latn",  # Malay → default to Standard Malay (MSA)
    "mt": "mlt_Latn",
    "my": "mya_Mymr",
    # N
    "nb": "nob_Latn",
    "ne": "npi_Deva",
    "nl": "nld_Latn",
    "nn": "nno_Latn",
    "no": "nob_Latn",  # Norwegian → default to Bokmål
    "ny": "nya_Latn",
    # O
    "oc": "oci_Latn",
    "om": "gaz_Latn",
    "or": "ory_Orya",
    # P
    "pa": "pan_Guru",  # default to Gurmukhi
    "pl": "pol_Latn",
    "ps": "pbt_Arab",  # Pashto → default to Southern
    "pt": "por_Latn",
    # Q
    "qu": "quy_Latn",
    # R
    "rn": "run_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "rw": "kin_Latn",
    # S
    "sa": "san_Deva",
    "sc": "srd_Latn",
    "sd": "snd_Arab",
    "sg": "sag_Latn",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sm": "smo_Latn",
    "sn": "sna_Latn",
    "so": "som_Latn",
    "sq": "als_Latn",  # Albanian → default to 'als' in this set
    "sr": "srp_Cyrl",  # default to Cyrillic
    "ss": "ssw_Latn",
    "st": "sot_Latn",
    "su": "sun_Latn",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    # T
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "tg": "tgk_Cyrl",
    "th": "tha_Thai",
    "ti": "tir_Ethi",
    "tk": "tuk_Latn",
    "tl": "fil_Latn",  # fil (Filipino) used, tgl not available
    "tn": "tsn_Latn",
    "tr": "tur_Latn",
    "ts": "tso_Latn",
    "tt": "tat_Cyrl",
    "tw": "twi_Latn_akua1239",  # Akuapem Twi chosen over Asante (asan1239)
    # U
    "ug": "uig_Arab",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",  # default to Latin
    # V
    "vi": "vie_Latn",
    # W
    "wo": "wol_Latn",
    # X
    "xh": "xho_Latn",
    # Y
    "yi": "ydd_Hebr",
    "yo": "yor_Latn",
    # Z
    "zh": "cmn_Hans",  # Mandarin Simplified
    "zh_hant": "cmn_Hant",  # Mandarin Traditional
    "zu": "zul_Latn",
}

# The default langauge codes that are different from FLORES 200 plus
# see https://huggingface.co/datasets/openlanguagedata/flores_plus/blob/main/CHANGELOG.md
NLLB_DEFAULTS_MAP = {
    "zh": "zho_Hans",
    "zh_hant": "zho_Hant",
    "tl": "tgl_Latn",
    "gn": "grn_Latn",
    "et": "est_Latn",
    "tw": "twi_Latn",
}

# https://huggingface.co/datasets/facebook/bouquet
BOUQUET_DEFAULTS_MAP = {
    "ar": "arz_Arab",
    "bn": "ben_Beng",
    "cs": "ces_Latn",
    "zh": "cmn_Hans",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "fa": "pes_Arab",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "km": "khm_Khmr",
    "ko": "kor_Kore",
    "ln": "lin_Latn",
    "my": "mya_Mymr",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn_braz1246",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "tl": "tgl_Latn",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "vi": "vie_Latn",
    "ms": "zsm_Latn",
}


# Google Translate v2 get_languages()
GOOGLE_LANGS = {
    "ab",
    "ace",
    "ach",
    "af",
    "sq",
    "alz",
    "am",
    "ar",
    "hy",
    "as",
    "awa",
    "ay",
    "az",
    "ban",
    "bm",
    "ba",
    "eu",
    "btx",
    "bts",
    "bbc",
    "be",
    "bem",
    "bn",
    "bew",
    "bho",
    "bik",
    "bs",
    "br",
    "bg",
    "bua",
    "yue",
    "ca",
    "ceb",
    "ny",
    "zh",
    "zh-TW",
    "cv",
    "co",
    "crh",
    "hr",
    "cs",
    "da",
    "dv",
    "din",
    "doi",
    "dov",
    "nl",
    "dz",
    "en",
    "eo",
    "et",
    "ee",
    "fj",
    "tl",
    "fi",
    "fr",
    "fy",
    "ff",
    "gaa",
    "gl",
    "ka",
    "de",
    "el",
    "gn",
    "gu",
    "ht",
    "cnh",
    "ha",
    "haw",
    "iw",
    "hil",
    "hi",
    "hmn",
    "hu",
    "hrx",
    "is",
    "ig",
    "ilo",
    "id",
    "ga",
    "it",
    "ja",
    "jw",
    "kn",
    "pam",
    "kk",
    "km",
    "cgg",
    "rw",
    "ktu",
    "gom",
    "ko",
    "kri",
    "ku",
    "ckb",
    "ky",
    "lo",
    "ltg",
    "la",
    "lv",
    "lij",
    "li",
    "ln",
    "lt",
    "lmo",
    "lg",
    "luo",
    "lb",
    "mk",
    "mai",
    "mak",
    "mg",
    "ms",
    "ms-Arab",
    "ml",
    "mt",
    "mi",
    "mr",
    "chm",
    "mni-Mtei",
    "min",
    "lus",
    "mn",
    "my",
    "nr",
    "new",
    "ne",
    "no",
    "nus",
    "oc",
    "or",
    "om",
    "pag",
    "pap",
    "ps",
    "fa",
    "pl",
    "pt",
    "pa",
    "pa-Arab",
    "qu",
    "rom",
    "ro",
    "rn",
    "ru",
    "sm",
    "sg",
    "sa",
    "gd",
    "nso",
    "sr",
    "st",
    "crs",
    "shn",
    "sn",
    "scn",
    "szl",
    "sd",
    "si",
    "sk",
    "sl",
    "so",
    "es",
    "su",
    "sw",
    "ss",
    "sv",
    "tg",
    "ta",
    "tt",
    "te",
    "tet",
    "th",
    "ti",
    "ts",
    "tn",
    "tr",
    "tk",
    "ak",
    "uk",
    "ur",
    "ug",
    "uz",
    "vi",
    "cy",
    "xh",
    "yi",
    "yo",
    "yua",
    "zu",
    "he",
    "jv",
    "zh-CN",
}

# https://learn.microsoft.com/en-us/azure/ai-services/translator/language-support
MICROSOFT_LANGS = [
    "af",
    "sq",
    "am",
    "ar",
    "hy",
    "as",
    "az",
    "bn",
    "ba",
    "eu",
    "bho",
    "brx",
    "bs",
    "bg",
    "yue",
    "ca",
    "hne",
    "lzh",
    "zh-Hans",
    "zh-Hant",
    "sn",
    "hr",
    "cs",
    "da",
    "prs",
    "dv",
    "doi",
    "nl",
    "en",
    "et",
    "fo",
    "fj",
    "fil",
    "fi",
    "fr",
    "fr-ca",
    "gl",
    "ka",
    "de",
    "el",
    "gu",
    "ht",
    "ha",
    "he",
    "hi",
    "mww",
    "hu",
    "is",
    "ig",
    "id",
    "ikt",
    "iu",
    "iu-Latn",
    "ga",
    "it",
    "ja",
    "kn",
    "ks",
    "kk",
    "km",
    "rw",
    "tlh-Latn",
    "tlh-Piqd",
    "gom",
    "ko",
    "ku",
    "kmr",
    "ky",
    "lo",
    "lv",
    "lt",
    "ln",
    "dsb",
    "lug",
    "mk",
    "mai",
    "mg",
    "ms",
    "ml",
    "mt",
    "mni",
    "mi",
    "mr",
    "mn-Cyrl",
    "mn-Mong",
    "my",
    "ne",
    "nb",
    "nya",
    "or",
    "ps",
    "fa",
    "pl",
    "pt",
    "pt-pt",
    "pa",
    "otq",
    "ro",
    "run",
    "ru",
    "sm",
    "sr-Cyrl",
    "sr-Latn",
    "st",
    "nso",
    "tn",
    "sd",
    "si",
    "sk",
    "sl",
    "so",
    "es",
    "sw",
    "sv",
    "ty",
    "ta",
    "tt",
    "te",
    "th",
    "bo",
    "ti",
    "to",
    "tr",
    "tk",
    "uk",
    "hsb",
    "ur",
    "ug",
    "uz",
    "vi",
    "cy",
    "xh",
    "yo",
    "yua",
    "zu",
]

PONTOON_LANGUAGES = {"aa", "aat", "ab", "abb", "abq", "ace", "ach", "ady", "af", "ajg", "ak",
                     "aln", "am", "an", "ann", "anp", "ar", "arn", "as", "ast", "ay", "az", "azb", "azz",
                     "ba", "bag", "bal", "ban", "bas", "bax", "bba", "bbj", "bbl", "bce", "bci", "be", "beb",
                     "bew", "bfd", "bft", "bg", "bgp", "bkh", "bkm", "bm", "bn", "bnm", "bnn", "bo", "bqi",
                     "br", "brh", "bri", "brx", "bs", "bsh", "bsk", "bsy", "btv", "bum", "bxk", "bxr", "byv",
                     "ca", "cak", "cdo", "ceb", "cgg", "cjk", "ckb", "cnh", "co", "cpx", "cpy", "crh", "cs",
                     "csb", "cut", "cux", "cv", "cy", "da", "dag", "dar", "dav", "de", "din", "dmk", "dml",
                     "dru", "dsb", "dua", "dv", "dyu", "ebr", "ee", "eko", "el", "en", "eo", "es", "esu",
                     "et", "eto", "eu", "ewo", "fa", "fan", "ff", "fi", "fmp", "fo", "fr", "frp", "fub",
                     "fue", "fuf", "fur", "fy", "ga", "gaa", "gd", "gej", "ggg", "gid", "gig", "giz", "gjk",
                     "gju", "gl", "gn", "gom", "gor", "gos", "gsw", "gu", "guc", "gv", "gwc", "gwt", "gya",
                     "ha", "hac", "haz", "hch", "he", "hem", "hi", "hil", "hno", "hr", "hrx", "hsb", "ht",
                     "hu", "hus", "hux", "hy", "hye", "hyw", "ia", "iba", "ibb", "id", "ie", "ig", "ilo",
                     "ipk", "is", "it", "ixl", "izh", "ja", "jam", "jbo", "jgo", "jiv", "jqr", "jv", "ka",
                     "kaa", "kab", "kam", "kbd", "kcn", "kdh", "khw", "ki", "kk", "kln", "kls", "km", "kmr",
                     "kn", "knn", "ko", "kok", "koo", "kpv", "krc", "ks", "ksf", "kvx", "kw", "kxp", "ky",
                     "kzi", "lb", "led", "leu", "lg", "lij", "lke", "lld", "ln", "lo", "lrk", "lrl", "lss",
                     "lt", "ltg", "lth", "lua", "luo", "lus", "lv", "lzz", "mai", "mau", "mbf", "mbo", "mcf",
                     "mcn", "mcx", "mdd", "mdf", "meh", "mel", "mfe", "mg", "mgg", "mhk", "mhr", "mix", "mk",
                     "mki", "ml", "mmc", "mn", "mni", "mos", "mqh", "mr", "mrh", "mrj", "ms", "mse", "msi",
                     "mt", "mua", "mug", "mve", "mvy", "mxu", "my", "myv", "nan", "nb", "ncx", "nd", "ne",
                     "new", "nhe", "nhi", "nia", "nl", "nla", "nlv", "nmg", "nmz", "nn", "nnh", "nqo", "nr",
                     "nso", "nv", "ny", "nyn", "nyu", "oc", "odk", "om", "or", "oru", "os", "pa", "pai",
                     "pap", "pcd", "pcm", "pez", "phl", "phr", "pl", "plk", "pne", "ppl", "prq", "ps", "pt",
                     "pua", "pwn", "quc", "qug", "qup", "qur", "qus", "qux", "quy", "qva", "qvi", "qvj", "qvl",
                     "qwa", "qws", "qxa", "qxp", "qxq", "qxt", "qxu", "qxw", "rif", "rm", "rn", "ro", "rof",
                     "ru", "ruc", "rup", "rw", "rwm", "sah", "sat", "sbn", "sc", "scl", "scn", "sco", "sd",
                     "sdh", "sdo", "seh", "sei", "ses", "shi", "shn", "si", "sk", "skr", "sl", "sn", "snk",
                     "snv", "so", "son", "sq", "sr", "ss", "ssi", "st", "su", "sv", "sva", "sw", "syr", "szl",
                     "szy", "ta", "tar", "tay", "te", "teg", "tg", "th", "ti", "tig", "tk", "tl", "tli", "tn",
                     "tob", "tok", "top", "tr", "trs", "trv", "trw", "ts", "tsz", "tt", "ttj", "tui", "tvu",
                     "tw", "ty", "tyv", "tzm", "uby", "udl", "udm", "ug", "uk", "ukv", "ur", "ush", "uz", "var",
                     "ve", "vec", "vi", "vmw", "vot", "wbl", "wep", "wes", "wo", "xcl", "xdq", "xh", "xhe",
                     "xka", "xkl", "xmf", "xsm", "yaq", "yav", "ydg", "yi", "yo", "yua", "yue", "zam", "zgh",
                     "zh", "zoc", "zu", "zza"
                     }  # fmt: skip

PONTOON_DEFAULTS_BCP_MAP = {
    "sv": "sv-SE",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "nn": "nn-NO",
    "nb": "nb-NO",
    "no": "nb-NO",
    "ne": "ne-NP",
    "hi": "hi-IN",
    "hy": "hy-AM",
    "ga": "ga-IE",
    "bn": "bn-IN",
    "zh": "zh-CN",
    "zh_hant": "zh-TW",
}

# https://github.com/facebookresearch/flores/blob/main/previous_releases/flores101/README.md
FLORES_101_LANGUAGES = {
    "afr",
    "amh",
    "ara",
    "hye",
    "asm",
    "ast",
    "azj",
    "bel",
    "ben",
    "bos",
    "bul",
    "mya",
    "cat",
    "ceb",
    "zho_simpl",
    "zho_trad",
    "hrv",
    "ces",
    "dan",
    "nld",
    "eng",
    "est",
    "tgl",
    "fin",
    "fra",
    "ful",
    "glg",
    "lug",
    "kat",
    "deu",
    "ell",
    "guj",
    "hau",
    "heb",
    "hin",
    "hun",
    "isl",
    "ibo",
    "ind",
    "gle",
    "ita",
    "jpn",
    "jav",
    "kea",
    "kam",
    "kan",
    "kaz",
    "khm",
    "kor",
    "kir",
    "lao",
    "lav",
    "lin",
    "lit",
    "luo",
    "ltz",
    "mkd",
    "msa",
    "mal",
    "mlt",
    "mri",
    "mar",
    "mon",
    "npi",
    "nso",
    "nob",
    "nya",
    "oci",
    "ory",
    "orm",
    "pus",
    "fas",
    "pol",
    "por",
    "pan",
    "ron",
    "rus",
    "srp",
    "sna",
    "snd",
    "slk",
    "slv",
    "som",
    "ckb",
    "spa",
    "swh",
    "swe",
    "tgk",
    "tam",
    "tel",
    "tha",
    "tur",
    "ukr",
    "umb",
    "urd",
    "uzb",
    "vie",
    "cym",
    "wol",
    "xho",
    "yor",
    "zul",
}

FLORES_101_DEFAULTS_MAP = {
    "zh": "zho_simpl",
    "zh_hant": "zho_trad",
    "hbs": "hrv",
}

# https://huggingface.co/bitextor/models
BICLEANER_AI_DEFAULTS_MAP = {
    "zh_hant": "zh",
    # Serbo-Croatian model
    "sr": "hbs",
    "bs": "hbs",
    "hr": "hbs",
    "cnr": "hbs",
    "hbs": "hbs",
    # Default to Norwegian Bokmal
    "no": "nb",
}

MONOCLEANER_DEFAULTS_MAP = {
    # Jump over iso639-1 conversion for hbs
    "hbs": "hbs",
    "hr": "hbs",
    "sr": "hbs",
    "bs": "hbs",
    "cnr": "hbs",
}

ISO6393_DEFAULTS_MAP = {
    # ICU returns Kore by default which is a mix of Hang and Hani
    "ko": "kor_Hang",
    # zh is a macro language, map to Mandarin Chinese by default
    "zh": "cmn_Hans",
    "zh_hant": "cmn_Hant",
    # hbs is a macro, default to Croatian Latin (which is the one with better support)
    "hbs": "hrv_Latn",
}

ISO6393_MACRO_ISO6391_DEFAULTS_MAP = {
    # Default to Croatian for Serbo-Croatian
    "hbs": "hr",
}

ISO6393_MACRO_DEFAULTS_MAP = {
    # Serbo-Croatian is digraphic, default to Latin
    "hbs": "hbs_Latn",
}

ISO6393_DEFAULTS_REVERSED_MAP = {v: k for k, v in ISO6393_DEFAULTS_MAP.items()}

ISO6393_MACRO_DEFAULTS_MAP_NO_SCRIPT = {
    k: v.split("_")[0] for k, v in ISO6393_MACRO_DEFAULTS_MAP.items()
}

NAMES_DEFAULT_MAP = {
    "hbs": "Serbo-Croatian",
}

FASTTEXT_DEFAULTS_MAP = {
    "hbs": ("hrv_Latn", "bos_Latn", "bos_Cyrl", "srp_Cyrl", "srp_Latn"),
}

COMMON_FALLBACKS = {
    "ca_valencia": ["ca"],
    # Fallback to the old code (MetricX)
    "he": ["iw"],
    # Norwegian Bokmal is standard
    "no": ["nb", "no_NO"],
    "nb": ["no", "no_NO"],
    "nn": ["no", "no_NO"],
    "pt_pt": ["pt"],
    "sq": ["als"],
    "hbs": ["hr", "sr", "bs"],
    "sr": ["sr-Cyrl"],
    "tl": ["fil", "fil_PH"],
    "zh": ["cmn_Hans", "zh-CN", "zh-Hans"],
    "zh_hant": ["cmn_Hant", "zh-TW", "zh-Hant"],
}

# Language variants that can be supported by the pipeline
# Regenerate langs JSON after adding more variants to verify support and required fixes
#   under Docker run: task generate-langs-map
PIPELINE_SUPPORT = [
    "af",
    "ar",
    "as",
    "az",
    "be",
    "bg",
    "bn",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "ff",
    "fi",
    "fr",
    "ga",
    "gd",
    "gl",
    "gn",
    "gu",
    "hbs",
    "he",
    "hi",
    "hr",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "lt",
    "lv",
    "mk",
    "ml",
    "mr",
    "ms",
    "my",
    "nb",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "or",
    "pa",
    "pl",
    "pt",
    "ro",
    "ru",
    "si",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "xh",
    "zh",
    "zh_hant",
]
