from datasets import load_dataset

EVAL_PAIRS = (
    "en-ar_EG",
    "en-ar_SA",
    "en-bg_BG",
    "en-bn_IN",
    "en-ca_ES",
    "en-cs_CZ",
    "en-da_DK",
    "en-de_DE",
    "en-el_GR",
    "en-es_MX",
    "en-et_EE",
    "en-fa_IR",
    "en-fi_FI",
    "en-fil_PH",
    "en-fr_CA",
    "en-fr_FR",
    "en-gu_IN",
    "en-he_IL",
    "en-hi_IN",
    "en-hr_HR",
    "en-hu_HU",
    "en-id_ID",
    "en-is_IS",
    "en-it_IT",
    "en-ja_JP",
    "en-kn_IN",
    "en-ko_KR",
    "en-lt_LT",
    "en-lv_LV",
    "en-ml_IN",
    "en-mr_IN",
    "en-nl_NL",
    "en-no_NO",
    "en-pa_IN",
    "en-pl_PL",
    "en-pt_BR",
    "en-pt_PT",
    "en-ro_RO",
    "en-ru_RU",
    "en-sk_SK",
    "en-sl_SI",
    "en-sr_RS",
    "en-sv_SE",
    "en-sw_KE",
    "en-sw_TZ",
    "en-ta_IN",
    "en-te_IN",
    "en-th_TH",
    "en-tr_TR",
    "en-uk_UA",
    "en-ur_PK",
    "en-vi_VN",
    "en-zh_CN",
    "en-zh_TW",
    "en-zu_ZA",
)


lang_map = {
    pair.split("_")[0].split("-")[1]: pair
    for pair in EVAL_PAIRS
    if pair.split("_")[1] not in {"TW", "PT", "CA", "EG", "TZ"}
}


def load_data(lang):
    if lang not in lang_map:
        raise ValueError(f"Language {lang} is not supported")

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("google/wmt24pp")
    filtered = ds.filter(lambda ex: not ex["is_bad_source"] and ex["lp"] == lang_map[lang])[
        "train"
    ]
    return filtered["source"], filtered["target"]


def eval(source_texts, target_translations, target_references):
    pass
