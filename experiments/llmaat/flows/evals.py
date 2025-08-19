from typing import List

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
    from datasets import load_dataset

    #
    # if lang not in lang_map:
    #     raise ValueError(f"Language {lang} is not supported")

    # Login using e.g. `huggingface-cli login` to access this dataset
    print(f"Downloading dataset for {lang}")
    lp = f"en-{lang}"
    ds = load_dataset("google/wmt24pp", lp)
    filtered = ds.filter(lambda ex: not ex["is_bad_source"] and ex["lp"] == lp)["train"]
    return filtered["source"], filtered["target"]


def eval_comet(source_texts, target_translations, target_references):
    import comet

    comet_checkpoint = comet.download_model("Unbabel/wmt22-comet-da")
    comet_model = comet.load_from_checkpoint(comet_checkpoint)
    comet_data = []
    for source, target, target_ref in zip(source_texts, target_translations, target_references):
        comet_data.append({"src": source, "mt": target, "ref": target_ref})
    comet_results = comet_model.predict(comet_data, gpus=1)
    return round(comet_results.system_score * 100, 2)


def eval_metricx(
    source_texts,
    target_translations,
    target_references,
    model_size="xl",
    fp16=True,
    batch_size=8,
):
    """
    https://huggingface.co/google/metricx-24-hybrid-xxl-v2p6

    Available model sizes: "large" (1.2B), "xl" (3.7B), "xxl" (13b)
    """

    import json
    from statistics import mean
    from metricx.predict import predict

    with open("input.jsonl", "w") as in_file:
        for source, target, target_ref in zip(
            source_texts, target_translations, target_references
        ):
            ex_dict = {"source": source, "reference": target_ref, "hypothesis": target}
            in_file.write(json.dumps(ex_dict) + "\n")

    model_name = f"google/metricx-24-hybrid-{model_size}-v2p6"
    if fp16:
        model_name += "-bfloat16"

    # batch size is divided by number of GPUs, set equal or higher
    print(f"Running evaluation with {model_name} reference based")
    predict(
        tokenizer=f"google/mt5-{model_size}",
        model_name_or_path=model_name,
        max_input_length=1536,
        batch_size=batch_size,
        input_file="input.jsonl",
        output_file="output.ref.jsonl",
        qe=False,
    )

    print(f"Running evaluation with {model_name} reference free QE")
    predict(
        tokenizer=f"google/mt5-{model_size}",
        model_name_or_path=model_name,
        max_input_length=1536,
        batch_size=batch_size,
        input_file="input.jsonl",
        output_file="output.qe.jsonl",
        qe=True,
    )

    with open("output.qe.jsonl") as out_qe:
        qe_score = mean([float(json.loads(line)["prediction"]) for line in out_qe])
    with open("output.ref.jsonl") as out_ref:
        ref_score = mean([float(json.loads(line)["prediction"]) for line in out_ref])

    return {f"metricx24-{model_size}-qe": qe_score, f"metricx24-{model_size}": ref_score}


def select_best(
    source: List[str], translations: List[List[str]], model_size, batch_size, fp16=True
) -> List[str]:
    import json
    from metricx.predict import predict

    with open("input.jsonl", "w") as in_file:
        for (
            source,
            tr_candidates,
        ) in zip(source, translations):
            for translation in tr_candidates:
                ex_dict = {"source": source, "hypothesis": translation}
                in_file.write(json.dumps(ex_dict) + "\n")

    model_name = f"google/metricx-24-hybrid-{model_size}-v2p6"
    if fp16:
        model_name += "-bfloat16"

    print(f"Running evaluation with {model_name} reference free QE")
    predict(
        tokenizer=f"google/mt5-{model_size}",
        model_name_or_path=model_name,
        max_input_length=1536,
        batch_size=batch_size,
        input_file="input.jsonl",
        output_file="output.qe.jsonl",
        qe=True,
    )

    with open("output.qe.jsonl") as out_qe:
        scores = [json.loads(line)["prediction"] for line in out_qe]

    num_candidates = len(translations[0])

    best = []
    best_scores = []
    for i, candidates in enumerate(translations):
        start = i * num_candidates
        candidate_scores = scores[start : start + num_candidates]
        best_idx = candidate_scores.index(min(candidate_scores))
        best.append(candidates[best_idx])
        best_scores.append(candidate_scores[best_idx])
    return best, best_scores


def _run_cmd(cmd):
    import subprocess

    try:
        subprocess.run(cmd, check=True, capture_output=True, shell=True)
    except subprocess.CalledProcessError as e:
        print("STDOUT:", e.stdout.decode("utf-8", errors="replace"))
        print("STDERR:", e.stderr.decode("utf-8", errors="replace"))
        raise
