"""
Tests evaluations
"""

import json
import os

import pytest
from fixtures import DataDir, en_sample, ru_sample

en_fake_translated = "\n".join([line.upper() for line in ru_sample.split("\n")])
ru_fake_translated = "\n".join([line.upper() for line in en_sample.split("\n")])

current_folder = os.path.dirname(os.path.abspath(__file__))
fixtures_path = os.path.join(current_folder, "fixtures")
root_path = os.path.abspath(os.path.join(current_folder, ".."))


def get_base_marian_args(data_dir: DataDir, model_name: str):
    return [
        "--models", data_dir.join(model_name),
        "--config", data_dir.join("final.model.npz.best-chrf.npz.decoder.yml"),
        "--quiet",
        "--quiet-translation",
        "--log", data_dir.join("artifacts/wmt09.log"),
        '--workspace', '12000',
        '--devices', '0',
    ]  # fmt: skip


def get_quantized_marian_args(data_dir: DataDir, model_name: str):
    return [
        "--models", data_dir.join(model_name),
        "--config", os.path.join(root_path, "pipeline/quantize/decoder.yml"),
        "--quiet",
        "--quiet-translation",
        "--log", data_dir.join("artifacts/wmt09.log"),
        '--int8shiftAlphaAll',
        '--vocabs', data_dir.join("vocab.en.spm"), data_dir.join("vocab.ru.spm"),
        '--shortlist', data_dir.join("lex.s2t.pruned"),
    ]  # fmt: skip


comet_score = 0.3268
comet_skipped = "skipped"

test_data = [
    # task_name                                          model_type   model_name                       comet_score unaligned-ratio score
    ("evaluate-backward-sacrebleu-wmt09-en-ru",          "base",      "final.model.npz.best-chrf.npz", comet_skipped, comet_skipped),
    ("evaluate-finetuned-student-sacrebleu-wmt09-en-ru", "base",      "final.model.npz.best-chrf.npz", comet_skipped, comet_skipped),
    ("evaluate-teacher-ensemble-sacrebleu-wmt09-en-ru",  "base",      "model*/*.npz",                  comet_skipped, comet_skipped),
    ("evaluate-quantized-sacrebleu-wmt09-en-ru",         "quantized", "model.intgemm.alphas.bin",      comet_skipped, comet_skipped)
]  # fmt:skip


@pytest.mark.parametrize("params", test_data, ids=[d[0] for d in test_data])
def test_evaluate(params) -> None:
    run_eval_test(params)


# COMET is quite slow on CPU, so split out only a single test that exercises it.
# unaligned-ratio metric will be skipped if COMET is skipped
test_data_comet = [
    ("evaluate-student-sacrebleu-wmt09-en-ru",                    "base",      "final.model.npz.best-chrf.npz", comet_score, 0.46875),
]  # fmt:skip


@pytest.mark.slow  # comet is slow to evaluate.
@pytest.mark.parametrize("params", test_data_comet, ids=[d[0] for d in test_data_comet])
def test_evaluate_comet(params) -> None:
    run_eval_test(params)


def run_eval_test(params) -> None:
    (task_name, model_type, model_name, comet, unaligned_ratio) = params

    data_dir = DataDir("test_eval")
    data_dir.create_zst("wmt09.en.zst", en_sample)
    data_dir.create_zst("wmt09.ru.zst", ru_sample)
    data_dir.create_file("final.model.npz.best-chrf.npz.decoder.yml", "{}")

    model_path = os.path.join(root_path, "data/models")
    os.makedirs(model_path, exist_ok=True)

    bleu = 0.4
    chrf = 0.64

    if model_type == "base":
        expected_marian_args = get_base_marian_args(data_dir, model_name)
        env = {
            # This is where the marian_decoder_args will get stored.
            "TEST_ARTIFACTS": data_dir.path,
            # Replace marian with the one in the fixtures path.
            "MARIAN": fixtures_path,
            "COMET_MODEL_DIR": model_path,
            "COMET_CPU": "1",
        }
    elif model_type == "quantized":
        expected_marian_args = get_quantized_marian_args(data_dir, model_name)
        env = {
            # This is where the marian_decoder_args will get stored.
            "TEST_ARTIFACTS": data_dir.path,
            # Replace marian with the one in the fixtures path.
            "BMT_MARIAN": fixtures_path,
            "COMET_MODEL_DIR": model_path,
            "COMET_CPU": "1",
        }

    if comet == "skipped":
        env["COMET_SKIP"] = "1"

    # Run the evaluation.
    data_dir.run_task(
        task_name,
        env=env,
    )

    # Test that the data files are properly written out.
    if "backward" in task_name:
        # Backwards evaluation.
        assert data_dir.read_text("artifacts/wmt09.ru") == ru_sample
        assert data_dir.read_text("artifacts/wmt09.en.ref") == en_sample
        assert data_dir.read_text("artifacts/wmt09.en") == en_fake_translated
    else:
        # Forwards evaluation.
        assert data_dir.read_text("artifacts/wmt09.en") == en_sample
        assert data_dir.read_text("artifacts/wmt09.ru.ref") == ru_sample
        assert data_dir.read_text("artifacts/wmt09.ru") == ru_fake_translated

    # Test that the JSON metrics get properly generated.
    metrics_json = json.loads(data_dir.read_text("artifacts/wmt09.metrics.json"))

    assert metrics_json["bleu"]["details"]["name"] == "BLEU"
    assert metrics_json["bleu"]["details"]["score"] == bleu
    assert metrics_json["bleu"]["score"] == bleu

    assert metrics_json["chrf"]["details"]["name"] == "chrF2"
    assert metrics_json["chrf"]["details"]["score"] == chrf
    assert metrics_json["chrf"]["score"] == chrf

    assert metrics_json["comet"]["details"]["model"] == "Unbabel/wmt22-comet-da"
    assert metrics_json["comet"]["details"]["score"] == comet
    assert metrics_json["comet"]["score"] == comet

    assert metrics_json["unaligned-ratio"]["score"] == unaligned_ratio

    # Test that marian is given the proper arguments.
    marian_decoder_args = json.loads(data_dir.read_text("marian-decoder.args.txt"))
    assert marian_decoder_args == expected_marian_args, "The marian arguments matched."
