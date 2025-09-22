#!/usr/bin/env python3
"""
Evaluate a trained model with both the BLEU and chrF metrics.

Kinds:
   taskcluster/kinds/evaluate/kind.yml
   taskcluster/kinds/evaluate-quantized/kind.yml
   taskcluster/kinds/evaluate-teacher-ensemble/kind.yml

Example usage:

    $VCS_PATH/pipeline/eval/eval.py
        --src               en                                                 \\
        --trg               ca                                                 \\
        --marian_config     fetches/final.model.npz.best-chrf.npz.decoder.yml  \\
        --models            fetches/final.model.npz.best-chrf.npz              \\
        --dataset_prefix    fetches/wmt09                                      \\
        --artifacts_prefix  artifacts/wmt09                                    \\
        --model_variant     gpu                                                \\
        --workspace         12000                                              \\
        --gpus              4

Artifacts:

For instance for a artifacts_prefix of: "artifacts/wmt09":

  artifacts
  ├── wmt09.en             The source sentences
  ├── wmt09.ca             The target output
  ├── wmt09.ca.ref         The original target sentences
  ├── wmt09.log            The Marian log
  ├── wmt09.metrics        The BLEU and chrF score
  └── wmt09.metrics.json   The BLEU and chrF score in json format

Fetches:

For instance for a value of: "fetches/wmt09":
  fetches
  ├── wmt09.en.zst
  └── wmt09.ca.zst
"""


import argparse
import json
import os
import subprocess
from textwrap import dedent, indent
from typing import Optional, List

from sacrebleu.metrics.bleu import BLEU, BLEUScore
from sacrebleu.metrics.chrf import CHRF, CHRFScore
from simalign import SentenceAligner
import torch

from pipeline.alignments.tokenizer import IcuTokenizer
from pipeline.common.downloads import decompress_file
from pipeline.common.logging import get_logger
from pipeline.common.marian import assert_gpus_available

logger = get_logger("eval")
try:
    import wandb
    from translations_parser.publishers import METRIC_KEYS, WandB
    from translations_parser.utils import metric_from_tc_context
    from translations_parser.wandb import (
        add_wandb_arguments,
        get_wandb_publisher,
        list_existing_group_logs_metrics,
    )

    WANDB_AVAILABLE = "TASKCLUSTER_PROXY_URL" in os.environ
except ImportError as e:
    print(f"Failed to import tracking module: {e}")
    WANDB_AVAILABLE = False


def compute_unaliged_ratio(
    src: str, trg: str, source_lines: List[str], target_lines: List[str], device: torch.device
):
    """
    Compute ratio of unaligned tokens
    """
    aligner = SentenceAligner(
        model="bert", token_type="bpe", matching_methods="i", device=device
    )
    logger.info(f"Using device '{aligner.device}'")
    src_tokenizer = IcuTokenizer(src)
    trg_tokenizer = IcuTokenizer(trg)
    src_tokens = [src_tokenizer.tokenize_nospace(i) for i in source_lines]
    trg_tokens = [trg_tokenizer.tokenize_nospace(i) for i in target_lines]

    def gen_alignments():
        for st, tt in zip(src_tokens, trg_tokens):
            if len(st) == 0 or len(tt) == 0:
                # if there are empty sentences, alignment is empty
                # avoid feeding it to the aligner because it crashes
                # this mainly happens on the CI because tested models are poorly trained
                yield {"itermax": []}
                continue
            try:
                yield aligner.get_word_aligns(st, tt)
            except ValueError as e:
                # When there is a sentences containing only characters that are not present
                # in the sentencealigner vocab, it cannot align because there are no embeddings
                # so, in this case we just return an empty alignment pairs
                # which means nothing could be aligned
                # this mainly happens on the CI because tested models are poorly trained
                if e.args and e.args[0].startswith("Found array with 0 sample"):
                    yield {"itermax": []}
                else:
                    raise e
            except Exception as e:
                # If it fails print sentence pair for easier debugging
                logger.error("Getting word alignments failed on this sentence pair:")
                logger.error(f"Source: {st}")
                logger.error(f"Target: {tt}")
                raise e

    alignment_lines = list(gen_alignments())

    # for each line, count the number of target tokens that are not present in alignment indices
    # (unaligned tokens)
    ratio_sum = 0
    for tokens, alignment in zip(trg_tokens, alignment_lines):
        trg_indices = set(range(len(tokens)))
        for alignment_indexes in alignment["itermax"]:
            trg_index = alignment_indexes[1]
            if trg_index in trg_indices:
                trg_indices.remove(trg_index)
        ratio_sum += len(trg_indices) / len(trg_tokens)

    return ratio_sum / len(target_lines)


def run_bash_oneliner(command: str):
    """
    Runs multi-line bash with comments as a one-line command.
    """
    command_dedented = dedent(command)

    # Remove comments and whitespace.
    lines = [
        line.strip() for line in command_dedented.split("\n") if line and not line.startswith("#")
    ]
    command = " \\\n".join(lines)

    logger.info("-----------------Running bash in one line--------------")
    logger.info(indent(command_dedented, "  "))
    logger.info("-------------------------------------------------------")
    return subprocess.check_call(command, shell=True)


def main(args_list: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves whitespace in the help text.
    )
    parser.add_argument(
        "--artifacts_prefix",
        type=str,
        help="The location where the translated results will be saved",
    )
    parser.add_argument(
        "--dataset_prefix", type=str, help="The evaluation datasets prefix, used in the form."
    )
    parser.add_argument("--src", type=str, help='The source language, e.g "en".')
    parser.add_argument("--trg", type=str, help='The target language, e.g "ca".')
    parser.add_argument("--marian", type=str, help="The path the to marian binaries.")
    parser.add_argument("--marian_config", type=str, help="The marian yaml config for the model.")
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Use a quantized model. This requires the browsermt fork of Marian",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="The Marian model (or models if its an ensemble) to use for translations",
    )
    parser.add_argument(
        "--vocab_src",
        required=False,
        type=str,
        help="The path to a src vocab file (optional)",
    )
    parser.add_argument(
        "--vocab_trg",
        required=False,
        type=str,
        help="The path to a trg vocab file (optional)",
    )
    parser.add_argument(
        "--shortlist",
        required=False,
        type=str,
        help="The path to a lexical shortlist (optional)",
    )
    parser.add_argument("--workspace", type=str, help="The preallocated MB for the workspace")
    parser.add_argument(
        "--gpus",
        required=False,
        type=str,
        help="Which GPUs to use (only for the gpu model variant)",
    )
    parser.add_argument(
        "--model_variant", type=str, help="The model variant to use, (gpu, cpu, quantized)"
    )

    # Add Weight & Biases CLI args when module is loaded
    if WANDB_AVAILABLE:
        add_wandb_arguments(parser)

    args = parser.parse_args(args_list)

    src = args.src
    trg = args.trg
    dataset_prefix = args.dataset_prefix
    artifacts_prefix = args.artifacts_prefix

    artifacts_dir = os.path.dirname(artifacts_prefix)
    source_file_compressed = f"{dataset_prefix}.{src}.zst"
    source_file = f"{artifacts_prefix}.{src}"
    target_file_compressed = f"{dataset_prefix}.{trg}.zst"
    target_file = f"{artifacts_prefix}.{trg}"
    target_ref_file = f"{artifacts_prefix}.{trg}.ref"
    marian_decoder = f'"{args.marian}"/marian-decoder'
    marian_log_file = f"{artifacts_prefix}.log"
    language_pair = f"{src}-{trg}"
    metrics_file = f"{artifacts_prefix}.metrics"
    metrics_json = f"{artifacts_prefix}.metrics.json"

    # Configure Marian for the different model variants.
    marian_extra_args = []
    if args.model_variant == "quantized":
        marian_extra_args = ["--int8shiftAlphaAll"]
    elif args.model_variant == "gpu":
        if not args.workspace:
            raise Exception("The workspace size was not provided")
        marian_extra_args = [
            '--workspace', args.workspace,
            '--devices', args.gpus,
        ]  # fmt: skip
    elif not args.model_variant == "cpu":
        raise Exception(f"Unsupported model variant {args.model_variant}")

    if args.vocab_src and args.vocab_trg:
        marian_extra_args = [*marian_extra_args, "--vocabs", args.vocab_src, args.vocab_trg]

    if args.shortlist:
        # No arguments to the shortlist, so default ones are used
        # this way it doesn't matter if the shortlist is binary or text
        # because they have different arguments
        # text shortlist args: firstNum bestNum threshold
        # binary shortlist (has the arguments embedded) args: bool (check integrity)
        marian_extra_args = marian_extra_args + ["--shortlist", args.shortlist]

    logger.info("The eval script is configured with the following:")
    logger.info(f" >          artifacts_dir: {artifacts_dir}")
    logger.info(f" > source_file_compressed: {source_file_compressed}")
    logger.info(f" >            source_file: {source_file}")
    logger.info(f" >            target_file: {target_file}")
    logger.info(f" >        target_ref_file: {target_ref_file}")
    logger.info(f" >         marian_decoder: {marian_decoder}")
    logger.info(f" >        marian_log_file: {marian_log_file}")
    logger.info(f" >          language_pair: {language_pair}")
    logger.info(f" >           metrics_file: {metrics_file}")
    logger.info(f" >           metrics_json: {metrics_json}")
    logger.info(f" >      marian_extra_args: {marian_extra_args}")
    logger.info(f" >                   gpus: {args.gpus}")

    logger.info("Ensure that the artifacts directory exists.")
    os.makedirs(artifacts_dir, exist_ok=True)

    logger.info("Save the original target sentences to the artifacts")

    assert_gpus_available(logger)

    decompress_file(target_file_compressed, keep_original=False, decompressed_path=target_ref_file)

    run_bash_oneliner(
        f"""
        # Decompress the source file, e.g. $fetches/wmt09.en.zst
        zstdmt -dc "{source_file_compressed}"

        # Tee the source file into the artifacts directory, e.g. $artifacts/wmt09.en
        | tee "{source_file}"

        # Take the source and pipe it in to be decoded (translated) by Marian.
        | {marian_decoder}
            --models {args.models}
            --config {args.marian_config}
            --quiet
            --quiet-translation
            --log {marian_log_file}
            {" ".join(marian_extra_args)}

        # The translations be "tee"ed out to the artifacts, e.g. $artifacts/wmt09.ca
        | tee "{target_file}"
        """
    )

    with open(target_ref_file, "r") as file:
        target_ref_lines = file.readlines()
    with open(target_file, "r") as file:
        target_lines = file.readlines()
    with open(source_file, "r") as file:
        source_lines = file.readlines()

    compute_bleu = BLEU(trg_lang=trg)
    compute_chrf = CHRF()

    logger.info("Computing the BLEU score.")
    bleu_score: BLEUScore = compute_bleu.corpus_score(target_lines, [target_ref_lines])
    bleu_details = json.loads(
        bleu_score.format(signature=compute_bleu.get_signature().format(), is_json=True)
    )

    logger.info("Computing the chrF score.")
    chrf_score: CHRFScore = compute_chrf.corpus_score(target_lines, [target_ref_lines])
    chrf_details = json.loads(
        chrf_score.format(signature=compute_chrf.get_signature().format(), is_json=True)
    )

    # The default comet model.
    # It should match the model used in https://github.com/mozilla/firefox-translations-models/
    comet_model_name = "Unbabel/wmt22-comet-da"

    if os.environ.get("COMET_SKIP"):
        comet_score = "skipped"
        unaligned_ratio_ref = "skipped"
        unaligned_ratio_dif = "skipped"
        unaligned_ratio_translation = "skipped"
        print("COMET_SKIP was set, so the COMET score will not be computed.")
        print("COMET_SKIP was set, so the unaligned-ratio score will not be computed.")
    else:
        logger.info("Loading COMET")
        import comet

        # COMET_MODEL_DIR allows tests to place the model in a data directory
        comet_checkpoint = comet.download_model(
            comet_model_name, saving_directory=os.environ.get("COMET_MODEL_DIR")
        )
        comet_model = comet.load_from_checkpoint(comet_checkpoint)
        comet_data = []
        for source, target, target_ref in zip(source_lines, target_lines, target_ref_lines):
            comet_data.append({"src": source, "mt": target, "ref": target_ref})
        # GPU information comes in the form of a list of numbers, e.g. "0 1 2 3". Split these to
        # get the GPU count.
        gpu_count = len(args.gpus.split(" "))
        if os.environ.get("COMET_CPU"):
            gpu_count = 0  # Let tests override the CPU count.
        comet_mode = "cpu" if gpu_count == 0 else "gpu"
        logger.info(f'Computing the COMET score with "{comet_model_name}" using the {comet_mode}')

        comet_results = comet_model.predict(comet_data, gpus=gpu_count)
        # Reduce the precision.
        comet_score = round(comet_results.system_score, 4)
        del comet_model

        if gpu_count == 0:
            aligner_device = torch.device("cpu")
        else:
            aligner_device = torch.device(f"cuda:{args.gpus.split()[0]}")
        logger.info("Computing unaliged ratio.")
        unaligned_ratio_translation = compute_unaliged_ratio(
            src, trg, source_lines, target_lines, aligner_device
        )
        unaligned_ratio_ref = compute_unaliged_ratio(
            src, trg, source_lines, target_ref_lines, aligner_device
        )
        unaligned_ratio_dif = unaligned_ratio_translation - unaligned_ratio_ref

    metrics = {
        "bleu": {
            "score": bleu_details["score"],
            # Example details:
            # {
            #     "name": "BLEU",
            #     "score": 0.4,
            #     "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0",
            #     "verbose_score": "15.6/0.3/0.2/0.1 (BP = 0.823 ratio = 0.837 hyp_len = 180 ref_len = 215)",
            #     "nrefs": "1",
            #     "case": "mixed",
            #     "eff": "no",
            #     "tok": "13a",
            #     "smooth": "exp",
            #     "version": "2.0.0"
            # }
            "details": bleu_details,
        },
        "chrf": {
            "score": chrf_details["score"],
            # Example details:
            # {
            #     "name": "chrF2",
            #     "score": 0.64,
            #     "signature": "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.0.0",
            #     "nrefs": "1",
            #     "case": "mixed",
            #     "eff": "yes",
            #     "nc": "6",
            #     "nw": "0",
            #     "space": "no",
            #     "version": "2.0.0"
            # }
            "details": chrf_details,
        },
        "comet": {
            "score": comet_score,
            "details": {
                "model": comet_model_name,
                "score": comet_score,
            },
        },
        "unaligned-ratio": {
            "score": unaligned_ratio_dif,
            "details": {
                "unaligned_ratio_translation": unaligned_ratio_translation,
                "unaligned_ratio_ref": unaligned_ratio_ref,
            },
        },
    }

    logger.info(f"Writing {metrics_json}")
    with open(metrics_json, "w") as file:
        file.write(json.dumps(metrics, indent=2))

    logger.info(f'Writing the metrics in the older "text" format: {metrics_file}')
    with open(metrics_file, "w") as file:
        file.write(f"{bleu_details['score']}\n" f"{chrf_details['score']}\n" f"{comet_score}\n")

    if WANDB_AVAILABLE:
        metric = metric_from_tc_context(
            chrf=chrf_details["score"], bleu=bleu_details["score"], comet=comet_score
        )

        run_client = get_wandb_publisher(  # noqa
            project_name=args.wandb_project,
            group_name=args.wandb_group,
            run_name=args.wandb_run_name,
            taskcluster_secret=args.taskcluster_secret,
            artifacts=args.wandb_artifacts,
            publication=args.wandb_publication,
        )
        if run_client is None:
            # W&B publication may be direclty disabled through WANDB_PUBLICATION
            return

        logger.info(f"Publishing metrics to Weight & Biases ({run_client.extra_kwargs})")
        run_client.open()
        run_client.handle_metrics(metrics=[metric])
        run_client.close()

        # Publish an extra row on the group_logs summary run
        group_logs_client = WandB(  # noqa
            project=run_client.wandb.project,
            group=run_client.wandb.group,
            name="group_logs",
            suffix=run_client.suffix,
        )
        logger.info("Adding metric row to the 'group_logs' run")
        group_logs_client.open()

        # Restore existing metrics data
        data = list_existing_group_logs_metrics(group_logs_client.wandb)
        data.append(
            [
                run_client.wandb.group,
                run_client.wandb.name,
                metric.importer,
                metric.dataset,
                metric.augmentation,
            ]
            + [getattr(metric, attr) for attr in METRIC_KEYS]
        )
        group_logs_client.wandb.log(
            {
                "metrics": wandb.Table(
                    columns=[
                        "Group",
                        "Model",
                        "Importer",
                        "Dataset",
                        "Augmenation",
                        *METRIC_KEYS,
                    ],
                    data=data,
                )
            }
        )
        group_logs_client.close()


if __name__ == "__main__":
    main()
