"""
Run an LLM to evaluate a repo.
"""

import argparse
import time
import json
import taskcluster
import json5
import os
from pathlib import Path
from typing import Any, Optional
from openai import OpenAI
from pipeline.common.downloads import read_lines
from pipeline.common.logging import get_logger

logger = get_logger(__file__)


class Config:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=__doc__,
            # Preserves whitespace in the help text.
            formatter_class=argparse.RawTextHelpFormatter,
        )

        parser.add_argument(
            "--corpus_src",
            required=True,
            type=str,
            help="The url or path to a source evaluation corpus.",
        )
        parser.add_argument(
            "--corpus_trg",
            required=True,
            type=str,
            help="The url or path to a target evaluation corpus.",
        )
        parser.add_argument(
            "--corpus_ref",
            required=True,
            type=str,
            help="The url or path to a reference evaluation corpus (same language as target).",
        )

        parser.add_argument(
            "--model_service",
            required=True,
            type=str,
            help='Which service is being evaluated, e.g. "mozilla", "google"',
        )
        parser.add_argument(
            "--model_architecture", type=Path, help="The target evaluation corpus."
        )
        parser.add_argument(
            "--model_name",
            type=Path,
            help="The reference evaluation corpus (same language as target).",
        )

        parser.add_argument(
            "--artifacts", required=True, type=Path, help="The path to the artifacts folder"
        )
        parser.add_argument("--src", required=True, type=str, help="The source language")
        parser.add_argument("--trg", required=True, type=str, help="The target language")
        parser.add_argument(
            "--max_count", default=None, type=int, help="The maximum sentences to use"
        )
        parser.add_argument(
            "--api_batch_size", default=10, type=int, help="How many sentences to send in per call"
        )
        parser.add_argument(
            "--mini", action="store_true", help="Use the mini model for faster results"
        )

        args = parser.parse_args()

        self.corpus_src = args.corpus_src
        self.corpus_trg = args.corpus_trg
        self.corpus_ref = args.corpus_ref
        self.artifacts: Path = args.artifacts
        self.src: str = args.src
        self.trg: str = args.trg
        self.mini: str = args.mini
        self.max_count: Optional[int] = args.max_count
        self.api_batch_size: Optional[int] = args.api_batch_size

        # https://platform.openai.com/docs/pricing
        if args.mini:
            self.input_cost = 2.50 / 1_000_000
            self.output_cost = 10.00 / 1_000_000
            self.model = "gpt-4o-mini"
        else:
            self.input_cost = 0.15 / 1_000_000
            self.output_cost = 10.60 / 1_000_000
            self.model = "gpt-4o"


def yield_batched_translations(config: Config):
    translations: list[dict[str, str]] = []

    with (
        read_lines(config.corpus_src) as corpus_src,
        read_lines(config.corpus_trg) as corpus_trg,
        read_lines(config.corpus_ref) as corpus_ref,
    ):
        batch_size = 0
        for i, (src_line, trg_line, ref_line) in enumerate(
            zip(corpus_src, corpus_trg, corpus_ref)
        ):
            if config.max_count and i >= config.max_count:
                break

            translations.append(
                {
                    "src": src_line.strip(),
                    "trg": trg_line.strip(),
                    "ref": ref_line.strip(),
                }
            )

            batch_size += 1
            if batch_size == config.api_batch_size:
                yield translations
                translations = []
                batch_size = 0

        if translations:
            yield translations


def translations_batch_to_text(translations: list[dict[str, str]]) -> str:
    input = ""
    for i, translation in enumerate(translations):
        input += "\n".join(
            [
                # For batching of evaluations, the LLM requires an index, or else it
                # gets lost and doesn't produce the correct number of results.
                f"example {i} {{",
                f"\tsrc: {translation['src']}",
                f"\ttrg: {translation['trg']}",
                f"\tref: {translation['ref']}",
                "}",
                "",
            ]
        )
    return input


def run_eval_batch_prompt(
    client: OpenAI,
    translations: list[dict[str, str]],
    config: Config,
    instructions: str,
    batch_index: int,
):
    start = time.time()
    input = translations_batch_to_text(translations)
    retry_count = 5

    # Attempt to pares the JSON evaluations.
    eval_batch: dict | None = None
    output_text = ""
    output_text_raw = ""

    usages = []

    for attempt in range(retry_count):
        if attempt == 0:
            # Start with a stable temperature to have consistent results.
            temperature = 0.0
            logger.info(
                f"Querying {config.model} with batch {batch_index} containing {len(translations)} translations."
            )
        else:
            # Increase the temperature so that the results will be varied on the second
            # attempt.
            temperature = 0.5
            logger.info(f"Retry {attempt+1}/{retry_count}: The last query failed to parse.")

        response = client.responses.create(
            model=config.model,
            instructions=instructions,
            input=input,
            temperature=temperature,
        )

        call_time = time.time() - start

        usage = response.usage
        if usage:
            usages.append(usage)
            logger.info(f" ├─ Input tokens: {usage.input_tokens}")
            logger.info(f" ├─ Output tokens: {usage.output_tokens}")
        logger.info(f" └─ Query took {call_time:.2f} seconds")

        output_text = response.output_text.strip()
        output_text_raw = output_text

        # Do any string cleanup for LLM output that doesn't quite match our specification.
        if output_text.startswith("```json\n"):
            start = len("```json\n")
            end = len("\n```")
            output_text = output_text[start:-end]

        try:
            # Parse with json5 as the content can have trailing commas which will not parse
            # using the built-in json module.
            eval_batch_any: Any = json5.loads(output_text)
            eval_batch = eval_batch_any
            assert eval_batch, "There is an eval batch."

            translations_len = len(translations)
            scores_returned = len(eval_batch["scores"])
            assert scores_returned == translations_len, (
                "The correct number of results was returned, "
                f"translations: {translations_len} scores: {scores_returned}"
            )
            break
        except Exception as err:
            # When we can't parse the output write out a debug file.
            print("Exception:", err)
            debug_file = config.artifacts / f"eval-error.{batch_index}.{attempt}.txt"
            logger.error("Failed to decode the scores for batch.")
            logger.error(f"See {debug_file}")

            # Include a byte order mark for the text file so that Taskcluster correctly
            # displays it as UTF-8.
            with debug_file.open("w", encoding="utf-8-sig") as file:
                file.write("=== Instructions ============================================\n")
                file.write(instructions)
                file.write("=== Input ===================================================\n")
                file.write(input)
                file.write("=== Output Raw ==============================================\n")
                file.write(output_text_raw)
                file.write("=== Output Cleaned ==========================================\n")
                file.write(output_text)
                if eval_batch:
                    file.write("=== Eval Batch ==========================================\n")
                    json.dump(eval_batch, file, indent=2)
            eval_batch = None

    return eval_batch, usages


def run_eval_final_summary(
    client: OpenAI, config: Config, instructions: str, summaries: list[dict]
):
    start = time.time()
    logger.info(f"Querying {config.model} for the final evaluation summary.")
    input = json.dumps(summaries, indent=2)
    response = client.responses.create(
        model=config.model,
        instructions=instructions,
        input=input,
        temperature=0.0,
    )
    call_time = time.time() - start
    output_text = response.output_text.strip()
    output_text_raw = output_text

    # Do any string cleanup for LLM output that doesn't quite match our specification.
    if output_text.startswith("```json\n"):
        start = len("```json\n")
        end = len("\n```")
        output_text = output_text[start:-end]

    # Attempt to pares the evalulation.
    summary: dict | None = None
    try:
        summary = json.loads(output_text)
    except json.decoder.JSONDecodeError as err:
        # When we can't parse the output write out a debug file.
        debug_file = "summary-error.txt"
        print(err)
        logger.error(f"Failed to decode the scores for batch. See {debug_file}")
        # Include a byte order mark for the text file so that Taskcluster correctly
        # displays it as UTF-8.
        with (config.artifacts / debug_file).open("w", encoding="utf-8-sig") as file:
            file.write("=== Instructions ============================================\n")
            file.write(instructions)
            file.write("=== Input ===================================================\n")
            file.write(input)
            file.write("=== Output Raw ==============================================\n")
            file.write(output_text_raw)
            file.write("=== Output Cleaned ==========================================\n")
            file.write(output_text)

    usage = response.usage
    if usage:
        logger.info(f" ├─ Input tokens: {usage.input_tokens}")
        logger.info(f" ├─ Output tokens: {usage.output_tokens}")
    logger.info(f" └─ Query took {call_time:.2f} seconds")

    return summary, usage


def get_open_ai_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        logger.info("Found the api key from OPENAI_API_KEY.")
        return api_key

    root_url = os.environ.get("TASKCLUSTER_PROXY_URL")

    assert root_url, (
        "The OPENAI_API_KEY environment variable must be set when running locally. "
        "When running in Taskcluster the TASKCLUSTER_PROXY_URL must be set."
    )
    secrets = taskcluster.Secrets({"rootUrl": root_url})

    try:
        response: Any = secrets.get("project/translations/level-1/chatgpt")
        return response["secret"]["token"]
    except Exception as e:
        raise Exception(f"Could not retrieve the OpenAI secret key: {e}")


def main() -> None:
    config = Config()

    client = OpenAI(api_key=get_open_ai_key())

    config.artifacts.mkdir(exist_ok=True)

    logger.info("Load in the evaluation instructions")
    with open(Path(__file__).parent / "eval-batch-instructions.md", "r") as file:
        eval_batch_instructions = file.read().format(src=config.src, trg=config.trg)
    with open(Path(__file__).parent / "eval-final-summary.md", "r") as file:
        eval_final_summary = file.read().format(src=config.src, trg=config.trg)

    score_results: list[dict] = []
    summaries: list[dict] = []
    input_tokens = 0
    output_tokens = 0
    for batch_index, translations in enumerate(yield_batched_translations(config)):
        eval_batch, usages = run_eval_batch_prompt(
            client, translations, config, eval_batch_instructions, batch_index
        )

        for usage in usages:
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens

        if eval_batch:
            summaries.append(eval_batch["summary"])
            scores_list = eval_batch["scores"]
            for i, translation in enumerate(translations):
                scores = scores_list[i] if i < len(scores_list) else None
                score_results.append(
                    {
                        "translation": translation,
                        "scores": scores,
                    }
                )

    scores_path = config.artifacts / "scores.json"
    logger.info(f"Outputing the scores to {scores_path}")
    with scores_path.open("w") as outfile:
        json.dump(score_results, outfile, ensure_ascii=False, indent=2)

    summary, usage = run_eval_final_summary(client, config, eval_final_summary, summaries)
    if usage:
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens

    summary_path = config.artifacts / "summary.json"
    logger.info(f"Outputing the summary to {summary_path}")
    with summary_path.open("w") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)

    logger.info("Summary of API calls")
    logger.info(f" ├─ Input tokens: {input_tokens}")
    logger.info(f" ├─ Output tokens: {output_tokens}")
    logger.info(f" ├─ Input cost: ${config.input_cost * input_tokens:.2f}")
    logger.info(f" └─ Output cost: ${config.output_cost * output_tokens:.2f}")


if __name__ == "__main__":
    main()
