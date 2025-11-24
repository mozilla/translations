import json
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
import time
from pathlib import Path
from statistics import mean
from typing import Any, List, Iterable

import sacrebleu.metrics.base
import toolz
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF

from pipeline.common.logging import get_logger
from pipeline.eval.langs import COMET22_SUPPORT, METRICX24_SUPPORT

logger = get_logger(__file__)


@dataclass
class MetricResults:
    name: str
    segment_scores: list[Any]
    corpus_score: float
    details: dict[str, Any]


class Metric:
    name = None

    @staticmethod
    def supports_lang(src_lang: str, trg_lang: str) -> bool:
        ...


class RegularMetric(Metric):
    def score(
        self,
        src_lang: str,
        trg_lang: str,
        source_texts: list[str],
        translated_texts: list[str],
        reference_texts: list[str],
    ) -> MetricResults:
        ...


class ReferencelessMetric(Metric):
    def score_qe(
        self, src_lang: str, trg_lang: str, source_texts: list[str], translated_texts: list[str]
    ) -> MetricResults:
        ...


class SacrebleuMetric(RegularMetric):
    name = None

    def __init__(self):
        self.metric: sacrebleu.metrics.base.Metric = None

    @staticmethod
    def supports_lang(src_lang: str, trg_lang: str) -> bool:
        return True

    def score(
        self,
        src_lang: str,
        trg_lang: str,
        source_texts: list[str],
        translated_texts: list[str],
        reference_texts: list[str],
    ) -> MetricResults:
        corpus_score = self.metric.corpus_score(translated_texts, [reference_texts])
        segment_scores = [
            self.metric.sentence_score(tr, [ref]).score
            for tr, ref in zip(translated_texts, reference_texts)
        ]

        return MetricResults(
            name=self.name,
            corpus_score=corpus_score.score,
            segment_scores=segment_scores,
            # for compatibility with eval.py
            details=json.loads(
                corpus_score.format(signature=self.metric.get_signature().format(), is_json=True)
            ),
        )


class Chrf(SacrebleuMetric):
    name = "chrf"

    def __init__(self):
        super().__init__()
        self.metric = CHRF()


class Chrfpp(SacrebleuMetric):
    name = "chrfpp"

    def __init__(self):
        super().__init__()
        self.metric = CHRF(word_order=2)


class Bleu(SacrebleuMetric):
    name = "bleu"

    def __init__(self):
        # todo: double check whats'up with the CJK tokenizers
        super().__init__()
        # it is recommended to enable effective_order for sentence-level scores
        self.metric = BLEU(effective_order=True)


class Comet22(RegularMetric):
    name = "comet22"

    def __init__(self):
        super().__init__()
        import comet
        import torch

        self.gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        # COMET_MODEL_DIR allows tests to place the model in a data directory
        self.hf_name = "Unbabel/wmt22-comet-da"
        comet_checkpoint = comet.download_model(
            self.hf_name,
        )
        self.comet_model = comet.load_from_checkpoint(comet_checkpoint)

    @staticmethod
    def supports_lang(src_lang: str, trg_lang: str) -> bool:
        if src_lang not in COMET22_SUPPORT or trg_lang not in COMET22_SUPPORT:
            return False
        return True

    def score(
        self,
        src_lang: str,
        trg_lang: str,
        source_texts: list[str],
        translated_texts: list[str],
        reference_texts: list[str],
    ) -> MetricResults:
        comet_data = []
        for source, target, target_ref in zip(source_texts, translated_texts, reference_texts):
            comet_data.append({"src": source, "mt": target, "ref": target_ref})

        comet_results = self.comet_model.predict(comet_data, gpus=self.gpus)

        return MetricResults(
            name=self.name,
            corpus_score=100 * comet_results.system_score,
            segment_scores=[100 * s for s in comet_results.scores],
            details={"model": self.hf_name},
        )


class MetricX24(RegularMetric):
    name = "metricx24"

    def __init__(self, model_size: str = "xl", batch_size=8):
        super().__init__()
        import os

        os.environ["WANDB_DISABLED"] = "true"

        from metricx import MT5ForRegression
        import torch
        import transformers

        self.hf_name = f"google/metricx-24-hybrid-{model_size}-v2p6"

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.per_device_batch_size = batch_size // torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.per_device_batch_size = batch_size

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f"google/mt5-{model_size}")
        self.model = MT5ForRegression.from_pretrained(self.hf_name, torch_dtype="auto")
        self.model.to(self.device)
        self.model.eval()
        self.max_input_length = 1536

    @staticmethod
    def supports_lang(src_lang: str, trg_lang: str) -> bool:
        if src_lang not in METRICX24_SUPPORT or trg_lang not in METRICX24_SUPPORT:
            return False
        return True

    def score(
        self,
        src_lang: str,
        trg_lang: str,
        source_texts: list[str],
        translated_texts: list[str],
        reference_texts: list[str],
    ) -> MetricResults:
        scores = self._predict(source_texts, translated_texts, reference_texts)
        return MetricResults(
            name=self.name,
            corpus_score=mean(scores),
            segment_scores=scores,
            details={"model": self.hf_name},
        )

    def _get_dataset(
        self,
        source_texts: list[str],
        translated_texts: list[str],
        reference_texts: list[str] | None,
    ):
        import datasets
        from transformers.data.data_collator import DataCollatorWithPadding

        def _make_input(example):
            if reference_texts is None:
                # Use Quality Estimation mode
                example["input"] = (
                    "source: " + example["source"] + " candidate: " + example["hypothesis"]
                )
            else:
                example["input"] = (
                    "source: "
                    + example["source"]
                    + " candidate: "
                    + example["hypothesis"]
                    + " reference: "
                    + example["reference"]
                )
            return example

        def _tokenize(example):
            return self.tokenizer(
                example["input"],
                max_length=self.max_input_length,
                truncation=True,
                padding=False,
            )

        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example

        data_dict = {"source": source_texts, "hypothesis": translated_texts}
        if reference_texts is not None:
            data_dict["reference"] = reference_texts

        ds = datasets.Dataset.from_dict(data_dict)
        ds = ds.map(_make_input)
        ds = ds.map(_tokenize, batched=True)
        ds = ds.map(_remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=self.device,
            output_all_columns=True,
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
        return ds, data_collator

    def _predict(
        self,
        source_texts: list[str],
        translated_texts: list[str],
        reference_texts: list[str] | None,
    ) -> list[float]:
        import os

        import transformers

        ds, datacollator = self._get_dataset(
            source_texts,
            translated_texts,
            reference_texts,
        )

        training_args = transformers.TrainingArguments(
            output_dir=os.getcwd(),
            per_device_eval_batch_size=self.per_device_batch_size,
            dataloader_pin_memory=False,
        )
        trainer = transformers.Trainer(
            model=self.model, args=training_args, data_collator=datacollator
        )
        predictions, _, _ = trainer.predict(test_dataset=ds)

        return [float(pred) for pred in predictions]


class Metricx24Qe(MetricX24, ReferencelessMetric):
    name = "metricx24-qe"

    def score(
        self,
        src_lang: str,
        trg_lang: str,
        source_texts: list[str],
        translated_texts: list[str],
        reference_texts: list[str],
    ) -> MetricResults:
        return self.score_qe(src_lang, trg_lang, source_texts, translated_texts)

    def score_qe(
        self, src_lang: str, trg_lang: str, source_texts: list[str], translated_texts: list[str]
    ) -> MetricResults:
        scores = self._predict(source_texts, translated_texts, None)
        return MetricResults(
            name=self.name,
            corpus_score=mean(scores),
            segment_scores=scores,
            details={"model": self.hf_name},
        )


class UnalignedRatio(RegularMetric):
    name = "unaligned-ratio"

    @staticmethod
    def supports_lang(src_lang: str, trg_lang: str) -> bool:
        return True

    def score(
        self,
        src_lang: str,
        trg_lang: str,
        source_texts: list[str],
        translated_texts: list[str],
        reference_texts: list[str],
    ):
        unaligned_ratio_translation = self._compute_unaliged_ratio(
            src_lang, trg_lang, source_texts, translated_texts
        )
        unaligned_ratio_ref = self._compute_unaliged_ratio(
            src_lang, trg_lang, source_texts, reference_texts
        )
        unaligned_ratio_dif = unaligned_ratio_translation - unaligned_ratio_ref
        return MetricResults(
            name=self.name,
            corpus_score=unaligned_ratio_dif,
            segment_scores=[],
            details={
                "unaligned_ratio_translation": unaligned_ratio_translation,
                "unaligned_ratio_ref": unaligned_ratio_ref,
            },
        )

    def _compute_unaliged_ratio(
        self, src: str, trg: str, source_lines: List[str], target_lines: List[str]
    ):
        from simalign import SentenceAligner
        import torch
        from pipeline.alignments.tokenizer import IcuTokenizer

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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


class LlmRef(RegularMetric):
    name = "llm-ref"

    def __init__(self, is_mini: bool = True):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # https://platform.openai.com/docs/pricing
        if is_mini:
            self.input_cost = 0.15 / 1_000_000
            self.output_cost = 0.60 / 1_000_000
            self.model = "gpt-4o-mini"
        else:
            self.input_cost = 2.50 / 1_000_000
            self.output_cost = 10.00 / 1_000_000
            self.model = "gpt-4o"
        self.api_batch_size = 10
        self.max_count = None

        self.debug = True
        self.debug_prefix = Path("llm_errors")
        if self.debug:
            self.max_count = 2
            self.api_batch_size = 2

    @staticmethod
    def supports_lang(src_lang: str, trg_lang: str) -> bool:
        return True

    def score(
        self,
        src_lang: str,
        trg_lang: str,
        source_texts: list[str],
        translated_texts: list[str],
        reference_texts: list[str],
    ) -> MetricResults:
        with open(Path(__file__).parent / "eval-batch-instructions.md", "r") as file:
            eval_batch_instructions = file.read().format(src=src_lang, trg=trg_lang)
        with open(Path(__file__).parent / "eval-final-summary.md", "r") as file:
            eval_final_summary = file.read().format(src=src_lang, trg=trg_lang)

        score_results = []
        summaries: list[dict] = []
        input_tokens = 0
        output_tokens = 0
        for batch_index, translations in enumerate(
            self._yield_batched_translations(source_texts, translated_texts, reference_texts)
        ):
            eval_batch, usages = self.run_eval_batch_prompt(
                translations, eval_batch_instructions, batch_index
            )

            for usage in usages:
                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens

            if eval_batch:
                summaries.append(eval_batch["summary"])
                scores_list = eval_batch["scores"]
                for i, translation in enumerate(translations):
                    scores = scores_list[i] if i < len(scores_list) else []
                    score_results.append(scores)

        summary, usage = self.run_eval_final_summary(eval_final_summary, summaries)
        if usage:
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens

        logger.info("Summary of API calls")
        logger.info(f" ├─ Input tokens: {input_tokens}")
        logger.info(f" ├─ Output tokens: {output_tokens}")
        logger.info(f" ├─ Input cost: ${self.input_cost * input_tokens:.2f}")
        logger.info(f" └─ Output cost: ${self.output_cost * output_tokens:.2f}")

        totals = defaultdict(float)
        for res in score_results:
            for criteria, score in res.items():
                totals[criteria] += float(score[0])
        avg_scores = {k: totals[k] / len(score_results) for k in totals}

        return MetricResults(
            name=self.name,
            corpus_score=statistics.mean(avg_scores.values()),
            segment_scores=score_results,
            details={
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "summary": summary,
                "scores": avg_scores,
            },
        )

    def run_eval_batch_prompt(
        self,
        translations: list[dict[str, str]],
        instructions: str,
        batch_index: int,
    ):
        import json5

        start = time.time()
        input = self._translations_batch_to_text(translations)
        retry_count = 5

        # Attempt to parse the JSON evaluations.
        eval_batch: dict | None = None
        output_text = ""
        output_text_raw = ""

        usages = []

        for attempt in range(retry_count):
            if attempt == 0:
                # Start with a stable temperature to have consistent results.
                temperature = 0.0
                logger.info(
                    f"Querying {self.model} with batch {batch_index} containing {len(translations)} translations."
                )
            else:
                # Increase the temperature so that the results will be varied on the second
                # attempt.
                temperature = 0.5
                logger.info(f"Retry {attempt+1}/{retry_count}: The last query failed to parse.")

            response = self.client.responses.create(
                model=self.model,
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
                logger.error(f"Failed to decode the scores for batch: {err}")
                if self.debug:
                    debug_file = self.debug_prefix / f"eval-error.{batch_index}.{attempt}.txt"
                    self._write_debug_file(
                        debug_file, instructions, input, output_text_raw, output_text, eval_batch
                    )
                raise

        return eval_batch, usages

    def run_eval_final_summary(self, instructions: str, summaries: list[dict]):
        start = time.time()
        logger.info(f"Querying {self.model} for the final evaluation summary.")
        input = json.dumps(summaries, indent=2)
        response = self.client.responses.create(
            model=self.model,
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
            debug_file = self.debug_prefix / "summary-error.txt"
            logger.error(f"Failed to decode the scores for batch: {err} See {debug_file}")
            if self.debug:
                self._write_debug_file(
                    debug_file, instructions, input, output_text_raw, output_text, None
                )
            raise

        usage = response.usage
        if usage:
            logger.info(f" ├─ Input tokens: {usage.input_tokens}")
            logger.info(f" ├─ Output tokens: {usage.output_tokens}")
        logger.info(f" └─ Query took {call_time:.2f} seconds")

        return summary, usage

    def _yield_batched_translations(
        self, source_texts: list[str], translated_texts: list[str], reference_texts: list[str]
    ) -> Iterable[list[dict[str, str]]]:
        i = 0
        for batch in toolz.partition_all(
            self.api_batch_size, zip(source_texts, translated_texts, reference_texts)
        ):
            yield [
                {
                    "src": src_line.strip(),
                    "trg": trg_line.strip(),
                    "ref": ref_line.strip(),
                }
                for src_line, trg_line, ref_line in batch
            ]

            i += len(batch)
            if self.max_count and i >= self.max_count:
                break

    def _translations_batch_to_text(self, translations: list[dict[str, str]]) -> str:
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

    def _write_debug_file(
        self,
        debug_file: Path,
        instructions: str,
        input: str,
        output_text_raw: str,
        output_text: str,
        eval_batch: dict,
    ):
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
