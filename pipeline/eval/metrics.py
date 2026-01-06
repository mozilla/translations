import json
import logging
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
import time
from pathlib import Path
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from itertools import islice
from typing import Any, List, Iterable

from pydantic import BaseModel

from pipeline.common.logging import get_logger
from pipeline.eval.langs import COMET22_SUPPORT, METRICX24_SUPPORT, GOOGLE_DEFAULTS_MAP
from pipeline.eval.translators import adjust_codes

logger = get_logger(__file__)
logger.setLevel(logging.INFO)


class ScoreItem(BaseModel):
    score: int
    explanation: str


class TranslationScore(BaseModel):
    adequacy: ScoreItem
    fluency: ScoreItem
    terminology: ScoreItem
    hallucination: ScoreItem
    punctuation: ScoreItem


class BatchSummary(BaseModel):
    adequacy: str
    fluency: str
    terminology: str
    hallucination: str
    punctuation: str


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


class SubprocessMetric(RegularMetric):
    """Wrapper that runs any metric class in an isolated subprocess with a custom venv."""

    def __init__(
        self,
        metric_cls: type[RegularMetric],
        venv_path: Path,
        requirements_path: Path,
        **metric_kwargs,
    ):
        self._metric_cls = metric_cls
        self._metric_kwargs = metric_kwargs
        self._venv_path = venv_path
        self._requirements_path = requirements_path
        self._process = None

        self._ensure_venv()
        self._start_worker()

    @property
    def name(self):
        return self._metric_cls.name

    def _ensure_venv(self):
        import subprocess
        import sys

        if (self._venv_path / "bin" / "python").exists():
            logging.debug(f"venv already installed in {self._venv_path}, skipping venv setup")
            return
        logger.info(f"Creating venv at {self._venv_path}...")
        subprocess.run([sys.executable, "-m", "venv", str(self._venv_path)], check=True)
        subprocess.run(
            [str(self._venv_path / "bin" / "pip"), "install", "-r", str(self._requirements_path)],
            check=True,
        )

    def _start_worker(self):
        import subprocess

        logger.info(f"Starting {self._metric_cls.name} worker process...")
        env = os.environ.copy()
        pythonpath = os.getcwd()
        if env.get("PYTHONPATH"):
            pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath
        self._process = subprocess.Popen(
            [str(self._venv_path / "bin" / "python"), "-m", "pipeline.eval.metric_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # Stream to parent's stderr for real-time logs
            text=True,
            env=env,
        )
        init_msg = {
            "cmd": "init",
            "module": self._metric_cls.__module__,
            "class": self._metric_cls.__name__,
            "kwargs": self._metric_kwargs,
        }
        self._send(init_msg)
        response = self._recv()
        if response.get("status") != "ready":
            raise RuntimeError(f"Worker failed to initialize: {response}")
        logger.info(f"{self._metric_cls.name} worker ready")

    def _send(self, msg: dict):
        self._process.stdin.write(json.dumps(msg) + "\n")
        self._process.stdin.flush()

    def _recv(self, timeout: int = 600) -> dict:
        import select

        ready, _, _ = select.select([self._process.stdout], [], [], timeout)
        if not ready:
            stderr = self._process.stderr.read() if self._process.stderr else None
            self._process.kill()
            raise RuntimeError(f"Worker timed out after {timeout}s. stderr: {stderr}")
        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else None
            exit_code = self._process.poll()
            # -9 = SIGKILL (often OOM killer), -6 = SIGABRT, -11 = SIGSEGV
            signal_info = ""
            if exit_code is not None and exit_code < 0:
                signal_num = -exit_code
                signal_names = {9: "SIGKILL (likely OOM)", 6: "SIGABRT", 11: "SIGSEGV"}
                signal_info = f" Signal: {signal_names.get(signal_num, signal_num)}"
            raise RuntimeError(f"Worker died (exit={exit_code}).{signal_info} stderr: {stderr}")
        return json.loads(line)

    def __del__(self):
        if not self._process or self._process.poll() is not None:
            return
        try:
            self._send({"cmd": "shutdown"})
            self._process.stdin.close()
            self._process.wait(timeout=5)
        except Exception:
            self._process.kill()
            self._process.wait()

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
        request = {
            "cmd": "score",
            "src_lang": src_lang,
            "trg_lang": trg_lang,
            "source_texts": source_texts,
            "translated_texts": translated_texts,
            "reference_texts": reference_texts,
        }
        self._send(request)
        response = self._recv()

        if response.get("status") != "ok":
            raise RuntimeError(f"Worker error: {response.get('error')}")

        return MetricResults(
            name=response["name"],
            corpus_score=response["corpus_score"],
            segment_scores=response["segment_scores"],
            details=response["details"],
        )


class SacrebleuMetric(RegularMetric):
    name = None

    def __init__(self):
        import sacrebleu.metrics.base

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
        from sacrebleu.metrics.chrf import CHRF

        self.metric = CHRF()


class Chrfpp(SacrebleuMetric):
    name = "chrfpp"

    def __init__(self):
        super().__init__()
        from sacrebleu.metrics.chrf import CHRF

        self.metric = CHRF(word_order=2)


class Bleu(SacrebleuMetric):
    name = "bleu"

    @staticmethod
    def supports_lang(src_lang: str, trg_lang: str) -> bool:
        # requires using special tokenizers, skip, spBLEU is sufficient
        if len({src_lang, trg_lang} & {"zh", "zt", "ja", "ko"}) > 0:
            return False
        return True

    def __init__(self):
        super().__init__()
        from sacrebleu.metrics.bleu import BLEU

        # it is recommended to enable effective_order for sentence-level scores
        self.metric = BLEU(effective_order=True)


class SpBleu(SacrebleuMetric):
    name = "spbleu"

    def __init__(self):
        super().__init__()
        from sacrebleu.metrics.bleu import BLEU

        # it is recommended to enable effective_order for sentence-level scores
        self.metric = BLEU(effective_order=True, tokenize="flores200")


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
            corpus_score=comet_results.system_score,
            segment_scores=comet_results.scores,
            details={"model": self.hf_name},
        )


class MetricX24(RegularMetric):
    name = "metricx24"

    def __init__(self, model_size: str = "large", batch_size=4, fp16=True):
        super().__init__()
        os.environ["WANDB_DISABLED"] = "true"

        from pipeline.eval.metricx import MT5ForRegression
        import torch
        import transformers

        self.hf_name = f"google/metricx-24-hybrid-{model_size}-v2p6"
        if fp16:
            self.hf_name += "-bfloat16"

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
        import transformers

        ds, datacollator = self._get_dataset(source_texts, translated_texts, reference_texts)

        training_args = transformers.TrainingArguments(
            output_dir=os.getcwd(),
            per_device_eval_batch_size=self.per_device_batch_size,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
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
    CRITERIA = ("adequacy", "fluency", "terminology", "hallucination", "punctuation")

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
        self.max_parallel = 100
        self.max_count = None

        with open(Path(__file__).parent / "eval-batch-instructions.md", "r") as file:
            self.eval_batch_instructions = file.read()
        with open(Path(__file__).parent / "eval-final-summary.md", "r") as file:
            self.eval_final_summary = file.read()

        self.errors_dir = Path("data/llm_errors")

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
        src_lang, trg_lang = adjust_codes(src_lang, trg_lang, GOOGLE_DEFAULTS_MAP)
        eval_batch_instructions = self.eval_batch_instructions.format(src=src_lang, trg=trg_lang)
        eval_final_summary = self.eval_final_summary.format(src=src_lang, trg=trg_lang)

        batches = list(
            enumerate(
                self._yield_batched_translations(source_texts, translated_texts, reference_texts)
            )
        )
        logger.info(
            f"Processing {len(batches)} batches of max size {self.api_batch_size} in parallel (max {self.max_parallel} workers)"
        )

        results: dict[int, tuple[tuple[list[TranslationScore], BatchSummary], list]] = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {
                executor.submit(
                    self.run_eval_batch_prompt, translations, eval_batch_instructions, batch_index
                ): batch_index
                for batch_index, translations in batches
            }
            for future in as_completed(futures):
                batch_index = futures[future]
                eval_batch, usages = future.result()
                results[batch_index] = (eval_batch, usages)

        score_results: list[TranslationScore] = []
        summaries: list[BatchSummary] = []
        input_tokens = 0
        output_tokens = 0
        for batch_index, translations in batches:
            eval_batch, usages = results[batch_index]
            for usage in usages:
                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens
            if eval_batch:
                scores, batch_summary = eval_batch
                summaries.append(batch_summary)
                score_results.extend(scores)

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
            for criteria in self.CRITERIA:
                totals[criteria] += getattr(res, criteria).score
        avg_scores = (
            {k: round(totals[k] / len(score_results), 2) for k in totals} if score_results else {}
        )

        return MetricResults(
            name=self.name,
            corpus_score=statistics.mean(avg_scores.values()) if avg_scores else 0.0,
            segment_scores=[s.model_dump() for s in score_results],
            details={
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "summary": summary.model_dump() if summary else None,
                "scores": avg_scores,
            },
        )

    def run_eval_batch_prompt(
        self,
        translations: list[dict[str, str]],
        instructions: str,
        batch_index: int,
        max_retries: int = 3,
    ) -> tuple[tuple[list[TranslationScore], BatchSummary] | None, list]:
        input_text = self._translations_batch_to_text(translations)
        batch_size = len(translations)
        response_model = self._create_batch_response_model(batch_size)
        usages = []
        last_response = None
        last_error = None

        for attempt in range(max_retries):
            start = time.time()
            logger.debug(
                f"Querying {self.model} with batch {batch_index} "
                f"(attempt {attempt + 1}/{max_retries}, {batch_size} translations)"
            )

            response = self.client.responses.parse(
                model=self.model,
                store=False,
                instructions=instructions,
                input=input_text,
                text_format=response_model,
                temperature=0.0 if attempt == 0 else 0.5,
            )

            call_time = time.time() - start
            usage = response.usage
            if usage:
                usages.append(usage)
                logger.debug(f" ├─ Input tokens: {usage.input_tokens}")
                logger.debug(f" ├─ Output tokens: {usage.output_tokens}")
            logger.debug(f" └─ Query took {call_time:.2f} seconds")

            last_response = response
            parsed = response.output_parsed

            if not parsed:
                last_error = f"Model refused to evaluate batch {batch_index}"
                logger.warning(f"Attempt {attempt + 1}: {last_error}. Retrying...")
                continue

            # Extract scores in order from the dynamic model slots
            scores = [getattr(parsed, f"example_{i}") for i in range(1, batch_size + 1)]
            return (scores, parsed.summary), usages

        logger.error(
            f"Failed to decode the scores for batch after {max_retries} attempts: {last_error}"
        )
        os.makedirs(self.errors_dir, exist_ok=True)
        debug_file = self.errors_dir / f"eval-error.{batch_index}.txt"
        self._write_debug_file(debug_file, instructions, input_text, last_response, None)
        raise ValueError(last_error)

    def run_eval_final_summary(
        self, instructions: str, summaries: list[BatchSummary]
    ) -> tuple[BatchSummary | None, Any]:
        start = time.time()
        logger.debug(f"Querying {self.model} for the final evaluation summary.")
        input_text = json.dumps([s.model_dump() for s in summaries], indent=2)

        response = self.client.responses.parse(
            model=self.model,
            store=False,
            instructions=instructions,
            input=input_text,
            text_format=BatchSummary,
            temperature=0.0,
        )

        call_time = time.time() - start
        summary = response.output_parsed
        if not summary:
            os.makedirs(self.errors_dir, exist_ok=True)
            debug_file = self.errors_dir / "summary-error.txt"
            self._write_debug_file(debug_file, instructions, input_text, response, None)
            raise ValueError(f"Model refused to generate final summary: {response}")

        usage = response.usage
        if usage:
            logger.debug(f" ├─ Input tokens: {usage.input_tokens}")
            logger.debug(f" ├─ Output tokens: {usage.output_tokens}")
        logger.debug(f" └─ Query took {call_time:.2f} seconds")

        return summary, usage

    @staticmethod
    @lru_cache(maxsize=16)
    def _create_batch_response_model(batch_size: int) -> type[BaseModel]:
        """Create a dynamic Pydantic model with specific slots for each example."""
        from pydantic import create_model

        fields = {f"example_{i}": (TranslationScore, ...) for i in range(1, batch_size + 1)}
        fields["summary"] = (BatchSummary, ...)
        return create_model("EvalBatchResponse", **fields)

    def _yield_batched_translations(
        self, source_texts: list[str], translated_texts: list[str], reference_texts: list[str]
    ) -> Iterable[list[dict[str, str]]]:
        import toolz

        pairs = zip(source_texts, translated_texts, reference_texts)
        if self.max_count:
            pairs = islice(pairs, self.max_count)
        for batch in toolz.partition_all(self.api_batch_size, pairs):
            yield [
                {"src": src.strip(), "trg": trg.strip(), "ref": ref.strip()}
                for src, trg, ref in batch
            ]

    @staticmethod
    def _translations_batch_to_text(translations: list[dict[str, str]]) -> str:
        lines = []
        for i, t in enumerate(translations):
            lines.append(
                f"example_{i+1} {{\n\tsrc: {t['src']}\n\ttrg: {t['trg']}\n\tref: {t['ref']}\n}}\n"
            )
        return "".join(lines)

    @staticmethod
    def _write_debug_file(
        debug_file: Path,
        instructions: str,
        input_text: str,
        response,
        eval_batch,
    ):
        logger.error(f"See {debug_file}")

        # Include a byte order mark for the text file so that Taskcluster correctly
        # displays it as UTF-8.
        with debug_file.open("w", encoding="utf-8-sig") as file:
            file.write("=== Instructions ============================================\n")
            file.write(instructions)
            file.write("\n=== Input ===================================================\n")
            file.write(input_text)
            file.write("\n=== Response ================================================\n")
            if hasattr(response, "output_text"):
                file.write(response.output_text or "")
            else:
                file.write(str(response))
            if hasattr(response, "refusal") and response.refusal:
                file.write("\n=== Refusal =================================================\n")
                file.write(response.refusal)
            if eval_batch:
                file.write("\n=== Eval Batch ==============================================\n")
                if hasattr(eval_batch, "model_dump"):
                    json.dump(eval_batch.model_dump(), file, indent=2)
                else:
                    json.dump(eval_batch, file, indent=2)
