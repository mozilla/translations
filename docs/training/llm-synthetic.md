# LLM generated data

The goal is to be able to produce high quality parallel translation datasets with LLMs.
This allows finetuning NMT models to improve quality. 
It can potentially replace teacher training stage by using the LLM-produced data directly.

This work follows the paper [Introducing the NewsPaLM MBR and QE Dataset:
LLM-Generated High-Quality Parallel Data Outperforms Traditional
Web-Crawled Data](https://arxiv.org/pdf/2408.06537).

It also uses the evaluation dataset and the prompt from [WMT24++: Expanding the Language Coverage of WMT24 to 55 Languages & Dialects](https://arxiv.org/html/2502.12404v1).

See also [this presentation](https://docs.google.com/presentation/d/1LtGxVbhmI4O08kGTjUp1FzmdckOZFD2GaF7zA953yjg/edit?slide=id.g82d2da351e_5_3803#slide=id.g82d2da351e_5_3803) for experimental results.

The generation pipeline is still considered experimental and is not fully automated. It is located at [experiments/llmaat/](https://github.com/mozilla/translations/tree/main/experiments/llmaat/) (LLM-as-a-teacher).

## Selecting a corpus

The idea is to have a diverse monolingual dataset to translate by an LLM. 
We use pre-trained text embeddings to get vectors of text segments and then cluster them and sample uniformly form clusters to get a diverse dataset.

It's more efficient to cluster a sample first and then assign clusters based on centroids.

This part is not fully automated.

Steps:

1. Find a big monolingual corpus (100+M sentences). It can be a part of HPLT and NewsCrawl or just one side of our typical merged parallel corpus. It should be deduplicated.
2. Sample a part of it using `shuf -n 1000000`
3. Calculate and save embeddings for the sample (see [experiments/llmaat/notebooks/Select corpus.ipynb](https://github.com/mozilla/translations/tree/main/experiments/llmaat/notebooks/Select corpus.ipynb). 
We use https://huggingface.co/intfloat/multilingual-e5-small. To speed it up and utilize all GPUs on a machine, we split the sample with `split` and 
run [experiments/llmaat/scripts/emb_corpus_ddp.py](https://github.com/mozilla/translations/tree/main/experiments/llmaat/scripts/emb_corpus_ddp.py) with `torchrun --nproc_per_node=8 emb_corpus_ddp.py`
4. Load the embeddings, cluster them with K-Means (5000 clusters) and save centroids to a file
5. Go through the whole corpus and assign clusters based on the closest centroids by doing a NN search. Run `torchrun --nproc_per_node=8 cluster_corpus_ddp.py`, the cluster IDs are saved to a file.
6. Select 1M, 10M and 50M lines by sampling uniformly from the clusters.

The diverse samples for English are located here: `gs://releng-translations-dev/data/mono-llm/diverse_sample.{1,10,50}M.en.zst` 

## Evaluating LLMs

Run [experiments/llmaat/flows/llm_eval_flow.py](https://github.com/mozilla/translations/tree/main/experiments/llmaat/flows/llm_eval_flow.py) on Mozilla Outerbounds Metaflow:

```bash
export HUGGING_FACE_HUB_TOKEN=...
export WANDB_API_KEY=...
python llm_eval_flow.py --environment=pypi --config config ./configs/config.vllm.greedy.json run --experiment greedy --model gemma-3-27b-vllm
```

The evaluation results are available on Weights and Biases: https://wandb.ai/moz-translations/llm-evals

It's possible to add more LLMs and inference methods to [experiments/llmaat/flows/llm_runner.py](https://github.com/mozilla/translations/tree/main/experiments/llmaat/flows/llm_runner.py). `--model gemma-3-27b-vllm` points to one of the available implementations.

Decoding config can be modified in [experiments/llmaat/flows/configs/](https://github.com/mozilla/translations/tree/main/experiments/llmaat/flows/configs/).

The prompt can be set in the config. Available prompt templates are in [experiments/llmaat/flows/prompts.py](https://github.com/mozilla/translations/tree/main/experiments/llmaat/flows/prompts.py).

It allows running evaluation for multiple language pairs in one run by adding more languages to the config. All pairs are en-xx.

The translation produced by an LLM during evaluation are uploaded to `gs://releng-translations-dev/data/llm-evals/wmt24pp/`.

We caluculate COMET22 and MetricX-24 scores. The size of the MetricX model is set in the step `eval_metricx`.

It's preferable to use vLLM as it has up to 10x higher throughput than the naive inference with HF Transformers.

vLLM config:

```python
{
  "batch_size": 1024, # Should be big enough to get the most of the vLLM optimizations
  "langs": ["ru_RU"], # Languages to evaluate
  "max_tok_alpha": 2.0, # A factor to multiply the number of imput tokens to get the maximum number of output tokens. It might depend on the output language. An optimization.
  "prompt": "noomit_fewshot", # Prompt template key
  "llm": {
    "max_model_len": 1024, # The model context size (maximum total of input and output tokens)
    "tensor_parallel_size": 1 # The number of GPUs
  },
  "decoding": {
    "temperature": 0, # Tempreture 0 means greedy decoding, change to activate sampling
    "n": 1 # Produce only 1 candidate, increase for QE reranking
  }
}
```

## Generating datasets

Run [experiments/llmaat/flows/llm_run_flow.py](https://github.com/mozilla/translations/tree/main/experiments/llmaat/flows/llm_run_flow.py) on Mozilla Outerbounds Metaflow:

```bash
export HUGGING_FACE_HUB_TOKEN=...
python llm_run_flow.py \
    --environment=pypi --config config ./configs/config.vllm.greedy.json run --experiment finetune10M \
    --model gemma-3-27b-vllm --data_size 10 --lang ru_RU --part_size 500000 --max-workers 4
```

`--data_size 10` - use 10M dataset to produce 10M translations

`--part_size 500000` - how many lines to process in one Metaflow task

`--max-workers 4` - run 4 tasks max simultaniously (current limitation on the number of GPUs)

The translations will be uploaded to `gs://releng-translations-dev/data/llm/`.

## Quality aware decoding (QE reranking)

Following the NewsPALM paper it's possible to replace regular greedy decoding with sampling of multiple candidates and choosing the best one using MetricX-24-Hybrid quality estimation model.

It required activating the code branch with `pick_best` metaflow step and changing the decoding config (for vllm `decoding.n` > 1, e.g. `decoding.n: 32`).
Decoding will become significantly slower as the model needs to generate N samples instead of one now.

Also, the activated `pick_best` step that runs MetricX model is unoptimized and quite slow now.

## Language codes

We use WMT24++ format of the language codes that include a reference to a country because some prompts require specifying it.

See all available codes in [experiments/llmaat/flows/langs.py](https://github.com/mozilla/translations/tree/main/experiments/llmaat/flows/langs.py)

## Filtering

The generated data is imperfect and can include hallucinations. We filter it for finetuning based on MetricX-QE scores used for QE-reranking.
This part is not included in Metaflow pipeline. We inspect the distribution of the scores and texts manually and choose a threthold to make sure we don't filter too much.
[Example notebook](https://github.com/mozilla/translations/tree/main/experiments/llmaat/notebooks/Filter generated corpus.ipynb) (requires downloading data manually from GCP and then uploading the filtered data back).

Typically, we use threshold `2` and end up with 7-8 million segments out of 10.

## Fine-tuning

We use a pre-trained student model and finetune it on the LLM generated corpus directly. 
See [this PR](https://github.com/mozilla/translations/pull/1218). The continuation config looks like:
```yaml
continuation:
  models:
    student:
      url: https://firefox-ci-tc.services.mozilla.com/api/queue/v1/task/JW1UXdmzSoW1CFjUgfD74w/artifacts/public/build
      mode: init
      type: default
  vocab:
    src: https://firefox-ci-tc.services.mozilla.com/api/queue/v1/task/JW1UXdmzSoW1CFjUgfD74w/runs/0/artifacts/public%2Fbuild%2Fvocab.en.spm
    trg: https://firefox-ci-tc.services.mozilla.com/api/queue/v1/task/JW1UXdmzSoW1CFjUgfD74w/runs/0/artifacts/public%2Fbuild%2Fvocab.zh.spm
  corpora:
    distillation:
      src: https://storage.googleapis.com/releng-translations-dev/data/llm/en-zh_TW/qwen-3-235b-a22b-fp8-vllm/qererank8/diverse_sample.10M.filtered2.en.zst
      trg: https://storage.googleapis.com/releng-translations-dev/data/llm/en-zh_TW/qwen-3-235b-a22b-fp8-vllm/qererank8/diverse_sample.10M.filtered2.zh.zst
```


