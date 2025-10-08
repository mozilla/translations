# Final Evaluations

After models are trained, final evaluations can be triggered.

Run an evaluation:

```sh
task eval -- --config taskcluster/configs/eval.yml
```

Make sure and update the `eval.yml` file for your particular model. The evals will be logged to `trigger-eval.log` and uploaded to the bucket specified.

## LLM Evaluation

An LLM can provide an evaluation using the OpenAI API. This will provide an analysis for an evaluation datasets of the following metrics, with a score of 1-5 and an explanation of the score:

 * adequacy
 * fluency
 * terminology
 * hallucination
 * punctuation

See [pipeline/eval/eval-batch-instructions.md](https://github.com/mozilla/translations/blob/main)(pipeline/eval/eval-batch-instructions.md) for the full prompt for this analysis.

This evaluation can be viewed using the [LLM Evals dashboard](https://mozilla.github.io/translations/llm-evals) by providing the root URL to where the JSON files are located.
