# Training Pipeline - mozilla/translations

Training pipelines and the inference engine for Firefox Translations machine translation models.

The trained models are hosted in a public Google Cloud Storage bucket (see Model Registry [UI](https://mozilla.github.io/translations/model-registry/) and [JSON](https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/db/models.json)).
The models are compatible with [bergamot-translator](https://github.com/mozilla/bergamot-translator) and
power the Firefox web page translation starting with version 118.

The pipeline was originally developed as a part of [Bergamot](https://browser.mt/) project  that focuses on improving client-side machine translation in a web browser.

## Pipeline

The pipeline is capable of training a translation model for a language pair end to end.
Translation quality depends on the chosen datasets, data cleaning procedures and hyperparameters.
Some settings, especially low resource languages might require extra tuning.

We use fast translation engine [Marian](https://marian-nmt.github.io).

See [more details about the pipeline steps](training/pipeline-steps.md).

## Orchestrators

An orchestrator is responsible for workflow management and parallelization.

- [Taskcluster](https://taskcluster.net/) - Mozilla task execution framework. It is also used for Firefox CI.
  It provides access to the hybrid cloud workers (GCP + on-prem) with increased scalability and observability.
  [Usage instructions](infrastructure/task-cluster.md).
- [Snakemake](https://snakemake.github.io/) - a file based orchestrator that allows to run the pipeline locally or on a Slurm cluster.
  [Usage instructions](infrastructure/snakemake.md). (The integration is not maintained since Mozilla has switched to Taskcluster. Contributions are welcome.)

## Experiment tracking

[Public training dashboard in Weights & Biases](https://wandb.ai/moz-translations/projects)

Marian training metrics are parsed from logs and published using a custom module within the `tracking` directory.
More information is available [here](infrastructure/tracking.md).

## Contributing

Contributions are welcome! See the [documentation on Contributing](contributing/index.md) for more details.

Feel free to ask questions in our Matrix channel [#firefoxtranslations:mozilla.org](https://matrix.to/#/#firefoxtranslations:mozilla.org).

## Useful Links

- [Model training guide](training/README.md) - practical advice on how to use the pipeline
- [High level overview post on Mozilla Hacks](https://hacks.mozilla.org/2022/06/training-efficient-neural-network-models-for-firefox-translations/)
- [The Training Pipeline DAG](https://docs.google.com/presentation/d/1HkypImI_hbA3n1ljU57ZPAzW8PuQqdv2wrXqj688KtQ/edit?slide=id.g3421e8f521e_1_419#slide=id.g3421e8f521e_1_419)
- [Lightning Talk on the Training Pipeline Overview](https://www.youtube.com/watch?v=TfDEAYCeF6s)
- [Model registry that shows all trained models](https://mozilla.github.io/translations/model-registry/)
- [JSON with exported models](https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/db/models.json)
- [Models released to Firefox](https://mozilla.github.io/translations/firefox-models/)
- [Final evaluation results](https://mozilla.github.io/translations/final-evals/)
- [Running Experiments Dashboard](https://docs.google.com/spreadsheets/d/1Kiz9xUjo2jpeeVGtaL3jA_cLiCiiyz8GvIoQADMyYqo/edit?gid=0#gid=0)
- Production bucket with models, training corpus, configs etc.: [moz-fx-translations-data--303e-prod-translations-data](https://console.cloud.google.com/storage/browser/moz-fx-translations-data--303e-prod-translations-data) - Uploaded models
- [Documentation of the Firefox integration](https://firefox-source-docs.mozilla.org/toolkit/components/translations/index.html)

## Acknowledgements

This project uses materials developed by:

- Bergamot project ([github](https://github.com/browsermt), [website](https://browser.mt/)) that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825303
- HPLT project ([github](https://github.com/hplt-project), [website](https://hplt-project.org/)) that has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101070350 and from UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number 10052546]
- OPUS-MT project ([github](https://github.com/Helsinki-NLP/Opus-MT), [website](https://opus.nlpl.eu/))
- Many other open source projects and research papers (see [References](docs/README.md#references))

## References

Here is a list of selected publications on which the training pipeline is based. 
You can find more relevant publications on [Bergamot project web-site](https://browser.mt/publications).

1. V. M. Sánchez-Cartagena, M. Bañón, S. Ortiz-Rojas and G. Ramírez-Sánchez, 
"[Prompsit's submission to WMT 2018 Parallel Corpus Filtering shared task](http://www.statmt.org/wmt18/pdf/WMT116.pdf)",
in *Proceedings of the Third Conference on Machine Translation, Volume 2: Shared Task Papers*.
Brussels, Belgium: Association for Computational Linguistics, October 2018

2. Gema Ramírez-Sánchez, Jaume Zaragoza-Bernabeu, Marta Bañón and Sergio Ortiz Rojas 
"[Bifixer and Bicleaner: two open-source tools to clean your parallel data.](https://eamt2020.inesc-id.pt/proceedings-eamt2020.pdf#page=311)",
in *Proceedings of the 22nd Annual Conference of the European Association for Machine Translation*.
Lisboa, Portugal: European Association for Machine Translation, November 2020
   
3. Mölder F, Jablonski KP, Letcher B, et al. [Sustainable data analysis with Snakemake](https://pubmed.ncbi.nlm.nih.gov/34035898/). F1000Res. 2021;10:33. Published 2021 Jan 18. doi:10.12688/f1000research.29032.2

4. [Edinburgh’s Submissions to the 2020 Machine Translation Efficiency Task](https://aclanthology.org/2020.ngt-1.26) (Bogoychev et al., NGT 2020)

5. [From Research to Production and Back: Ludicrously Fast Neural Machine Translation](https://aclanthology.org/D19-5632) (Kim et al., EMNLP 2019)

6. [The University of Edinburgh’s Submissions to the WMT19 News Translation Task](https://aclanthology.org/W19-5304) (Bawden et al., 2019)

7. Jörg Tiedemann, 2012, [Parallel Data, Tools and Interfaces in OPUS](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf). In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC'2012)
8. [The University of Edinburgh’s Neural MT Systems for WMT17](https://arxiv.org/abs/1708.00726), Rico Sennrich, Alexandra Birch, Anna Currey, Ulrich Germann, Barry Haddow, Kenneth Heafield, Antonio Valerio Miceli Barone, and Philip Williams. In Proceedings of the EMNLP 2017 Second Conference on Machine Translation (WMT17), 2017.
9. [Marian: Fast Neural Machine Translation in C++](https://arxiv.org/abs/1804.00344), Marcin Junczys-Dowmunt, Roman Grundkiewicz, Tomasz Dwojak, Hieu Hoang, Kenneth Heafield, Tom Neckermann, Frank Seide, Ulrich Germann, Alham Fikri Aji, Nikolay Bogoychev, Andre ́ F. T. Martins, and Alexandra Birch.
10. [Improving Neural Machine Translation Models with Monolingual Data](https://arxiv.org/abs/1511.06709), Rico Sennrich,Barry Haddow,Alexandra Birch, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2016.
11. [A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319) (Post, 2018)
12. [The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation](https://ai.facebook.com/research/publications/the-flores-101-evaluation-benchmark-for-low-resource-and-multilingual-machine-translation), Facebook
13. [Many-to-English Machine Translation Tools, Data, and Pretrained Models](https://aclanthology.org/2021.acl-demo.37) (Gowda et al., ACL 2021)
14. Chris Dyer, Victor Chahuneau, and Noah A. Smith. (2013). [A Simple, Fast, and Effective Reparameterization of IBM Model 2](http://www.ark.cs.cmu.edu/cdyer/fast_valign.pdf). In Proc. of NAACL.
15. [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162) (Sennrich et al., ACL 2016)
16. [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959) (Taku Kudo, 2018)
17. [Bicleaner AI: Bicleaner Goes Neural](https://aclanthology.org/2022.lrec-1.87.pdf) (Zaragoza-Bernabeu et al., LREC 2022)
18. [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947) (Yoon Kim, Alexander M. Rush, EMNLP 2016)
