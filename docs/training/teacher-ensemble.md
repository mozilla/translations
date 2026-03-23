---
layout: default
title: Teacher Ensemble
parent: Pipeline steps
---

# Teacher Ensemble

Teacher models are larger and slower translation models that have higher BLEU scores. In the pipeline they are used to distill smaller and faster student models at the cost of a lower BLEU score.

In the config files, you can specify how many teachers to train via `experiment.teacher-ensemble` key. The teachers will be identical except they will be initialized with different random seeds. This has been shown to improve the performance during student distillation, as the translation probabilities will be combined from both models.

While our current implementation only changes seeds, it's also possible to have ensembles that use different configurations or are trained on different datasets.

Recommendations information from [Efficient machine translation](https://nbogoychev.com/efficient-machine-translation/#ensembling):

> One very easy way to improve translation quality of the teacher is to produce an ensemble of systems that produce translation together. This is done by training identical systems, initialising them with different random seed. The more systems, the better, although returns are diminishing.
>
> For example, if we want to have an ensemble of two systems, we need to separate configuration files for training, where the seed parameter is different. Configuration one would have seed: 1111, whereas configuration two would have seed: 2222.

We typically use two teacher models when training with an ensemble. 
However, the quality boost is usually marginal and the cost of training and distillation increases significantly, so we've been using mostly only one teacher lately.
