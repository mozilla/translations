# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# The transform sequences in this file are responsible for "fanning out" a job
# that operates on individual datasets into N jobs based on the parameters
# given. By default, it will fan out into one job for each dataset in the
# training config for from _all_ categories. This can be restricted by one or
# more of:
# - `category` to limit to the datasets in a particular category (eg: `train`)
# - `provider` to limit to datasets from particular provider (eg: `flores`)
# - `exclude-locales` to avoid generating jobs for given language pairs, eg:
#   {"src": "en", "trg": "ru"}. (This is primarily useful for tasks like
#   `corpus-clean-parallel-bicleaner-ai` which only work if a bicleaner pack
#   is available for a locale pair.
#
# These transform sequences will also perform string formatting in the given
# `substitution-fields`. (Normally this would be done with `task-context`, but
# this transform is much more aware of things like `provider` and `dataset`,
# so it's simply easier to do it here for fields that need these things.) Both
# transform sequences make the following variables available:
# - `provider` is the dataset provider. Eg: the `opus` part of `opus_Books/v1`.
# - `dataset` is the dataset name. Eg: the `Books/v1` part of `opus_Books/v1`.
# - `dataset_sanitized` is the dataset name with `/` and `.` characters replaced
#   with an `_` to make them more suitable in filenames and URLs.
#   Eg: `Books_v1` from `Books/V1`.
# - `src_locale` is the `src` from the training config.
# - `trg_locale` is the `trg` from the tarining config.
#
# Note that there are two available transform sequences here: `per_dataset`
# and `mono`. `mono` does everything that `per_dataset` does, but also:
# - Requires a `category` of either `mono-src` or `mono-trg`. (It doesn't make
#   sense to use this sequence without a category, or with other ones.)
# - Makes `locale` available as a substitution parameter - which will either
#   be set to the `src` or `trg` locale, depending on which category was used.

import copy

from taskgraph.transforms.base import TransformSequence
from taskgraph.util.schema import Schema
from voluptuous import ALLOW_EXTRA, Optional

from translations_taskgraph.util.substitution import substitute
from translations_taskgraph.util.dataset_helpers import sanitize_dataset_name

SCHEMA = Schema(
    {
        Optional("dataset-config"): {
            # Fields in each `job` that need to be substituted with data
            # provided by this transform.
            Optional("substitution-fields"): [str],
            Optional("category"): str,
            Optional("provider"): str,
            Optional("exclude-locales"): [
                {
                    "src": str,
                    "trg": str,
                },
            ],
        },
    },
    extra=ALLOW_EXTRA,
)

per_dataset = TransformSequence()
per_dataset.add_validate(SCHEMA)

mono = TransformSequence()
mono.add_validate(SCHEMA)


@per_dataset.add
def jobs_from_datasets(config, jobs):
    for job in jobs:
        dataset_config = job.pop("dataset-config", {})
        category = dataset_config.get("category", "")
        provider = dataset_config.get("provider", "")
        substitution_fields = dataset_config.get("substitution-fields", [])
        exclude_locales = dataset_config.get("exclude-locales", [])
        datasets = config.params["training_config"]["datasets"]
        src = config.params["training_config"]["experiment"]["src"]
        trg = config.params["training_config"]["experiment"]["trg"]

        included_datasets = set()
        if category:
            included_datasets.update(datasets.get(category, []))
        else:
            for sets in datasets.values():
                included_datasets.update(sets)

        if {"src": src, "trg": trg} in exclude_locales:
            continue

        for full_dataset in included_datasets:
            dataset_provider, dataset = full_dataset.split("_", 1)
            if provider and provider != dataset_provider:
                continue

            subjob = copy.deepcopy(job)

            subs = {
                "provider": dataset_provider,
                "dataset": full_dataset,
                "dataset_sanitized": sanitize_dataset_name(dataset),
                "src_locale": src,
                "trg_locale": trg,
            }
            for field in substitution_fields:
                container, subfield = subjob, field
                while "." in subfield:
                    f, subfield = subfield.split(".", 1)
                    container = container[f]

                container[subfield] = substitute(container[subfield], **subs)

            subjob.setdefault("attributes", {})
            subjob["attributes"]["provider"] = dataset_provider
            subjob["attributes"]["dataset"] = dataset
            subjob["attributes"]["src_locale"] = src
            subjob["attributes"]["trg_locale"] = trg

            yield subjob


@mono.add
def jobs_for_mono_datasets(config, jobs):
    for job in jobs:
        dataset_config = job.pop("dataset-config", {})
        category = dataset_config.get("category")
        provider = dataset_config.get("provider", "")
        substitution_fields = dataset_config.get("substitution-fields", [])
        exclude_locales = dataset_config.get("exclude-locales", [])
        datasets = config.params["training_config"]["datasets"]
        src = config.params["training_config"]["experiment"]["src"]
        trg = config.params["training_config"]["experiment"]["trg"]

        if {"src": src, "trg": trg} in exclude_locales:
            continue

        if category not in ("mono-src", "mono-trg"):
            raise Exception(
                "from_datasets:mono can only be used with mono-src and mono-trg categories"
            )

        included_datasets = set()
        if category:
            included_datasets.update(datasets.get(category, []))
        else:
            for sets in datasets.values():
                included_datasets.update(sets)

        for full_dataset in included_datasets:
            dataset_provider, dataset = full_dataset.split("_", 1)
            if provider and provider != dataset_provider:
                continue

            subjob = copy.deepcopy(job)

            if category == "mono-src":
                locale = src
            elif category == "mono-trg":
                locale = trg
            else:
                raise Exception(
                    "from_datasets:mono can only be used with mono-src and mono-trg categories"
                )

            subs = {
                "provider": dataset_provider,
                "dataset": full_dataset,
                "dataset_sanitized": sanitize_dataset_name(dataset),
                "locale": locale,
                "src_locale": src,
                "trg_locale": trg,
            }
            for field in substitution_fields:
                container, subfield = subjob, field
                while "." in subfield:
                    f, subfield = subfield.split(".", 1)
                    container = container[f]

                container[subfield] = substitute(container[subfield], **subs)

            subjob.setdefault("attributes", {})
            subjob["attributes"]["provider"] = dataset_provider
            subjob["attributes"]["dataset"] = dataset
            subjob["attributes"]["locale"] = locale
            subjob["attributes"]["src_locale"] = src
            subjob["attributes"]["trg_locale"] = trg

            yield subjob
