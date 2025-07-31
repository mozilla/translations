from taskgraph.target_tasks import register_target_task


@register_target_task("train-target-tasks")
def train_target_tasks(full_task_graph, parameters, graph_config):
    training_config = parameters["training_config"]
    stage = training_config["target-stage"]
    src = training_config["experiment"]["src"]
    trg = training_config["experiment"]["trg"]
    datasets = parameters["training_config"]["datasets"]

    def filter(task):
        # These attributes will be present on tasks from all stages
        if task.attributes.get("stage") != stage:
            return False

        if task.attributes.get("src_locale") != src:
            return False

        if task.attributes.get("trg_locale") != trg:
            return False

        # Datasets are only applicable to dataset-specific tasks. If these
        # attribute isn't present on the task it can be assumed to be included
        # if the above attributes matched, as it will be a task that is either
        # agnostic of datasets, or folds in datasets from earlier tasks.
        # (Pulling in the appropriate datasets for these task must be handled at
        # the task generation level, usually by the `find_upstreams` transform.)
        if "dataset" in task.attributes:
            dataset_category = task.attributes["dataset-category"]
            for ds in datasets.get(dataset_category, []):
                provider, dataset = ds.split("_", 1)
                # If the task is for any of the datasets in the specified category,
                # it's a match, and should be included in the target tasks.
                if (
                    task.attributes["provider"] == provider
                    and task.attributes["dataset"] == dataset
                ):
                    break

        return True

    return [label for label, task in full_task_graph.tasks.items() if filter(task)]


@register_target_task("final-eval-target-tasks")
def final_eval_target_tasks(full_task_graph, parameters, _):
    stage = parameters["target-stage"]
    def filter(task):
        # These attributes will be present on tasks from all stages
        if task.attributes.get("stage") != stage:
            return False

        return True

    return [label for label, task in full_task_graph.tasks.items() if filter(task)]
