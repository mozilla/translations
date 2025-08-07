from copy import deepcopy

from taskgraph.taskgraph import TaskGraph

from translations_taskgraph.parameters import get_ci_training_config

PARAMS = deepcopy(get_ci_training_config())


def test_monocleaner_params(full_task_graph: TaskGraph):
    tasks = {t.label: t for t in full_task_graph.tasks.values()}

    assert (
        float(
            tasks["corpus-clean-mono-news-crawl-ru-news_2008-mono-src"].task["payload"]["command"][
                -1
            ][-3:]
        )
        == PARAMS["training_config"]["experiment"]["monocleaner"]["mono-src"][
            "dataset-thresholds"
        ]["news-crawl_news_2008"]
    )
    assert (
        float(
            tasks["corpus-clean-mono-news-crawl-en-news_2007-mono-trg"].task["payload"]["command"][
                -1
            ][-3:]
        )
        == PARAMS["training_config"]["experiment"]["monocleaner"]["mono-trg"]["default-threshold"]
    )
    assert (
        float(
            tasks["corpus-clean-mono-opus-ru-tldr-pages_v2023-08-29-mono-src"].task["payload"][
                "command"
            ][-1][-3:]
        )
        == PARAMS["training_config"]["experiment"]["monocleaner"]["mono-src"][
            "dataset-thresholds"
        ]["opus_tldr-pages_v2023-08-29"]
    )
    assert (
        float(
            tasks["corpus-clean-mono-opus-en-tldr-pages_v2023-08-29-mono-trg"].task["payload"][
                "command"
            ][-1][-3:]
        )
        == PARAMS["training_config"]["experiment"]["monocleaner"]["mono-trg"][
            "dataset-thresholds"
        ]["opus_tldr-pages_v2023-08-29"]
    )


def test_bicleaner_params(full_task_graph: TaskGraph):
    tasks = {t.label: t for t in full_task_graph.tasks.values()}

    assert (
        str(PARAMS["training_config"]["experiment"]["bicleaner"]["default-threshold"])
        in tasks["corpus-clean-parallel-bicleaner-ai-mtdata-Tilde-airbaltic-1-eng-rus-ru-en"].task[
            "payload"
        ]["command"][-1][-50:]
    )
    assert (
        str(
            PARAMS["training_config"]["experiment"]["bicleaner"]["dataset-thresholds"][
                "opus_ada83_v1"
            ]
        )
        in tasks["corpus-clean-parallel-bicleaner-ai-opus-ada83_v1-ru-en"].task["payload"][
            "command"
        ][-1][-50:]
    )
