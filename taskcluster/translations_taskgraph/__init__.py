from importlib import import_module

from mozilla_taskgraph import register as register_mozilla_taskgraph


def register(graph_config):
    register_mozilla_taskgraph(graph_config)
    _import_modules(
        [
            "actions.train",
            "actions.rebuild_docker_images_and_toolchains",
            "parameters",
            "target_tasks",
            "worker_types",
        ]
    )


def _import_modules(modules):
    for module in modules:
        import_module(".{}".format(module), package=__name__)
