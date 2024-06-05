from importlib import import_module


def register(graph_config):
    _import_modules(
        [
            "actions.train",
            "actions.rebuild_docker_images_and_toolchains",
            "parameters",
            "target_tasks",
            "workertypes",
        ]
    )


def _import_modules(modules):
    for module in modules:
        import_module(".{}".format(module), package=__name__)
