def get_beetmover_upstream_dependency(dependencies):
    keys = list(dependencies.keys())
    match len(dependencies):
        case 0:
            raise Exception("upstream dependency is required for beetmover tasks!")
        case 1:
            if keys[0] == "decision":
                raise Exception("beetmover tasks must have a non-decision task dependency!")
            return keys[0], dependencies[keys[0]]

        case 2:
            if "decision" not in keys:
                raise Exception("beetmover tasks must only have one non-decision task dependency!")

            keys.remove("decision")
            return keys[0], dependencies[keys[0]]

    raise Exception("couldn't find beetmover upstream kind!")
