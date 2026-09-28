from collections.abc import Mapping


def deflate(node, index=None, path=None):
    if index is None:
        index = {}
    if path is None:
        path = []

    if node and "id" in node and "__typename" in node:
        location = ",".join(path)
        identifier = ":".join(
            (location, str(node["__typename"]), str(node["id"]))
        )
        if index.get(identifier) is True:
            return {
                "__typename": node["__typename"],
                "id": node["id"],
            }
        index[identifier] = True

    output = {}
    for name in node:
        item = node[name]
        child_path = path + [name]

        if isinstance(item, (list, tuple)):
            output[name] = [
                deflate(child, index, child_path)
                for child in item
            ]
        elif isinstance(item, Mapping):
            output[name] = deflate(item, index, child_path)
        else:
            output[name] = item

    return output