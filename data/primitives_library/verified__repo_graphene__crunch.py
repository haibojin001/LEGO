import json
from collections.abc import Mapping


def to_key(value):
    return json.dumps(value)


def insert(value, index, values):
    serialized = to_key(value)
    if serialized in index:
        return index.get(serialized)
    position = len(values)
    index[serialized] = position
    values.append(value)
    return position


def flatten(data, index, values):
    if isinstance(data, (list, tuple)):
        result = [flatten(item, index, values) for item in data]
    elif isinstance(data, Mapping):
        result = {name: flatten(item, index, values) for name, item in data.items()}
    else:
        result = data
    return insert(result, index, values)


def crunch(data):
    lookup = {}
    output = []
    flatten(data, lookup, output)
    return output