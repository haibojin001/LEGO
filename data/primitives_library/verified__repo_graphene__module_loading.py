from functools import partial
from importlib import import_module


def import_string(dotted_path, dotted_attributes=None):
    module_path, separator, attribute_name = dotted_path.rpartition(".")
    if not separator:
        raise ImportError("%s doesn't look like a module path" % dotted_path)

    module = import_module(module_path)

    try:
        imported_object = getattr(module, attribute_name)
    except AttributeError:
        raise ImportError(
            'Module "%s" does not define a "%s" attribute/class'
            % (module_path, attribute_name)
        )

    if not dotted_attributes:
        return imported_object

    visited = []
    try:
        for name in dotted_attributes.split("."):
            visited.append(name)
            imported_object = getattr(imported_object, name)
    except AttributeError:
        raise ImportError(
            'Module "%s" does not define a "%s" attribute inside attribute/class "%s"'
            % (module_path, ".".join(visited), attribute_name)
        )

    return imported_object


def lazy_import(dotted_path, dotted_attributes=None):
    return partial(import_string, dotted_path, dotted_attributes)