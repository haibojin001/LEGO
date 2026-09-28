import re

from .error import *
from .nodes import *

__all__ = ["BaseResolver", "Resolver"]


class ResolverError(YAMLError):
    pass


class BaseResolver:
    DEFAULT_SCALAR_TAG = "tag:yaml.org,2002:str"
    DEFAULT_SEQUENCE_TAG = "tag:yaml.org,2002:seq"
    DEFAULT_MAPPING_TAG = "tag:yaml.org,2002:map"

    yaml_implicit_resolvers = {}
    yaml_path_resolvers = {}

    def __init__(self):
        self.resolver_exact_paths = []
        self.resolver_prefix_paths = []

    @classmethod
    def add_implicit_resolver(cls, tag, regexp, first):
        if "yaml_implicit_resolvers" not in cls.__dict__:
            copied = {}
            for character, resolvers in cls.yaml_implicit_resolvers.items():
                copied[character] = resolvers[:]
            cls.yaml_implicit_resolvers = copied

        if first is None:
            first = [None]

        for character in first:
            cls.yaml_implicit_resolvers.setdefault(character, []).append(
                (tag, regexp)
            )

    @classmethod
    def add_path_resolver(cls, tag, path, kind=None):
        if "yaml_path_resolvers" not in cls.__dict__:
            cls.yaml_path_resolvers = cls.yaml_path_resolvers.copy()

        normalized_path = []

        for item in path:
            if isinstance(item, (list, tuple)):
                if len(item) == 2:
                    node_check, index_check = item
                elif len(item) == 1:
                    node_check = item[0]
                    index_check = True
                else:
                    raise ResolverError("Invalid path element: %s" % item)
            else:
                node_check = None
                index_check = item

            if node_check is str:
                node_check = ScalarNode
            elif node_check is list:
                node_check = SequenceNode
            elif node_check is dict:
                node_check = MappingNode
            elif (
                node_check not in (ScalarNode, SequenceNode, MappingNode)
                and not isinstance(node_check, str)
                and node_check is not None
            ):
                raise ResolverError("Invalid node checker: %s" % node_check)

            if not isinstance(index_check, (str, int)) and index_check is not None:
                raise ResolverError("Invalid index checker: %s" % index_check)

            normalized_path.append((node_check, index_check))

        if kind is str:
            kind = ScalarNode
        elif kind is list:
            kind = SequenceNode
        elif kind is dict:
            kind = MappingNode
        elif (
            kind not in (ScalarNode, SequenceNode, MappingNode)
            and kind is not None
        ):
            raise ResolverError("Invalid node kind: %s" % kind)

        cls.yaml_path_resolvers[(tuple(normalized_path), kind)] = tag

    def descend_resolver(self, current_node, current_index):
        if not self.yaml_path_resolvers:
            return

        exact_paths = {}
        prefix_paths = []

        if current_node:
            depth = len(self.resolver_prefix_paths)

            for path, kind in self.resolver_prefix_paths[-1]:
                if self.check_resolver_prefix(
                    depth, path, kind, current_node, current_index
                ):
                    if len(path) > depth:
                        prefix_paths.append((path, kind))
                    else:
                        exact_paths[kind] = self.yaml_path_resolvers[(path, kind)]
        else:
            for path, kind in self.yaml_path_resolvers:
                if not path:
                    exact_paths[kind] = self.yaml_path_resolvers[(path, kind)]
                else:
                    prefix_paths.append((path, kind))

        self.resolver_exact_paths.append(exact_paths)
        self.resolver_prefix_paths.append(prefix_paths)

    def ascend_resolver(self):
        if not self.yaml_path_resolvers:
            return

        self.resolver_exact_paths.pop()
        self.resolver_prefix_paths.pop()

    def check_resolver_prefix(
        self, depth, path, kind, current_node, current_index
    ):
        node_check, index_check = path[depth - 1]

        if isinstance(node_check, str):
            if current_node.tag != node_check:
                return
        elif node_check is not None:
            if not isinstance(current_node, node_check):
                return

        if index_check is True and current_index is not None:
            return

        if (index_check is False or index_check is None) and current_index is None:
            return

        if isinstance(index_check, str):
            if not (
                isinstance(current_index, ScalarNode)
                and current_index.value == index_check
            ):
                return
        elif isinstance(index_check, int) and not isinstance(index_check, bool):
            if current_index != index_check:
                return

        return True

    def resolve(self, kind, value, implicit):
        if kind is ScalarNode and implicit[0]:
            if value == "":
                resolvers = self.yaml_implicit_resolvers.get("", [])
            else:
                resolvers = self.yaml_implicit_resolvers.get(value[0], [])

            wildcard_resolvers = self.yaml_implicit_resolvers.get(None, [])

            for tag, regexp in resolvers + wildcard_resolvers:
                if regexp.match(value):
                    return tag

            implicit = implicit[1]

        if self.yaml_path_resolvers:
            exact_paths = self.resolver_exact_paths[-1]

            if kind in exact_paths:
                return exact_paths[kind]

            if None in exact_paths:
                return exact_paths[None]

        if kind is ScalarNode:
            return self.DEFAULT_SCALAR_TAG

        if kind is SequenceNode:
            return self.DEFAULT_SEQUENCE_TAG

        if kind is MappingNode:
            return self.DEFAULT_MAPPING_TAG


class Resolver(BaseResolver):
    pass


Resolver.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(
        r"""^(?:yes|Yes|YES|no|No|NO
                |true|True|TRUE|false|False|FALSE
                |on|On|ON|off|Off|OFF)$""",
        re.X,
    ),
    list("yYnNtTfFoO"),
)

Resolver.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
                |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
                |[-+]?\.(?:inf|Inf|INF)
                |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)

Resolver.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(
        r"""^(?:[-+]?0b[0-1_]+
                |[-+]?0[0-7_]+
                |[-+]?(?:0|[1-9][0-9_]*)
                |[-+]?0x[0-9a-fA-F_]+
                |[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$""",
        re.X,
    ),
    list("-+0123456789"),
)

Resolver.add_implicit_resolver(
    "tag:yaml.org,2002:merge",
    re.compile(r"^(?:<<)$"),
    ["<"],
)

Resolver.add_implicit_resolver(
    "tag:yaml.org,2002:null",
    re.compile(
        r"""^(?: ~
                |null|Null|NULL
                | )$""",
        re.X,
    ),
    ["~", "n", "N", ""],
)

Resolver.add_implicit_resolver(
    "tag:yaml.org,2002:timestamp",
    re.compile(
        r"""^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]
                |[0-9][0-9][0-9][0-9] -[0-9][0-9]? -[0-9][0-9]?
                 (?:[Tt]|[ \t]+)[0-9][0-9]?
                 :[0-9][0-9] :[0-9][0-9] (?:\.[0-9]*)?
                 (?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$""",
        re.X,
    ),
    list("0123456789"),
)

Resolver.add_implicit_resolver(
    "tag:yaml.org,2002:value",
    re.compile(r"^(?:=)$"),
    ["="],
)

Resolver.add_implicit_resolver(
    "tag:yaml.org,2002:yaml",
    re.compile(r"^(?:!|&|\*)$"),
    list("!&*"),
)