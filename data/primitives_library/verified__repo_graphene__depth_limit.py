try:
    from re import Pattern
except ImportError:
    from typing import Pattern

from typing import Callable, Dict, List, Optional, Tuple, Union

from graphql import GraphQLError
from graphql.language import (
    DefinitionNode,
    FieldNode,
    FragmentDefinitionNode,
    FragmentSpreadNode,
    InlineFragmentNode,
    Node,
    OperationDefinitionNode,
)
from graphql.validation import ValidationContext, ValidationRule

from ..utils.is_introspection_key import is_introspection_key


IgnoreType = Union[Callable[[str], bool], Pattern, str]


def depth_limit_validator(
    max_depth: int,
    ignore: Optional[List[IgnoreType]] = None,
    callback: Optional[Callable[[Dict[str, int]], None]] = None,
):
    class DepthLimitValidator(ValidationRule):
        def __init__(self, validation_context: ValidationContext):
            fragment_nodes = get_fragments(validation_context.document.definitions)
            operation_nodes = get_queries_and_mutations(
                validation_context.document.definitions
            )
            depths = {}

            for operation_name, operation in operation_nodes.items():
                depths[operation_name] = determine_depth(
                    operation,
                    fragment_nodes,
                    0,
                    max_depth,
                    validation_context,
                    operation_name,
                    ignore,
                )

            if callable(callback):
                callback(depths)

            super().__init__(validation_context)

    return DepthLimitValidator


def get_fragments(
    definitions: Tuple[DefinitionNode, ...],
) -> Dict[str, FragmentDefinitionNode]:
    result = {}

    for definition in definitions:
        if isinstance(definition, FragmentDefinitionNode):
            result[definition.name.value] = definition

    return result


def get_queries_and_mutations(
    definitions: Tuple[DefinitionNode, ...],
) -> Dict[str, OperationDefinitionNode]:
    result = {}

    for definition in definitions:
        if isinstance(definition, OperationDefinitionNode):
            name = definition.name.value if definition.name else "anonymous"
            result[name] = definition

    return result


def determine_depth(
    node: Node,
    fragments: Dict[str, FragmentDefinitionNode],
    depth_so_far: int,
    max_depth: int,
    context: ValidationContext,
    operation_name: str,
    ignore: Optional[List[IgnoreType]] = None,
) -> int:
    if depth_so_far > max_depth:
        context.report_error(
            GraphQLError(
                f"'{operation_name}' exceeds maximum operation depth of {max_depth}.",
                [node],
            )
        )
        return depth_so_far

    if isinstance(node, FieldNode):
        name = node.name.value
        excluded = is_introspection_key(name) or is_ignored(node, ignore)

        if excluded or not node.selection_set:
            return 0

        return 1 + max(
            determine_depth(
                selection,
                fragments,
                depth_so_far + 1,
                max_depth,
                context,
                operation_name,
                ignore,
            )
            for selection in node.selection_set.selections
        )

    if isinstance(node, FragmentSpreadNode):
        return determine_depth(
            fragments[node.name.value],
            fragments,
            depth_so_far,
            max_depth,
            context,
            operation_name,
            ignore,
        )

    if isinstance(
        node,
        (InlineFragmentNode, FragmentDefinitionNode, OperationDefinitionNode),
    ):
        return max(
            determine_depth(
                selection,
                fragments,
                depth_so_far,
                max_depth,
                context,
                operation_name,
                ignore,
            )
            for selection in node.selection_set.selections
        )

    raise Exception(f"Depth crawler cannot handle: {node.kind}.")


def is_ignored(node: FieldNode, ignore: Optional[List[IgnoreType]] = None) -> bool:
    if ignore is None:
        return False

    field_name = node.name.value

    for rule in ignore:
        if isinstance(rule, str):
            if rule == field_name:
                return True
        elif isinstance(rule, Pattern):
            if rule.match(field_name):
                return True
        elif callable(rule):
            if rule(field_name):
                return True
        else:
            raise ValueError(f"Invalid ignore option: {rule}.")

    return False