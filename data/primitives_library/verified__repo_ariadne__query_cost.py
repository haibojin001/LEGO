from functools import reduce
from operator import add, mul
from typing import Any, cast

from graphql import (
    GraphQLError,
    GraphQLInterfaceType,
    GraphQLNamedType,
    GraphQLObjectType,
    GraphQLSchema,
    get_named_type,
)
from graphql.execution.values import get_argument_values
from graphql.language import (
    BooleanValueNode,
    FieldNode,
    FragmentDefinitionNode,
    FragmentSpreadNode,
    InlineFragmentNode,
    IntValueNode,
    ListValueNode,
    Node,
    OperationDefinitionNode,
    OperationType,
    StringValueNode,
)
from graphql.type import GraphQLFieldMap
from graphql.validation import ValidationContext
from graphql.validation.rules import ASTValidationRule, ValidationRule


cost_directive = """
directive @cost(complexity: Int, multipliers: [String!], 
useMultipliers: Boolean) on FIELD | FIELD_DEFINITION
"""


CostAwareNode = (
    FieldNode
    | FragmentDefinitionNode
    | FragmentSpreadNode
    | InlineFragmentNode
    | OperationDefinitionNode
)


def report_error(context: ValidationContext, error: Exception) -> None:
    if isinstance(error, GraphQLError):
        context.report_error(error)
    else:
        context.report_error(GraphQLError(str(error)))


def _directive_value(value: Node) -> Any:
    if isinstance(value, IntValueNode):
        return int(value.value)
    if isinstance(value, BooleanValueNode):
        return value.value
    if isinstance(value, StringValueNode):
        return value.value
    if isinstance(value, ListValueNode):
        return [_directive_value(item) for item in value.values]
    return None


class CostValidator(ValidationRule):
    context: ValidationContext
    maximum_cost: int
    default_cost: int = 0
    default_complexity: int = 1
    variables: dict | None = None
    cost_map: dict[str, dict[str, Any]] | None = None

    def __init__(
        self,
        context: ValidationContext,
        maximum_cost: int,
        *,
        default_cost: int = 0,
        default_complexity: int = 1,
        variables: dict | None = None,
        cost_map: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(context)
        self.maximum_cost = maximum_cost
        self.variables = variables
        self.cost_map = cost_map
        self.default_cost = default_cost
        self.default_complexity = default_complexity
        self.cost = 0
        self.operation_multipliers: list[Any] = []
        self._fragment_cost_coefficients: dict[
            tuple[str, str | None], tuple[int, int]
        ] = {}

    def compute_node_cost(
        self, node: CostAwareNode, type_def, parent_multipliers=None
    ):
        if parent_multipliers is None:
            parent_multipliers = []

        if isinstance(node, FragmentSpreadNode) or not node.selection_set:
            return 0

        fields: GraphQLFieldMap = {}
        if isinstance(type_def, GraphQLObjectType | GraphQLInterfaceType):
            fields = type_def.fields

        total = 0

        for child_node in node.selection_set.selections:
            self.operation_multipliers = parent_multipliers[:]
            node_cost = self.default_cost

            if isinstance(child_node, FieldNode):
                field = fields.get(child_node.name.value)
                if not field:
                    continue

                field_type = get_named_type(field.type)

                try:
                    field_args: dict[str, Any] = get_argument_values(
                        field, child_node, self.variables
                    )
                except Exception as error:
                    report_error(self.context, error)
                    field_args = {}

                if self.cost_map:
                    cost_map_args = (
                        self.get_args_from_cost_map(
                            child_node, type_def.name, field_args
                        )
                        if type_def and type_def.name
                        else None
                    )
                    if cost_map_args is not None:
                        try:
                            node_cost = self.compute_cost(**cost_map_args)
                        except (TypeError, ValueError) as error:
                            report_error(self.context, error)
                else:
                    cost_is_computed = False

                    if field.ast_node and field.ast_node.directives:
                        directive_args = self.get_args_from_directives(
                            field.ast_node.directives, field_args
                        )
                        if directive_args is not None:
                            try:
                                node_cost = self.compute_cost(**directive_args)
                            except (TypeError, ValueError) as error:
                                report_error(self.context, error)
                            cost_is_computed = True

                    if (
                        field_type
                        and field_type.ast_node
                        and field_type.ast_node.directives
                    ):
                        if not cost_is_computed and isinstance(
                            field_type, GraphQLObjectType
                        ):
                            directive_args = self.get_args_from_directives(
                                field_type.ast_node.directives, field_args
                            )
                            if directive_args is not None:
                                try:
                                    node_cost = self.compute_cost(**directive_args)
                                except (TypeError, ValueError) as error:
                                    report_error(self.context, error)

                node_cost += self.compute_node_cost(
                    child_node, field_type, self.operation_multipliers
                )

            if isinstance(child_node, FragmentSpreadNode):
                fragment = self.context.get_fragment(child_node.name.value)
                if fragment:
                    fragment_type = self.context.schema.get_type(
                        fragment.type_condition.name.value
                    )
                    multiplier_product = reduce(mul, self.operation_multipliers, 1)
                    coefficient, offset = self.get_fragment_cost_coefficients(
                        fragment, fragment_type
                    )
                    node_cost = coefficient * multiplier_product + offset

            if isinstance(child_node, InlineFragmentNode):
                inline_fragment_type = type_def
                if child_node.type_condition and child_node.type_condition.name:
                    inline_fragment_type = self.context.schema.get_type(
                        child_node.type_condition.name.value
                    )
                node_cost = self.compute_node_cost(
                    child_node, inline_fragment_type, self.operation_multipliers
                )

            total += node_cost

        return total

    def get_fragment_cost_coefficients(
        self,
        fragment: FragmentDefinitionNode,
        fragment_type: GraphQLNamedType | None,
    ) -> tuple[int, int]:
        cache_key = (fragment.name.value, getattr(fragment_type, "name", None))
        if cache_key in self._fragment_cost_coefficients:
            return self._fragment_cost_coefficients[cache_key]

        cost_at_one = self.compute_node_cost(fragment, fragment_type, [])
        cost_at_two = self.compute_node_cost(fragment, fragment_type, [2])
        coefficient = cost_at_two - cost_at_one
        offset = cost_at_one - coefficient

        coefficients = (coefficient, offset)
        self._fragment_cost_coefficients[cache_key] = coefficients
        return coefficients

    def enter_operation_definition(self, node, key, parent, path, ancestors):
        if self.cost_map:
            try:
                validate_cost_map(self.cost_map, self.context.schema)
            except GraphQLError as error:
                self.context.report_error(error)
                return

        if node.operation is OperationType.QUERY:
            self.cost += self.compute_node_cost(node, self.context.schema.query_type)
        if node.operation is OperationType.MUTATION:
            self.cost += self.compute_node_cost(
                node, self.context.schema.mutation_type
            )
        if node.operation is OperationType.SUBSCRIPTION:
            self.cost += self.compute_node_cost(
                node, self.context.schema.subscription_type
            )

    def leave_operation_definition(self, node, key, parent, path, ancestors):
        if self.cost > self.maximum_cost:
            self.context.report_error(
                GraphQLError(
                    "The query exceeds the maximum cost of "
                    f"{self.maximum_cost}. Actual cost is {self.cost}",
                    node,
                )
            )

    def get_args_from_directives(self, directives, field_args: dict[str, Any]):
        for directive in directives:
            if directive.name.value != "cost":
                continue

            cost_args: dict[str, Any] = {}

            for argument in directive.arguments or ():
                if argument.name.value == "complexity":
                    cost_args["complexity"] = _directive_value(argument.value)
                elif argument.name.value == "multipliers":
                    multiplier_names = _directive_value(argument.value) or []
                    cost_args["multipliers"] = [
                        field_args.get(multiplier, 1)
                        for multiplier in cast(list[str], multiplier_names)
                    ]
                elif argument.name.value == "useMultipliers":
                    cost_args["use_multipliers"] = _directive_value(argument.value)

            return cost_args

        return None

    def get_args_from_cost_map(
        self,
        node: FieldNode,
        type_name: str,
        field_args: dict[str, Any],
    ):
        if not self.cost_map:
            return None

        type_cost_map = self.cost_map.get(type_name)
        if not type_cost_map:
            return None

        field_cost_map = type_cost_map.get(node.name.value)
        if field_cost_map is None:
            return None

        cost_args: dict[str, Any] = {}

        if "complexity" in field_cost_map:
            cost_args["complexity"] = field_cost_map["complexity"]

        if "multipliers" in field_cost_map:
            multiplier_names = cast(list[str], field_cost_map["multipliers"] or [])
            cost_args["multipliers"] = [
                field_args.get(multiplier, 1) for multiplier in multiplier_names
            ]

        if "use_multipliers" in field_cost_map:
            cost_args["use_multipliers"] = field_cost_map["use_multipliers"]
        elif "useMultipliers" in field_cost_map:
            cost_args["use_multipliers"] = field_cost_map["useMultipliers"]

        return cost_args

    def compute_cost(
        self,
        complexity: int | None = None,
        multipliers: list[Any] | None = None,
        use_multipliers: bool = False,
    ):
        if complexity is None:
            complexity = self.default_complexity

        if multipliers:
            normalized_multipliers = [
                multiplier
                if isinstance(multiplier, list)
                else [multiplier]
                for multiplier in multipliers
            ]
            self.operation_multipliers = reduce(
                add, normalized_multipliers, self.operation_multipliers
            )

        if use_multipliers:
            return complexity * reduce(mul, self.operation_multipliers, 1)

        return complexity


def validate_cost_map(
    cost_map: dict[str, dict[str, Any]], schema: GraphQLSchema
) -> None:
    for type_name, type_cost_map in cost_map.items():
        type_def = schema.get_type(type_name)

        if type_def is None:
            raise GraphQLError(
                f'Cost map contains type "{type_name}" that is not defined in schema.'
            )

        if not isinstance(type_def, GraphQLObjectType | GraphQLInterfaceType):
            raise GraphQLError(
                f'Cost map type "{type_name}" must be an object or interface type.'
            )

        if not isinstance(type_cost_map, dict):
            raise GraphQLError(
                f'Cost map entry for type "{type_name}" must be a dictionary.'
            )

        for field_name, field_cost_map in type_cost_map.items():
            if field_name not in type_def.fields:
                raise GraphQLError(
                    f'Cost map contains field "{type_name}.{field_name}" '
                    "that is not defined in schema."
                )

            if not isinstance(field_cost_map, dict):
                raise GraphQLError(
                    f'Cost map entry for field "{type_name}.{field_name}" '
                    "must be a dictionary."
                )


def cost_validator(
    maximum_cost: int,
    *,
    default_cost: int = 0,
    default_complexity: int = 1,
    variables: dict | None = None,
    cost_map: dict[str, dict[str, Any]] | None = None,
):
    def validator(context: ValidationContext):
        return CostValidator(
            context,
            maximum_cost,
            default_cost=default_cost,
            default_complexity=default_complexity,
            variables=variables,
            cost_map=cost_map,
        )

    return validator