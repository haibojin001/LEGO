from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple

from graphql import GraphQLError
from graphql.language import FieldNode, FragmentSpreadNode, InlineFragmentNode
from sqlalchemy.orm import class_mapper, joinedload, load_only, selectinload

from .types import LoadStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Mapper, RelationshipProperty
    from sqlalchemy.orm.strategy_options import _AbstractLoad

    from .objects import SQLAlchemyObjectType


logger = logging.getLogger(__name__)


class DepthLimit(NamedTuple):
    max: int
    set_by: str


def _flatten_selections(
    selections: Sequence[Any],
    fragments: dict[str, Any] | None,
) -> list[FieldNode]:
    fields: list[FieldNode] = []

    for selection in selections:
        if isinstance(selection, FieldNode):
            fields.append(selection)
            continue

        if isinstance(selection, InlineFragmentNode):
            if selection.selection_set:
                fields.extend(
                    _flatten_selections(selection.selection_set.selections, fragments)
                )
            continue

        if isinstance(selection, FragmentSpreadNode) and fragments:
            fragment = fragments.get(selection.name.value)
            if fragment is not None and fragment.selection_set:
                fields.extend(
                    _flatten_selections(fragment.selection_set.selections, fragments)
                )

    return fields


def _resolve_load_option(
    mapper: Mapper[Any],
    db_attr: str,
    strategy: LoadStrategy | None,
    rel: RelationshipProperty[Any],
    load_path: _AbstractLoad | None,
) -> _AbstractLoad:
    relationship_attribute = getattr(mapper.class_, db_attr)

    if strategy is None:
        strategy = selectinload if rel.uselist else joinedload

    if load_path is None:
        return strategy(relationship_attribute)

    strategy_name = getattr(strategy, "__name__")
    return getattr(load_path, strategy_name)(relationship_attribute)


def _build_options(
    mapper: Mapper[Any],
    selections: Sequence[Any],
    strategies: dict[str, LoadStrategy],
    aliases: dict[str, str],
    depth_limit: DepthLimit,
    type_registry: dict[type[Any], SQLAlchemyObjectType],
    current_depth: int = 1,
    load_path: _AbstractLoad | None = None,
    fragments: dict[str, Any] | None = None,
) -> list[_AbstractLoad]:
    object_type = type_registry.get(mapper.class_)
    if object_type is not None and object_type.max_depth < depth_limit.max:
        depth_limit = DepthLimit(object_type.max_depth, mapper.class_.__name__)

    if current_depth > depth_limit.max:
        raise GraphQLError(
            f"Query depth {current_depth} exceeds max_depth={depth_limit.max}"
            f" (set by '{depth_limit.set_by}')."
        )

    options: list[_AbstractLoad] = []
    fields = _flatten_selections(selections, fragments)
    scalar_names: list[str] = []

    for field in fields:
        graphql_name = field.name.value
        attribute_name = aliases.get(graphql_name, graphql_name)

        if (
            attribute_name not in mapper.relationships
            and hasattr(mapper.class_, attribute_name)
            and not callable(getattr(mapper.class_, attribute_name))
        ):
            scalar_names.append(attribute_name)

    if scalar_names:
        for relationship in mapper.relationships.values():
            for column in relationship.local_columns:
                column_name = column.key
                if (
                    column_name is not None
                    and hasattr(mapper.class_, column_name)
                    and column_name not in scalar_names
                ):
                    scalar_names.append(column_name)

        attributes = [getattr(mapper.class_, name) for name in scalar_names]

        if load_path is None:
            options.append(load_only(*attributes))
        else:
            options.append(load_path.load_only(*attributes))

    for field in fields:
        graphql_name = field.name.value
        attribute_name = aliases.get(graphql_name, graphql_name)

        if attribute_name not in mapper.relationships:
            continue

        relationship = mapper.relationships[attribute_name]
        target_model = relationship.mapper.class_
        target_object_type = type_registry.get(target_model)

        child_strategies = (
            target_object_type.strategies if target_object_type is not None else {}
        )
        child_aliases = (
            target_object_type.aliases if target_object_type is not None else {}
        )

        option = _resolve_load_option(
            mapper,
            attribute_name,
            strategies.get(graphql_name),
            relationship,
            load_path,
        )
        options.append(option)

        if field.selection_set:
            options.extend(
                _build_options(
                    relationship.mapper,
                    field.selection_set.selections,
                    child_strategies,
                    child_aliases,
                    depth_limit,
                    type_registry,
                    current_depth=current_depth + 1,
                    load_path=option,
                    fragments=fragments,
                )
            )

    return options


def auto_eager_load(
    query: Any,
    info: Any,
    model: type[Any],
    strategies: dict[str, LoadStrategy] | None = None,
    aliases: dict[str, str] | None = None,
    max_depth: int = 3,
    type_registry: dict[type[Any], SQLAlchemyObjectType] | None = None,
) -> Any:
    resolved_strategies = strategies or {}
    resolved_aliases = aliases or {}
    resolved_registry = type_registry or {}
    mapper = class_mapper(model)

    selections: list[Any] = []
    for node in info.field_nodes:
        if node.selection_set:
            selections.extend(node.selection_set.selections)

    if not selections:
        return query

    options = _build_options(
        mapper,
        selections,
        resolved_strategies,
        resolved_aliases,
        DepthLimit(max_depth, model.__name__),
        resolved_registry,
        fragments=getattr(info, "fragments", None),
    )

    if options:
        query = query.options(*options)

    return query