import os
import re
from typing import cast

from graphql import extend_schema, parse
from graphql.language.ast import ObjectTypeDefinitionNode
from graphql.type import GraphQLObjectType, GraphQLSchema, GraphQLUnionType

from ...executable_schema import (
    SchemaBindables,
    join_type_defs,
    make_executable_schema,
)
from ...load_schema import load_schema_from_path
from ...schema_names import SchemaNameConverter
from ...schema_visitor import SchemaDirectiveVisitor
from .utils import get_entity_types, purge_schema_directives, resolve_entities


base_federation_service_type_defs = """
    scalar _Any

    type _Service {{
        sdl: String
    }}

    {type_token} Query {{
        _service: _Service!
    }}
"""

federation_entity_type_defs = """
    union _Entity

    extend type Query {
        _entities(representations: [_Any!]!): [_Entity]!
    }
"""


def has_query_type(type_defs: str) -> bool:
    document = parse(type_defs)
    return any(
        isinstance(definition, ObjectTypeDefinitionNode)
        and definition.name.value == "Query"
        for definition in document.definitions
    )


def make_federated_schema(
    type_defs: str | list[str],
    *bindables: SchemaBindables,
    directives: dict[str, type[SchemaDirectiveVisitor]] | None = None,
    convert_names_case: bool | SchemaNameConverter = False,
) -> GraphQLSchema:
    if isinstance(type_defs, list):
        type_defs = join_type_defs(type_defs)

    service_sdl = purge_schema_directives(type_defs)
    query_keyword = "extend type" if has_query_type(service_sdl) else "type"
    service_type_defs = base_federation_service_type_defs.format(
        type_token=query_keyword
    )

    definitions = [type_defs, service_type_defs]

    link_match = re.search(
        r'(?<=@link).*?url:.*?"(.*?)".?[^)]+?',
        service_sdl,
        re.MULTILINE | re.DOTALL,
    )

    definitions_directory = os.path.join(os.path.dirname(__file__), "definitions")
    definition_names = os.listdir(definitions_directory)
    definition_names.sort()
    latest_definition = definition_names[-1]

    if (
        link_match is not None
        and link_match.group(1).startswith("https://specs.apollo.dev/federation/")
    ):
        federation_version = link_match.group(1).split("/")[-1]
        versioned_definition = os.path.join(
            definitions_directory,
            f"fed_{federation_version}.graphql",
        )
        if os.path.isfile(versioned_definition):
            federation_definitions = load_schema_from_path(versioned_definition)
        else:
            federation_definitions = load_schema_from_path(
                os.path.join(definitions_directory, latest_definition)
            )
    else:
        federation_definitions = load_schema_from_path(
            os.path.join(
                os.path.dirname(__file__),
                "./definitions/fed_v1.0.graphql",
            )
        )

    definitions.append(federation_definitions)

    schema = make_executable_schema(
        join_type_defs(definitions),
        *bindables,
        directives=directives,
        convert_names_case=convert_names_case,
    )

    entity_types = get_entity_types(schema)
    if entity_types:
        schema = extend_schema(schema, parse(federation_entity_type_defs))

        entity_union = schema.get_type("_Entity")
        if entity_union:
            entity_union = cast(GraphQLUnionType, entity_union)
            setattr(entity_union, "types", entity_types)

        query = schema.get_type("Query")
        if query:
            query = cast(GraphQLObjectType, query)
            query.fields["_entities"].resolve = resolve_entities

    query = schema.get_type("Query")
    if query:
        query = cast(GraphQLObjectType, query)
        query.fields["_service"].resolve = lambda _service, info: {"sdl": service_sdl}

    return schema