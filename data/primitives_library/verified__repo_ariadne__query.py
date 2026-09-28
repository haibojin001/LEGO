import inspect
from collections.abc import Sequence
from typing import Any

from graphql import GraphQLList, GraphQLNonNull, GraphQLObjectType, GraphQLSchema
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from ...objects import QueryType
from .objects import SQLAlchemyObjectType
from .utils import auto_eager_load


class SQLAlchemyQueryType(QueryType):
    """
    A custom Query type that automatically binds SQLAlchemy resolvers
    by inspecting the GraphQLSchema during the make_executable_schema build phase.
    """

    def __init__(
        self,
        object_types: Sequence[SQLAlchemyObjectType],
    ):
        super().__init__()
        self.object_types = {object_type.name: object_type for object_type in object_types}
        self._object_types_by_model = {
            object_type.model: object_type for object_type in object_types
        }

    @staticmethod
    def get_session_from_context(context: Any) -> Session | AsyncSession:
        try:
            return context["session"]
        except KeyError:
            raise RuntimeError("Session not found in context under key 'session'")

    def bind_to_schema(self, schema: GraphQLSchema) -> None:
        query_type = schema.type_map.get(self.name)

        if not isinstance(query_type, GraphQLObjectType):
            super().bind_to_schema(schema)
            return

        for field_name, field in query_type.fields.items():
            field_type = field.type
            returns_list = False

            while isinstance(field_type, (GraphQLList, GraphQLNonNull)):
                if isinstance(field_type, GraphQLList):
                    returns_list = True
                field_type = field_type.of_type

            object_type_name = getattr(field_type, "name", None)

            if (
                object_type_name is not None
                and object_type_name in self.object_types
                and field_name not in self._resolvers
            ):
                self.set_field(
                    field_name,
                    self._create_auto_resolver(
                        self.object_types[object_type_name],
                        returns_list,
                    ),
                )

        super().bind_to_schema(schema)

    def _create_auto_resolver(
        self,
        obj_type: SQLAlchemyObjectType,
        return_list: bool,
    ):
        def auto_resolve(obj: Any, info: Any, **kwargs: Any):
            session = self.get_session_from_context(info.context)
            model = obj_type.model
            statement = obj_type.get_base_query(info, **kwargs)

            statement = auto_eager_load(
                statement,
                info,
                model,
                strategies=obj_type.strategies,
                aliases=obj_type.aliases,
                max_depth=obj_type.max_depth,
                type_registry=self._object_types_by_model,
            )

            mapper = sa_inspect(model)
            for key, value in kwargs.items():
                column_name = obj_type.aliases.get(key, key)
                if column_name in mapper.columns:
                    statement = statement.where(
                        getattr(model, column_name) == value
                    )

            result = session.execute(statement)

            if inspect.isawaitable(result):

                async def await_result(awaitable_result: Any) -> Any:
                    execution_result = await awaitable_result
                    if return_list:
                        return execution_result.scalars().unique().all()
                    return execution_result.scalars().first()

                return await_result(result)

            if return_list:
                return result.scalars().unique().all()
            return result.scalars().first()

        return auto_resolve