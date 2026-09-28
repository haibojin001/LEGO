import inspect
import logging
from collections import defaultdict
from typing import Any

from aiodataloader import DataLoader
from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import RelationshipProperty, Session

logger = logging.getLogger(__name__)


class SQLAlchemyDataLoader(DataLoader):
    """DataLoader for SQLAlchemy relationship properties."""

    def __init__(
        self,
        session: Session | AsyncSession,
        relation_prop: RelationshipProperty,
        cache: bool = True,
    ):
        super().__init__(cache=cache)
        self.session = session
        self.relation_prop = relation_prop
        self.target_model = relation_prop.mapper.class_
        self.is_list = relation_prop.uselist

        if relation_prop.secondary is not None:
            self.local_cols = [
                local.key
                for local, remote in relation_prop.synchronize_pairs
                if local.key is not None
            ]
            self.remote_cols = [
                remote.key
                for local, remote in relation_prop.synchronize_pairs
                if remote.key is not None
            ]
        else:
            pairs = sorted(
                (
                    (local.key, remote.key)
                    for local, remote in (relation_prop.local_remote_pairs or [])
                    if local.key is not None and remote.key is not None
                ),
                key=lambda pair: pair[0],
            )
            self.local_cols = [local for local, _ in pairs]
            self.remote_cols = [remote for _, remote in pairs]

        self.secondary = relation_prop.secondary

    def get_query(self, keys: list[Any]):
        """Build the select statement used for a batch of relationship keys."""
        statement = select(self.target_model)

        if self.secondary is not None:
            statement = statement.join(self.secondary)
            filter_columns = [
                self.secondary.c[column] for column in self.remote_cols
            ]
        else:
            filter_columns = [
                getattr(self.target_model, column) for column in self.remote_cols
            ]

        statement = statement.add_columns(*filter_columns)

        if len(filter_columns) > 1:
            statement = statement.where(tuple_(*filter_columns).in_(keys))
        else:
            normalized_keys = [
                key[0] if isinstance(key, (list, tuple)) else key for key in keys
            ]
            statement = statement.where(filter_columns[0].in_(normalized_keys))

        return statement

    async def batch_load_fn(self, keys: list[Any]) -> list[Any]:
        logger.debug(
            "SQLAlchemyRelationLoader: Fetching %s for %d parents",
            self.target_model.__name__,
            len(keys),
        )

        result = self.session.execute(self.get_query(keys))
        if inspect.isawaitable(result):
            result = await result

        rows = result.all()
        grouped = defaultdict(list)
        column_count = len(self.remote_cols)

        for row in rows:
            item = row[0]
            values = row[1 : 1 + column_count]
            group_key = tuple(values) if column_count > 1 else values[0]
            grouped[group_key].append(item)

        return [
            grouped[key] if self.is_list else (grouped[key][0] if grouped[key] else None)
            for key in keys
        ]


class LoaderRegistry:
    def __init__(self, session: Session | AsyncSession):
        self.session = session
        self._loaders: dict[
            tuple[RelationshipProperty, type[DataLoader]], DataLoader
        ] = {}

    def get_loader(
        self,
        relation_prop: RelationshipProperty,
        loader_class: type[SQLAlchemyDataLoader] = SQLAlchemyDataLoader,
    ) -> DataLoader:
        key = (relation_prop, loader_class)
        if key not in self._loaders:
            self._loaders[key] = loader_class(self.session, relation_prop)
        return self._loaders[key]