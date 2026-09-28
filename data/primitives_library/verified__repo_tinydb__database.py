"""
Database facade and table manager for TinyDB.
"""

from collections.abc import Iterator

from .storages import JSONStorage, Storage
from .table import Document, Table
from .utils import with_typehint


TableBase: type[Table] = with_typehint(Table)


class TinyDB(TableBase):
    table_class = Table
    default_table_name = '_default'
    default_storage_class = JSONStorage

    def __init__(self, *args, **kwargs) -> None:
        storage_class = kwargs.pop('storage', self.default_storage_class)
        self._storage: Storage = storage_class(*args, **kwargs)
        self._opened = True
        self._tables: dict[str, Table] = {}

    def __repr__(self):
        table_names = self.tables()
        table_counts = [
            f'{table_name}={len(self.table(table_name))}'
            for table_name in table_names
        ]
        details = [
            f'tables={list(table_names)}',
            f'tables_count={len(table_names)}',
            f'default_table_documents_count={len(self)}',
            f'all_tables_documents_count={table_counts}',
        ]
        return '<{} {}>'.format(type(self).__name__, ', '.join(details))

    def table(self, name: str, **kwargs) -> Table:
        if name not in self._tables:
            self._tables[name] = self.table_class(self.storage, name, **kwargs)
        return self._tables[name]

    def tables(self) -> set[str]:
        data = self.storage.read()
        if data is None:
            return set()
        return set(data)

    def drop_tables(self) -> None:
        self.storage.write({})
        self._tables.clear()

    def drop_table(self, name: str) -> None:
        self._tables.pop(name, None)

        data = self.storage.read()
        if data is None or name not in data:
            return

        del data[name]
        self.storage.write(data)

    @property
    def storage(self) -> Storage:
        return self._storage

    def close(self) -> None:
        self._opened = False
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._opened:
            self.close()

    def __getattr__(self, name):
        return getattr(self.table(self.default_table_name), name)

    def __len__(self):
        return len(self.table(self.default_table_name))

    def __iter__(self) -> Iterator[Document]:
        return iter(self.table(self.default_table_name))