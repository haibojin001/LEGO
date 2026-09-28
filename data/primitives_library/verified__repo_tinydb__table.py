from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from typing import NoReturn, Optional, Union, cast, overload

from .queries import QueryLike
from .storages import Storage
from .utils import LRUCache

__all__ = ('Document', 'Table')


class Document(dict):
    def __init__(self, value: Mapping, doc_id: int):
        super().__init__(value)
        self.doc_id = doc_id


class Table:
    document_class = Document
    document_id_class = int
    query_cache_class = LRUCache
    default_query_cache_capacity = 10

    def __init__(
        self,
        storage: Storage,
        name: str,
        cache_size: int = default_query_cache_capacity,
        persist_empty: bool = False
    ):
        self._storage = storage
        self._name = name
        self._query_cache = self.query_cache_class(capacity=cache_size)
        self._next_id = None

        if persist_empty:
            self._update_table(lambda table: table.clear())

    def __repr__(self):
        values = [
            'name={!r}'.format(self.name),
            'total={}'.format(len(self)),
            'storage={}'.format(self._storage),
        ]
        return '<{} {}>'.format(type(self).__name__, ', '.join(values))

    @property
    def name(self) -> str:
        return self._name

    @property
    def storage(self) -> Storage:
        return self._storage

    def insert(self, document: Mapping) -> int:
        if not isinstance(document, Mapping):
            raise ValueError('Document is not a Mapping')

        if isinstance(document, self.document_class):
            doc_id = document.doc_id
            self._next_id = None
        else:
            doc_id = self._get_next_id()

        def add_document(table: dict) -> None:
            if doc_id in table:
                raise ValueError(
                    'Document with ID {} already exists'.format(str(doc_id))
                )

            table[doc_id] = dict(document)

        self._update_table(add_document)
        return doc_id

    def insert_multiple(self, documents: Iterable[Mapping]) -> list[int]:
        inserted_ids = []

        def add_documents(table: dict) -> None:
            for document in documents:
                if not isinstance(document, Mapping):
                    raise ValueError('Document is not a Mapping')

                if isinstance(document, self.document_class):
                    doc_id = document.doc_id

                    if doc_id in table:
                        raise ValueError(
                            'Document with ID {} already exists'.format(
                                str(doc_id)
                            )
                        )

                    inserted_ids.append(doc_id)
                    table[doc_id] = dict(document)
                    continue

                doc_id = self._get_next_id()
                inserted_ids.append(doc_id)
                table[doc_id] = dict(document)

        self._update_table(add_documents)
        return inserted_ids

    def all(self) -> list[Document]:
        return list(self)

    def search(self, cond: QueryLike) -> list[Document]:
        cached = self._query_cache.get(cond)
        if cached is not None:
            return cached[:]

        documents = [
            self.document_class(document, self.document_id_class(doc_id))
            for doc_id, document in self._read_table().items()
            if cond(document)
        ]

        cacheable = getattr(cond, 'is_cacheable', lambda: True)
        if cacheable():
            self._query_cache[cond] = documents[:]

        return documents

    @overload
    def get(self, cond: QueryLike) -> Optional[Document]:
        ...

    @overload
    def get(self, *, doc_id: int) -> Optional[Document]:
        ...

    def get(
        self,
        cond: Optional[QueryLike] = None,
        doc_id: Optional[int] = None
    ) -> Optional[Document]:
        if doc_id is not None:
            document = self._read_table().get(doc_id)

            if document is None:
                return None

            return self.document_class(
                document,
                self.document_id_class(doc_id)
            )

        if cond is not None:
            for found_id, document in self._read_table().items():
                if cond(document):
                    return self.document_class(
                        document,
                        self.document_id_class(found_id)
                    )

            return None

        raise RuntimeError('You have to pass either cond or doc_id')

    @overload
    def contains(self, cond: QueryLike) -> bool:
        ...

    @overload
    def contains(self, *, doc_id: int) -> bool:
        ...

    def contains(
        self,
        cond: Optional[QueryLike] = None,
        doc_id: Optional[int] = None
    ) -> bool:
        if doc_id is not None:
            return doc_id in self._read_table()

        if cond is not None:
            return any(cond(document) for document in self._read_table().values())

        raise RuntimeError('You have to pass either cond or doc_id')

    def update(
        self,
        fields: Union[Mapping, Callable[[MutableMapping], None]],
        cond: Optional[QueryLike] = None,
        doc_ids: Optional[Iterable[int]] = None
    ) -> None:
        def perform_update(table: dict) -> None:
            if doc_ids is not None:
                documents = (
                    (doc_id, table[doc_id])
                    for doc_id in doc_ids
                )
            else:
                documents = (
                    (doc_id, document)
                    for doc_id, document in table.items()
                    if cond(document)
                )

            for _, document in documents:
                if callable(fields):
                    fields(document)
                else:
                    document.update(fields)

        self._update_table(perform_update)

    def update_multiple(
        self,
        updates: Iterable[
            tuple[
                Union[Mapping, Callable[[MutableMapping], None]],
                QueryLike
            ]
        ]
    ) -> None:
        def perform_updates(table: dict) -> None:
            for fields, cond in updates:
                for document in table.values():
                    if cond(document):
                        if callable(fields):
                            fields(document)
                        else:
                            document.update(fields)

        self._update_table(perform_updates)

    def upsert(
        self,
        document: Mapping,
        cond: Optional[QueryLike] = None
    ) -> list[int]:
        if not isinstance(document, Mapping):
            raise ValueError('Document is not a Mapping')

        if isinstance(document, self.document_class):
            if self.contains(doc_id=document.doc_id):
                self.update(document, doc_ids=[document.doc_id])
                return [document.doc_id]

        if cond is None:
            return [self.insert(document)]

        documents = self.search(cond)

        if documents:
            self.update(document, cond)
            return [document.doc_id for document in documents]

        return [self.insert(document)]

    def remove(
        self,
        cond: Optional[QueryLike] = None,
        doc_ids: Optional[Iterable[int]] = None
    ) -> None:
        def perform_remove(table: dict) -> None:
            if doc_ids is not None:
                for doc_id in doc_ids:
                    table.pop(doc_id, None)
                return

            to_remove = [
                doc_id
                for doc_id, document in table.items()
                if cond(document)
            ]

            for doc_id in to_remove:
                del table[doc_id]

        self._update_table(perform_remove)

    def remove_multiple(self, conditions: Iterable[QueryLike]) -> None:
        def perform_remove(table: dict) -> None:
            for cond in conditions:
                to_remove = [
                    doc_id
                    for doc_id, document in table.items()
                    if cond(document)
                ]

                for doc_id in to_remove:
                    del table[doc_id]

        self._update_table(perform_remove)

    def truncate(self) -> None:
        self._update_table(lambda table: table.clear())
        self._next_id = None

    def count(self, cond: QueryLike) -> int:
        return len(self.search(cond))

    def clear_cache(self) -> None:
        self._query_cache.clear()

    def __len__(self) -> int:
        return len(self._read_table())

    def __iter__(self) -> Iterator[Document]:
        for doc_id, document in self._read_table().items():
            yield self.document_class(
                document,
                self.document_id_class(doc_id)
            )

    def __contains__(self, item) -> bool:
        if isinstance(item, self.document_class):
            return self.contains(doc_id=item.doc_id)

        return self.contains(cond=item)

    def _get_next_id(self) -> int:
        if self._next_id is None:
            table = self._read_table()

            if table:
                self._next_id = max(table.keys()) + 1
            else:
                self._next_id = 1

        next_id = self._next_id
        self._next_id += 1

        return next_id

    def _read_table(self) -> dict:
        tables = self._storage.read()

        if tables is None:
            return {}

        return tables.get(self.name, {})

    def _update_table(self, updater: Callable[[dict], None]) -> None:
        tables = self._storage.read()

        if tables is None:
            tables = {}

        table = tables.setdefault(self.name, {})
        updater(table)
        tables[self.name] = table

        self._storage.write(tables)
        self.clear_cache()