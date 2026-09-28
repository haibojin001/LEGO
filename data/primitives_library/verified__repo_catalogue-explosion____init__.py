import os
import re
import abc
import csv
import sys
import zipp
import email
import pathlib
import operator
import functools
import itertools
import posixpath
import collections

from ._compat import NullFinder, PyPy_repr, install, Protocol

from configparser import ConfigParser
from contextlib import suppress
from importlib import import_module
from importlib.abc import MetaPathFinder
from itertools import starmap
from typing import Any, List, Mapping, TypeVar, Union


__all__ = [
    'Distribution',
    'DistributionFinder',
    'PackageNotFoundError',
    'distribution',
    'distributions',
    'entry_points',
    'files',
    'metadata',
    'requires',
    'version',
]


class PackageNotFoundError(ModuleNotFoundError):
    """Raised when distribution metadata cannot be located."""

    def __str__(self):
        return 'No package metadata was found for {}'.format(self.name)

    @property
    def name(self):
        return self.args[0]


class EntryPoint(
    PyPy_repr, collections.namedtuple('EntryPointBase', 'name value group')
):
    """A single entry-point declaration."""

    pattern = re.compile(
        r'(?P<module>[\w.]+)\s*'
        r'(:\s*(?P<attr>[\w.]+))?\s*'
        r'(?P<extras>\[.*\])?\s*$'
    )

    def load(self):
        match = self.pattern.match(self.value)
        module = import_module(match.group('module'))
        attributes = filter(None, (match.group('attr') or '').split('.'))
        return functools.reduce(getattr, attributes, module)

    @property
    def module(self):
        return self.pattern.match(self.value).group('module')

    @property
    def attr(self):
        return self.pattern.match(self.value).group('attr')

    @property
    def extras(self):
        extras = self.pattern.match(self.value).group('extras') or ''
        return list(re.finditer(r'\w+', extras))

    @classmethod
    def _from_config(cls, config):
        return [
            cls(name, value, group)
            for group in config.sections()
            for name, value in config.items(group)
        ]

    @classmethod
    def _from_text(cls, text):
        parser = ConfigParser(delimiters='=')
        parser.optionxform = str
        parser.read_string(text)
        return EntryPoint._from_config(parser)

    def __iter__(self):
        return iter((self.name, self))

    def __reduce__(self):
        return self.__class__, (self.name, self.value, self.group)


class PackagePath(pathlib.PurePosixPath):
    """A package-relative file reference."""

    def read_text(self, encoding='utf-8'):
        with self.locate().open(encoding=encoding) as stream:
            return stream.read()

    def read_binary(self):
        with self.locate().open('rb') as stream:
            return stream.read()

    def locate(self):
        return self.dist.locate_file(self)


class FileHash:
    def __init__(self, spec):
        self.mode, _, self.value = spec.partition('=')

    def __repr__(self):
        return '<FileHash mode: {} value: {}>'.format(self.mode, self.value)


_T = TypeVar('_T')


class PackageMetadata(Protocol):
    def __len__(self) -> int:
        ...

    def __contains__(self, item: str) -> bool:
        ...

    def __getitem__(self, key: str) -> str:
        ...

    def get_all(self, name: str, failobj: _T = ...) -> Union[List[Any], _T]:
        ...


class Distribution:
    """Base class representing installed distribution metadata."""

    @abc.abstractmethod
    def read_text(self, filename):
        """Read a metadata member, returning text or ``None``."""

    @abc.abstractmethod
    def locate_file(self, path):
        """Resolve a package-relative path to a concrete path."""

    @classmethod
    def from_name(cls, name):
        for resolver in cls._discover_resolvers():
            candidate = next(
                iter(resolver(DistributionFinder.Context(name=name))), None
            )
            if candidate is not None:
                return candidate
        raise PackageNotFoundError(name)

    @classmethod
    def discover(cls, **kwargs):
        supplied_context = kwargs.pop('context', None)
        if supplied_context and kwargs:
            raise ValueError('cannot accept context and kwargs')
        context = supplied_context or DistributionFinder.Context(**kwargs)
        return itertools.chain.from_iterable(
            resolver(context) for resolver in cls._discover_resolvers()
        )

    @staticmethod
    def at(path):
        return PathDistribution(pathlib.Path(path))

    @staticmethod
    def _discover_resolvers():
        resolvers = (
            getattr(finder, '_catalogue_find_distributions', None)
            for finder in sys.meta_path
        )
        return filter(None, resolvers)

    @classmethod
    def _local(cls, root='.'):
        from pep517 import build, meta

        system = build.compat_system(root)
        builder = functools.partial(meta.build, source_dir=root, system=system)
        return PathDistribution(zipp.Path(meta.build_as_zip(builder)))

    @property
    def metadata(self) -> PackageMetadata:
        source = (
            self.read_text('METADATA')
            or self.read_text('PKG-INFO')
            or self.read_text('')
        )
        return email.message_from_string(source)

    @property
    def version(self):
        return self.metadata['Version']

    @property
    def entry_points(self):
        return EntryPoint._from_text(self.read_text('entry_points.txt'))

    @property
    def files(self):
        records = self._read_files_distinfo() or self._read_files_egginfo()

        def make_file(name, hash=None, size_str=None):
            item = PackagePath(name)
            item.hash = FileHash(hash) if hash else None
            item.size = int(size_str) if size_str else None
            item.dist = self
            return item

        return records and list(starmap(make_file, records))

    def _read_files_distinfo(self):
        text = self.read_text('RECORD')
        return text and csv.reader(text.splitlines())

    def _read_files_egginfo(self):
        text = self.read_text('SOURCES.txt')
        return text and csv.reader(text.splitlines())

    @property
    def requires(self):
        requirements = (
            self._read_dist_info_reqs() or self._read_egg_info_reqs()
        )
        return requirements and list(requirements)

    def _read_dist_info_reqs(self):
        return self.metadata.get_all('Requires-Dist')

    def _read_egg_info_reqs(self):
        text = self.read_text('requires.txt')
        return text and self._deps_from_requires_text(text)

    @classmethod
    def _deps_from_requires_text(cls, source):
        return cls._convert_egg_info_reqs_to_simple_reqs(Sectioned.read(source))

    @staticmethod
    def _convert_egg_info_reqs_to_simple_reqs(sections):
        return map(
            operator.add,
            map(operator.attrgetter('value'), sections),
            map(Distribution._adorn_non_simple_reqs, sections),
        )

    @staticmethod
    def _adorn_non_simple_reqs(section):
        if not section.name:
            return ''

        extra, separator, marker = section.name.partition(':')
        if separator:
            if extra:
                return ' ; {} and extra == "{}"'.format(marker, extra)
            return ' ; {}'.format(marker)

        return ' ; extra == "{}"'.format(extra)


class DistributionFinder(MetaPathFinder):
    """An import-system finder capable of locating distributions."""

    class Context:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

        @property
        def name(self):
            return vars(self).get('name')

        @property
        def path(self):
            return vars(self).get('path', sys.path)

    @abc.abstractmethod
    def find_distributions(self, context=Context()):
        """Find distributions matching *context*."""


class FastPath(str):
    """A cached representation of a metadata search root."""

    @functools.lru_cache()
    def __new__(cls, root):
        return super().__new__(cls, root)

    @property
    def root(self):
        return os.path.abspath(self)

    def joinpath(self, child):
        return pathlib.Path(self.root, child)

    @functools.lru_cache()
    def children(self):
        with suppress(Exception):
            return os.listdir(self.root or '.')
        with suppress(Exception):
            return self.zip_children()
        return []

    def zip_children(self):
        archive = zipp.Path(self.root)
        entries = archive.root.namelist()
        self.joinpath = archive.joinpath
        return list(
            dict.fromkeys(
                name.split(posixpath.sep, 1)[0]
                for name in entries
                if name
            )
        )

    @functools.lru_cache()
    def lookup(self):
        return Lookup(self)

    def search(self, name):
        return self.lookup().search(name)


class Prepared:
    """Normalized forms of a distribution name."""

    def __init__(self, name):
        self.name = name

    @staticmethod
    def normalize(name):
        return re.sub(r'[-_.]+', '-', name).lower().replace('-', '_')

    @staticmethod
    def legacy_normalize(name):
        return re.sub(r'[-_.]+', '-', name).lower()

    @property
    def normalized(self):
        return self.normalize(self.name) if self.name else None

    @property
    def legacy_normalized(self):
        return self.legacy_normalize(self.name) if self.name else None


class Lookup:
    """Indexes metadata directories available below a search root."""

    def __init__(self, path):
        self.infos = collections.defaultdict(list)
        self.eggs = collections.defaultdict(list)

        base = os.path.basename(path.root).lower()
        base_is_egg = base.endswith('.egg')

        for child in path.children():
            lowered = child.lower()

            if lowered.endswith(('.dist-info', '.egg-info')):
                project = lowered.rpartition('.')[0].partition('-')[0]
                key = Prepared.normalize(project)
                self.infos[key].append(path.joinpath(child))
            elif base_is_egg and lowered == 'egg-info':
                project = base[:-4].partition('-')[0]
                key = Prepared.legacy_normalize(project)
                self.eggs[key].append(path.joinpath(child))

    def search(self, prepared):
        if prepared is None:
            return itertools.chain.from_iterable(
                itertools.chain(self.infos.values(), self.eggs.values())
            )
        return itertools.chain(
            self.infos[prepared.normalized],
            self.eggs[prepared.legacy_normalized],
        )


class MetadataPathFinder(NullFinder, DistributionFinder):
    """Distribution finder for metadata available on ``sys.path``."""

    @classmethod
    def find_distributions(cls, context=DistributionFinder.Context()):
        prepared = Prepared(context.name)
        paths = map(FastPath, context.path)
        found = itertools.chain.from_iterable(
            path.search(prepared) for path in paths
        )
        return map(PathDistribution, found)

    _catalogue_find_distributions = find_distributions


class PathDistribution(Distribution):
    """A distribution backed by a pathlib or zipp metadata path."""

    def __init__(self, path):
        self._path = path

    def read_text(self, filename):
        with suppress(FileNotFoundError, IsADirectoryError, KeyError):
            return self._path.joinpath(filename).read_text(encoding='utf-8')

    def locate_file(self, path):
        return self._path.parent / path


class Sectioned(collections.namedtuple('SectionedBase', 'name value')):
    """An item in a sectioned requirements file."""

    @classmethod
    def read(cls, text):
        section = None
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]
                continue
            yield cls(section, line)


install(MetadataPathFinder)


def distribution(distribution_name):
    return Distribution.from_name(distribution_name)


def distributions(**kwargs):
    return Distribution.discover(**kwargs)


def metadata(distribution_name):
    return Distribution.from_name(distribution_name).metadata


def version(distribution_name):
    return Distribution.from_name(distribution_name).version


def entry_points():
    return itertools.chain.from_iterable(
        dist.entry_points for dist in distributions()
    )


def files(distribution_name):
    return Distribution.from_name(distribution_name).files


def requires(distribution_name):
    return Distribution.from_name(distribution_name).requires