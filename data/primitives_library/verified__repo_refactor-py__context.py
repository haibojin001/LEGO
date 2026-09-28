from __future__ import annotations

import ast
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, DefaultDict, Protocol, cast

import refactor.common as common
from refactor.ast import UNPARSER_BACKENDS, BaseUnparser


@dataclass
class Configuration:
    """Configuration settings for a refactoring session."""

    unparser: str | type[BaseUnparser] = "precise"
    debug_mode: bool = False


class _Dependable(Protocol):
    context_providers: ClassVar[tuple[type[Representative], ...]]

    def __init__(self, context: Context) -> None:
        ...


def _resolve_dependencies(
    dependables: Iterable[type[_Dependable]],
) -> set[type[Representative]]:
    dependencies: set[type[Representative]] = set()
    pool = deque(dependables)

    while pool:
        dependable = pool.pop()
        pool.extendleft(
            dependency
            for dependency in dependable.context_providers
            if dependency not in dependencies
        )

        if issubclass(dependable, Representative):
            dependencies.add(cast(type[Representative], dependable))

    return dependencies


@dataclass
class Context:
    """The knowledge base associated with a parsed module."""

    source: str
    tree: ast.AST
    file_info: common._FileInfo = field(default_factory=common._FileInfo)
    config: Configuration = field(default_factory=Configuration)
    metadata: dict[str, Representative] = field(default_factory=dict)

    ancestry: Ancestry = field(init=False)
    scope: Scope = field(init=False)

    def __post_init__(self) -> None:
        self.ancestry = Ancestry(self)
        self.scope = Scope(self)

    @classmethod
    def _from_dependencies(
        cls,
        dependencies: Iterable[type[Representative]],
        **kwargs: Any,
    ) -> Context:
        context = cls(**kwargs)
        context._import_dependencies(dependencies)
        return context

    def _import_dependencies(
        self,
        representatives: Iterable[type[Representative]],
    ) -> None:
        for raw_representative in representatives:
            representative = raw_representative(self)
            self.metadata[representative.name] = representative

    def _update_metadata(self) -> None:
        for key, representative in self.metadata.copy().items():
            self.metadata[key] = type(representative)(self)

    def unparse(self, node: ast.AST) -> str:
        """Re-synthesize source code for ``node``."""

        unparser_backend = self.config.unparser

        if isinstance(unparser_backend, str):
            if unparser_backend not in UNPARSER_BACKENDS:
                raise ValueError(
                    "'unparser_backend' must be one of these: "
                    f"{', '.join(UNPARSER_BACKENDS)}"
                )
            backend_cls = UNPARSER_BACKENDS[unparser_backend]
        elif isinstance(unparser_backend, type):
            if not issubclass(unparser_backend, BaseUnparser):
                raise ValueError(
                    "'unparser_backend' must inherit from 'BaseUnparser'"
                )
            backend_cls = unparser_backend
        else:
            raise ValueError(
                "'unparser_backend' must be either a string or a type"
            )

        return backend_cls(source=self.source).unparse(node)  # type: ignore[no-any-return]

    def __getitem__(self, key: str) -> Representative:
        if key in _BUILTIN_REPRESENTATIVES:
            self._import_dependencies(
                _resolve_dependencies((_BUILTIN_REPRESENTATIVES[key],))
            )

        try:
            return self.metadata[key]
        except KeyError:
            raise ValueError(
                f"{key!r} provider is not available on this context "
                "since none of the rules from this session specified it "
                "in it's 'context_providers' tuple."
            ) from None

    def __getattr__(self, attr: str) -> Representative:
        try:
            return self[attr]
        except ValueError:
            raise AttributeError(
                f"{self!r} has no attribute {attr!r}"
            ) from None

    def replace(self, *args: Any, **kwargs: Any) -> Context:
        """Return a copy of this context with selected values replaced."""

        context = replace(self, *args, **kwargs)
        context._update_metadata()
        return context

    @property
    def file(self) -> Path | None:
        return self.file_info.path


@dataclass
class Representative:
    """Base class for tree-scoped metadata collectors."""

    context_providers: ClassVar[tuple[type[Representative], ...]] = ()

    context: Context

    @cached_property
    def name(self) -> str:
        if type(self) is Representative:
            return "<base>"
        return common.pascal_to_snake(type(self).__name__)


class Ancestry(Representative):
    """Provides parent lookup for AST nodes."""

    def _marked(self, node: ast.AST) -> bool:
        return hasattr(node, "parent")

    def _mark(self, parent: ast.AST, field: str, value: Any) -> None:
        if isinstance(value, ast.AST):
            value.parent = parent
            value.parent_field = field

    def _annotate(self, node: ast.AST) -> None:
        if self._marked(node):
            return

        node.parent = None
        node.parent_field = None

        for parent in ast.walk(node):
            for field, value in ast.iter_fields(parent):
                if isinstance(value, list):
                    for item in value:
                        self._mark(parent, field, item)
                else:
                    self._mark(parent, field, value)

    def _ensure_annotated(self) -> None:
        self._annotate(self.context.tree)

    def infer(self, node: ast.AST) -> tuple[str | None, ast.AST | None]:
        """Return the field holding ``node`` and its immediate parent."""

        self._ensure_annotated()
        return node.parent_field, node.parent

    def traverse(self, node: ast.AST) -> Iterable[tuple[str, ast.AST]]:
        """Yield the field and parent for every ancestor of ``node``."""

        current = node
        while True:
            field, parent = self.infer(current)
            if parent is None:
                return
            yield cast(str, field), parent
            current = parent

    def get_parent(self, node: ast.AST) -> ast.AST | None:
        """Retrieve the direct parent of ``node``."""

        return self.infer(node)[1]

    def get_parents(self, node: ast.AST) -> Iterable[ast.AST]:
        """Yield parents of ``node``, beginning with its direct parent."""

        for _, parent in self.traverse(node):
            yield parent


class ScopeType(Enum):
    GLOBAL = auto()
    CLASS = auto()
    FUNCTION = auto()
    COMPREHENSION = auto()


@dataclass(unsafe_hash=True)
class ScopeInfo:
    """Information about a lexical scope."""

    node: ast.AST
    scope_type: ScopeType
    parent: ScopeInfo | None = field(default=None, repr=False)

    def _iter_reachable_scopes(self) -> Iterator[ScopeInfo]:
        yield self

        current: ScopeInfo = self
        while current := current.parent:
            if current.scope_type in (ScopeType.FUNCTION, ScopeType.GLOBAL):
                yield current

    def can_reach(self, other: ScopeInfo) -> bool:
        """Whether names from ``other`` can be read in this scope."""

        return any(scope is other for scope in self._iter_reachable_scopes())

    def get_definitions(self, name: str) -> list[ast.AST]:
        """Find definitions of ``name`` visible from this scope."""

        for scope in self._iter_reachable_scopes():
            if scope.defines(name):
                return scope.definitions[name]
        return []

    def defines(self, name: str) -> bool:
        """Whether this scope itself binds ``name``."""

        return name in self.definitions

    @cached_property
    def definitions(self) -> DefaultDict[str, list[ast.AST]]:
        """Bindings made directly by this scope."""

        return defaultdict(list)


class _ScopeVisitor(ast.NodeVisitor):
    def __init__(self, owner: Scope) -> None:
        self.owner = owner
        self.current = owner._global_scope
        self._globals: dict[ScopeInfo, set[str]] = defaultdict(set)
        self._nonlocals: dict[ScopeInfo, set[str]] = defaultdict(set)

    def visit(self, node: ast.AST | None) -> Any:
        if node is None:
            return None
        self.owner._node_scopes.setdefault(node, self.current)
        return super().visit(node)

    def _target_scope(self, name: str) -> ScopeInfo:
        if name in self._globals[self.current]:
            return self.owner._global_scope

        if name in self._nonlocals[self.current]:
            parent = self.current.parent
            while parent is not None:
                if parent.scope_type in (ScopeType.FUNCTION, ScopeType.GLOBAL):
                    return parent
                parent = parent.parent

        return self.current

    def _define(self, name: str, node: ast.AST) -> None:
        self._target_scope(name).definitions[name].append(node)

    def _visit_in_scope(
        self,
        node: ast.AST,
        scope_type: ScopeType,
        body: Iterable[ast.AST],
        arguments: ast.arguments | None = None,
    ) -> None:
        previous = self.current
        child = ScopeInfo(node, scope_type, previous)
        self.owner._scope_nodes[node] = child
        self.current = child

        if arguments is not None:
            for argument in (
                list(arguments.posonlyargs)
                + list(arguments.args)
                + list(arguments.kwonlyargs)
            ):
                self._define(argument.arg, argument)
            if arguments.vararg is not None:
                self._define(arguments.vararg.arg, arguments.vararg)
            if arguments.kwarg is not None:
                self._define(arguments.kwarg.arg, arguments.kwarg)

        for statement in body:
            self.visit(statement)

        self.current = previous

    def _visit_argument_expressions(self, arguments: ast.arguments) -> None:
        for default in list(arguments.defaults) + list(arguments.kw_defaults):
            self.visit(default)

        for argument in (
            list(arguments.posonlyargs)
            + list(arguments.args)
            + list(arguments.kwonlyargs)
            + [arguments.vararg, arguments.kwarg]
        ):
            if argument is not None:
                self.visit(argument.annotation)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._define(node.id, node)

    def visit_arg(self, node: ast.arg) -> None:
        self.visit(node.annotation)

    def visit_Global(self, node: ast.Global) -> None:
        self._globals[self.current].update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._nonlocals[self.current].update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._define(alias.asname or alias.name.split(".", 1)[0], node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self._define(alias.asname or alias.name, node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._define(node.name, node)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_argument_expressions(node.args)
        self.visit(node.returns)
        for type_param in getattr(node, "type_params", ()):
            self.visit(type_param)
        self._visit_in_scope(node, ScopeType.FUNCTION, node.body, node.args)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_argument_expressions(node.args)
        self._visit_in_scope(
            node,
            ScopeType.FUNCTION,
            (node.body,),
            node.args,
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._define(node.name, node)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for type_param in getattr(node, "type_params", ()):
            self.visit(type_param)
        self._visit_in_scope(node, ScopeType.CLASS, node.body)

    def _visit_comprehension(
        self,
        node: ast.AST,
        generators: list[ast.comprehension],
        values: Iterable[ast.AST],
    ) -> None:
        if not generators:
            for value in values:
                self.visit(value)
            return

        self.visit(generators[0].iter)

        previous = self.current
        child = ScopeInfo(node, ScopeType.COMPREHENSION, previous)
        self.owner._scope_nodes[node] = child
        self.current = child

        for index, generator in enumerate(generators):
            if index:
                self.visit(generator.iter)
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)

        for value in values:
            self.visit(value)

        self.current = previous

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, node.generators, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, node.generators, (node.elt,))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, node.generators, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, node.generators, (node.key, node.value))

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.visit(node.type)
        if node.name is not None:
            self._define(node.name, node)
        for statement in node.body:
            self.visit(statement)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        self.visit(node.pattern)
        if node.name is not None:
            self._define(node.name, node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self._define(node.name, node)


@dataclass
class Scope(Representative):
    """Provides lexical scope and name definition lookup."""

    context_providers: ClassVar[tuple[type[Representative], ...]] = (Ancestry,)

    _global_scope: ScopeInfo = field(init=False)
    _node_scopes: dict[ast.AST, ScopeInfo] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _scope_nodes: dict[ast.AST, ScopeInfo] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._global_scope = ScopeInfo(self.context.tree, ScopeType.GLOBAL)
        self._scope_nodes[self.context.tree] = self._global_scope
        _ScopeVisitor(self).visit(self.context.tree)

    def get_scope(self, node: ast.AST) -> ScopeInfo:
        """Return the lexical scope containing ``node``."""

        try:
            return self._scope_nodes.get(node, self._node_scopes[node])
        except KeyError:
            raise ValueError("The specified node does not belong to this context.") from None

    def resolve(self, node: ast.AST) -> ScopeInfo:
        """Alias for :meth:`get_scope`."""

        return self.get_scope(node)


_BUILTIN_REPRESENTATIVES: dict[str, type[Representative]] = {
    "ancestry": Ancestry,
    "scope": Scope,
}