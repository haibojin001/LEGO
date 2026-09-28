from __future__ import annotations

import ast
import tempfile
import tokenize
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, NoReturn

from refactor.actions import (
    Action,
    BaseAction,
    NewStatementAction,
    ReplacementAction,
    TargetedNewStatementAction,
)
from refactor.change import Change
from refactor.common import _FileInfo, has_positions
from refactor.context import (
    Configuration,
    Context,
    Representative,
    _resolve_dependencies,
)
from refactor.internal.action_optimizer import optimize


class MaybeOverlappingActions(Exception):
    pass


@dataclass
class Rule:
    context_providers: ClassVar[tuple[type[Representative], ...]] = ()

    context: Context

    def check_file(self, path: Path | None) -> bool:
        return True

    def match(self, node: ast.AST) -> BaseAction | None | Iterator[BaseAction]:
        raise NotImplementedError


@dataclass
class Session:
    rules: list[type[Rule]] = field(default_factory=list)
    config: Configuration = field(default_factory=Configuration)

    def _initialize_rules(
        self,
        tree: ast.Module,
        source: str,
        file_info: _FileInfo,
    ) -> list[Rule]:
        dependencies = _resolve_dependencies(self.rules)
        context = Context._from_dependencies(
            dependencies,
            tree=tree,
            source=source,
            file_info=file_info,
            config=self.config,
        )

        initialized: list[Rule] = []
        for rule_type in self.rules:
            rule = rule_type(context)
            if rule.check_file(file_info.path):
                initialized.append(rule)
        return initialized

    def _apply_single(
        self,
        context: Context,
        source_code: str,
        action: BaseAction,
        enable_optimizations: bool = True,
    ) -> str:
        selected_action = optimize(action, context) if enable_optimizations else action
        return selected_action.apply(context, source_code)

    def _apply_multiple(
        self,
        rule: Rule,
        source_code: str,
        actions: Iterator[BaseAction],
    ) -> str:
        from refactor.internal.graph_access import AccessFailure, GraphPath

        applied_shifts: list[tuple[GraphPath, int]] = []
        current_tree = rule.context.tree

        for action in actions:
            input_node, stack_effect = action._stack_effect()
            graph_path = GraphPath.backtrack_from(rule.context, input_node)
            graph_path = graph_path.shift(applied_shifts)

            try:
                current_input = graph_path.execute(current_tree)
            except AccessFailure:
                raise MaybeOverlappingActions(
                    "When using chained actions, individual actions should not"
                    " overlap with each other."
                ) from None

            applied_shifts.append((graph_path, stack_effect))
            current_action = action._replace_input(current_input)
            current_context = rule.context.replace(
                source=source_code,
                tree=current_tree,
            )

            source_code = self._apply_single(
                current_context,
                source_code,
                current_action,
                enable_optimizations=False,
            )

            try:
                current_tree = ast.parse(source_code)
            except SyntaxError as exc:
                return self._unparsable_source_code(source_code, exc)

        return source_code

    def _run(
        self,
        source: str,
        file_info: _FileInfo,
        *,
        _changed: bool = False,
        _known_sources: frozenset[str] = frozenset(),
    ) -> tuple[str, bool]:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            if not _changed:
                return source, _changed
            return self._unparsable_source_code(source, exc)

        known_sources = _known_sources | {source}
        rules = self._initialize_rules(tree, source, file_info)

        for node in ast.walk(tree):
            if not has_positions(type(node)):  # type: ignore
                continue

            for rule in rules:
                with suppress(AssertionError):
                    result = rule.match(node)

                    if result is None:
                        continue

                    if isinstance(result, BaseAction):
                        transformed = self._apply_single(rule.context, source, result)
                    elif isinstance(result, Iterator):
                        transformed = self._apply_multiple(rule, source, result)
                    else:
                        raise TypeError(
                            f"Unexpected action type: {type(result).__name__}"
                        )

                    if transformed not in known_sources:
                        return self._run(
                            transformed,
                            file_info,
                            _changed=True,
                            _known_sources=known_sources,
                        )

        return source, _changed

    def _unparsable_source_code(self, source: str, exc: SyntaxError) -> NoReturn:
        message = "Generated source is unparsable."

        if self.config.debug_mode:
            descriptor, filename = tempfile.mkstemp(prefix="refactor", text=True)
            with open(descriptor, "w") as stream:
                stream.write(source)
            message += f"\nSee {filename} for the generated source."

        raise ValueError(message) from exc

    def run(self, source: str) -> str:
        result, _ = self._run(source, file_info=_FileInfo())
        return result

    def run_file(self, file: Path) -> Change | None:
        try:
            with tokenize.open(file) as stream:
                source = stream.read()
                encoding = stream.encoding
        except (SyntaxError, UnicodeDecodeError):
            return None

        file_info = _FileInfo(file, encoding)
        transformed, changed = self._run(source, file_info)

        if changed:
            return Change(file_info, source, transformed)

        return None