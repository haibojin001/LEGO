from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .errors import _ErrorCollector
from .requirements import InvalidRequirement, Requirement

__all__ = [
    "CyclicDependencyGroup",
    "DependencyGroupInclude",
    "DependencyGroupResolver",
    "DuplicateGroupNames",
    "InvalidDependencyGroupObject",
    "resolve_dependency_groups",
]


def __dir__() -> list[str]:
    return __all__


class DuplicateGroupNames(ValueError):
    """
    The same dependency groups were defined twice, with different non-normalized names.

    .. versionadded:: 26.1
    """


class CyclicDependencyGroup(ValueError):
    """
    The dependency group includes form a cycle.

    .. versionadded:: 26.1
    """

    def __init__(self, requested_group: str, group: str, include_group: str) -> None:
        self.requested_group = requested_group
        self.group = group
        self.include_group = include_group

        if include_group == group:
            reason = f"{group} includes itself"
        else:
            reason = f"{include_group} -> {group}, {group} -> {include_group}"

        super().__init__(
            "Cyclic dependency group include while resolving "
            f"{requested_group}: {reason}"
        )

    def __reduce__(self) -> tuple[type[CyclicDependencyGroup], tuple[str, str, str]]:
        return (self.__class__, (self.requested_group, self.group, self.include_group))


class InvalidDependencyGroupObject(ValueError):
    """
    A member of a dependency group was identified as a dict, but was not in a valid
    format.

    .. versionadded:: 26.1
    """


class DependencyGroupInclude:
    """
    A reference to another dependency group by name.

    .. versionadded:: 26.1
    """

    __slots__ = ("include_group",)

    def __init__(self, include_group: str) -> None:
        self.include_group = include_group

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.include_group!r})"


class DependencyGroupResolver:
    """
    A resolver for Dependency Group data.

    This class handles caching, name normalization, cycle detection, and other
    parsing requirements. There are only two public methods for exploring the data:
    ``lookup()`` and ``resolve()``.

    :param dependency_groups: A mapping, as provided via pyproject
        ``[dependency-groups]``.

    .. versionadded:: 26.1
    """

    def __init__(
        self,
        dependency_groups: Mapping[str, Sequence[str | Mapping[str, str]]],
    ) -> None:
        errors = _ErrorCollector()
        self.dependency_groups = _normalize_group_names(dependency_groups, errors)
        self._parsed_groups: dict[
            str, tuple[Requirement | DependencyGroupInclude, ...]
        ] = {}
        self._include_graph_ancestors: dict[str, tuple[str, ...]] = {}
        self._resolve_cache: dict[str, tuple[Requirement, ...]] = {}
        errors.finalize("[dependency-groups] data was invalid")

    def lookup(self, group: str) -> tuple[Requirement | DependencyGroupInclude, ...]:
        """
        Lookup a group name, returning the parsed dependency data for that group.
        This will not resolve includes.

        :param group: the name of the group to lookup
        """
        group = _normalize_name(group)

        with _ErrorCollector().on_exit(
            f"[dependency-groups] data for {group!r} was malformed"
        ) as errors:
            return self._parse_group(group, errors)

    def resolve(self, group: str) -> tuple[Requirement, ...]:
        """
        Resolve a dependency group to a list of requirements.

        :param group: the name of the group to resolve
        """
        group = _normalize_name(group)

        with _ErrorCollector().on_exit(
            f"[dependency-groups] data for {group!r} was malformed"
        ) as errors:
            return self._resolve(group, group, errors)

    def _resolve(
        self, group: str, requested_group: str, errors: _ErrorCollector
    ) -> tuple[Requirement, ...]:
        if group in self._resolve_cache:
            return self._resolve_cache[group]

        parsed = self._parse_group(group, errors)
        resolved: list[Requirement] = []

        for item in parsed:
            if isinstance(item, Requirement):
                resolved.append(item)
                continue

            if isinstance(item, DependencyGroupInclude):
                included = _normalize_name(item.include_group)

                if included in self._include_graph_ancestors.get(group, ()):
                    errors.error(
                        CyclicDependencyGroup(requested_group, group, item.include_group)
                    )
                    continue

                self._include_graph_ancestors[included] = (
                    *self._include_graph_ancestors.get(group, ()),
                    group,
                )
                resolved.extend(self._resolve(included, requested_group, errors))
                continue

            raise NotImplementedError(
                f"Invalid dependency group item after parse: {item}"
            )

        if errors.errors:
            return ()

        result = tuple(resolved)
        self._resolve_cache[group] = result
        return result

    def _parse_group(
        self, group: str, errors: _ErrorCollector
    ) -> tuple[Requirement | DependencyGroupInclude, ...]:
        if group in self._parsed_groups:
            return self._parsed_groups[group]

        if group not in self.dependency_groups:
            errors.error(LookupError(f"Dependency group '{group}' not found"))
            return ()

        raw_group = self.dependency_groups[group]

        if isinstance(raw_group, str):
            errors.error(
                TypeError(
                    f"Dependency group {group!r} contained a string rather than a list."
                )
            )
            return ()

        if not isinstance(raw_group, Sequence):
            errors.error(
                TypeError(f"Dependency group {group!r} is not a sequence type.")
            )
            return ()

        parsed: list[Requirement | DependencyGroupInclude] = []

        for item in raw_group:
            if isinstance(item, str):
                with errors.collect(InvalidRequirement):
                    parsed.append(Requirement(item))
                continue

            if isinstance(item, Mapping):
                if tuple(item.keys()) != ("include-group",):
                    errors.error(
                        InvalidDependencyGroupObject(
                            f"Invalid dependency group item: {item!r}"
                        )
                    )
                    continue

                include_group = item["include-group"]
                if not isinstance(include_group, str):
                    errors.error(
                        InvalidDependencyGroupObject(
                            "Dependency group include-group value is not a string: "
                            f"{include_group!r}"
                        )
                    )
                    continue

                parsed.append(DependencyGroupInclude(include_group))
                continue

            errors.error(
                TypeError(
                    "Dependency group item is not a string or mapping: "
                    f"{item!r}"
                )
            )

        result = tuple(parsed)
        self._parsed_groups[group] = result
        return result


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _normalize_group_names(
    dependency_groups: Mapping[str, Sequence[str | Mapping[str, str]]],
    errors: _ErrorCollector,
) -> dict[str, Sequence[str | Mapping[str, str]]]:
    normalized: dict[str, tuple[str, Sequence[str | Mapping[str, str]]]] = {}

    for name, group in dependency_groups.items():
        normalized_name = _normalize_name(name)

        if normalized_name in normalized:
            previous_name = normalized[normalized_name][0]
            errors.error(
                DuplicateGroupNames(
                    f"Duplicate dependency group names: {name!r} and "
                    f"{previous_name!r}"
                )
            )

        normalized[normalized_name] = (name, group)

    return {
        normalized_name: group
        for normalized_name, (_, group) in normalized.items()
    }


def resolve_dependency_groups(
    dependency_groups: Mapping[str, Sequence[str | Mapping[str, str]]],
    groups: Sequence[str],
) -> tuple[Requirement, ...]:
    """
    Resolve the requested dependency groups.

    :param dependency_groups: Dependency group data from ``[dependency-groups]``.
    :param groups: The dependency group names to resolve.
    """
    resolver = DependencyGroupResolver(dependency_groups)
    return tuple(
        requirement
        for group in groups
        for requirement in resolver.resolve(group)
    )