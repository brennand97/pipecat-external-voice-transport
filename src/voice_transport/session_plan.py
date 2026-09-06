"""Immutable, provider-independent policy models for an External session."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


class SessionPlanError(ValueError):
    """A trusted profile or requested session policy is invalid."""


@dataclass(frozen=True, slots=True)
class ToolNamePattern:
    """An exact name or a single anchored terminal wildcard."""

    value: str

    @classmethod
    def parse(cls, value: object) -> ToolNamePattern:
        if not isinstance(value, str) or not value:
            raise SessionPlanError("tool pattern must be a non-empty string")
        if "*" not in value:
            return cls(value)
        if value.count("*") != 1 or not value.endswith("*") or len(value) == 1:
            raise SessionPlanError(
                "tool patterns must be exact names or a non-empty terminal "
                "prefix wildcard"
            )
        return cls(value)

    def matches(self, tool_name: str) -> bool:
        if self.value.endswith("*"):
            return tool_name.startswith(self.value[:-1])
        return tool_name == self.value


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Trusted attachment facts fixed before provider construction."""

    client_id: str
    client_kind: str
    conversation_id: str | None
    satellite_entity_id: str | None
    home_assistant_device_id: str | None
    input_modalities: frozenset[str]
    output_modalities: frozenset[str]

    @property
    def has_physical_satellite(self) -> bool:
        return self.satellite_entity_id is not None


@dataclass(frozen=True, slots=True)
class SessionProfileDefinition:
    """Trusted named profile loaded from deployment configuration."""

    name: str
    allowed_patterns: tuple[ToolNamePattern, ...]


@dataclass(frozen=True, slots=True)
class CompiledToolPlan:
    """Concrete tool names visible to the provider for one session."""

    profile: str
    tool_names: tuple[str, ...]


def compile_tool_names(
    *,
    profile: SessionProfileDefinition,
    provider_patterns: Iterable[ToolNamePattern],
    discovered_names: Iterable[str],
    requested_names: tuple[str, ...] | None,
) -> CompiledToolPlan:
    """Intersect trusted provider/profile policy with an exact client subset."""
    provider = tuple(provider_patterns)
    discovered = tuple(discovered_names)
    if requested_names is not None and len(set(requested_names)) != len(
        requested_names
    ):
        raise SessionPlanError("requested tool names must be unique")
    selected = tuple(
        name
        for name in discovered
        if any(pattern.matches(name) for pattern in provider)
        and any(pattern.matches(name) for pattern in profile.allowed_patterns)
        and (requested_names is None or name in requested_names)
    )
    if not selected:
        raise SessionPlanError("tool policy selected no discovered tools")
    return CompiledToolPlan(profile.name, selected)
