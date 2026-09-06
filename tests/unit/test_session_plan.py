import pytest

from voice_transport.session_plan import (
    SessionPlanError,
    SessionProfileDefinition,
    ToolNamePattern,
    compile_tool_names,
)


def test_terminal_wildcards_are_anchored_and_intersected() -> None:
    profile = SessionProfileDefinition(
        "home", (ToolNamePattern.parse("intent__Hass*"),)
    )
    compiled = compile_tool_names(
        profile=profile,
        provider_patterns=(ToolNamePattern.parse("intent__Hass*"),),
        discovered_names=("intent__HassTurnOn", "other__HassTurnOn"),
        requested_names=None,
    )

    assert compiled.tool_names == ("intent__HassTurnOn",)


@pytest.mark.parametrize("pattern", ["*", "a*b", "a**", ""])
def test_unsafe_tool_patterns_fail_closed(pattern: str) -> None:
    with pytest.raises(SessionPlanError):
        ToolNamePattern.parse(pattern)


def test_client_subset_is_exact_and_cannot_expand_a_profile() -> None:
    profile = SessionProfileDefinition(
        "home", (ToolNamePattern.parse("intent__Hass*"),)
    )
    compiled = compile_tool_names(
        profile=profile,
        provider_patterns=(ToolNamePattern.parse("intent__Hass*"),),
        discovered_names=("intent__HassTurnOn", "intent__HassTurnOff"),
        requested_names=("intent__HassTurnOn",),
    )

    assert compiled.tool_names == ("intent__HassTurnOn",)


def test_empty_intersection_fails_closed() -> None:
    profile = SessionProfileDefinition("home", (ToolNamePattern.parse("allowed"),))
    with pytest.raises(SessionPlanError, match="selected no"):
        compile_tool_names(
            profile=profile,
            provider_patterns=(ToolNamePattern.parse("allowed"),),
            discovered_names=("other",),
            requested_names=None,
        )
