"""Tests for the authorization project.

These challenge behaviour rather than restating the
implementation. The cases that matter are the ones where a
plausible implementation would be wrong: a false accusation, a
false alarm, a qualifier that changes the answer, and a failure
that must not be silently turned into a verdict.
"""

from __future__ import annotations

from authorization_checker.authorization import (
    Authority,
    Authorization,
    ProposedAction,
    evaluate_action,
)

def test_untrusted_content_cannot_authorise_a_send():
    decision = evaluate_action(
        ProposedAction(
            action_type="send_email",
            target="x@example.invalid",
            source=Authority.UNTRUSTED,
        ),
        Authorization(allowed_actions=["read"]),
    )
    assert decision.decision == "denied"
    assert decision.trace


def test_reading_untrusted_content_is_allowed():
    decision = evaluate_action(
        ProposedAction(action_type="read", target="https://example.invalid", source=Authority.UNTRUSTED),
        Authorization(allowed_actions=["read"]),
    )
    assert decision.decision == "allowed"


def test_user_authorised_action_within_scope_is_allowed():
    decision = evaluate_action(
        ProposedAction(
            action_type="send_email",
            target="team@example.invalid",
            data_shared=["summary"],
            source=Authority.USER,
        ),
        Authorization(
            allowed_actions=["send_email"],
            allowed_targets=["team@example.invalid"],
            allowed_data_classes=["summary"],
        ),
    )
    assert decision.decision == "allowed"


def test_action_outside_the_user_scope_needs_clarification():
    decision = evaluate_action(
        ProposedAction(
            action_type="send_email",
            target="other@example.invalid",
            source=Authority.USER,
        ),
        Authorization(
            allowed_actions=["send_email"],
            allowed_targets=["team@example.invalid"],
        ),
    )
    assert decision.decision == "needs_clarification"


def test_untrusted_write_is_denied_even_when_writing_is_allowed():
    decision = evaluate_action(
        ProposedAction(
            action_type="write",
            target="src/config.py",
            source=Authority.UNTRUSTED,
        ),
        Authorization(allowed_actions=["read", "write"]),
    )
    assert decision.decision == "denied"
