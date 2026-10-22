"""Project 10: agent action authorization checker.

Problem
-------
An AI agent may treat text from a webpage, a document, a log, or a tool
result as permission to act. The instruction inside the content looks
exactly like the instruction the user gave, and the agent follows it.

Approach
--------
This is a **dry-run policy evaluator**. It never performs the proposed
action. It answers one question: was the agent authorised to take this
action, given who the authority came from?

Three things are checked:

* the **action type** (read, write, send, delete, accept terms, ...),
* the **target** (which file, which URL, which recipient),
* the **source of authority** (user, trusted application policy, or
  untrusted external content).

The source of authority is the decisive one. A request that arrives in
untrusted content carries no authority at all, whatever it says. A
request that arrives from the user is checked against the scope of what
the user actually authorised.

Deliberate limits
-----------------
* This is a constrained policy decision, not a claim that prompt
  injection is solved. It assumes the provenance tag is correct; it
  cannot detect a page that has been mislabelled as trusted.
* Benign quoted instructions are included on purpose. A checker that
  blocks every page containing an imperative would be useless.
* No action is ever taken. The output is a decision and a trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core.types import CaseResult, CheckStatus, EvalCase, Method


class Authority(str):
    """Where an instruction came from."""

    USER = "user"
    TRUSTED_POLICY = "trusted_application_policy"
    UNTRUSTED = "untrusted_external_content"


@dataclass
class Authorization:
    """What the user actually authorised."""

    allowed_actions: list[str] = field(default_factory=list)
    allowed_targets: list[str] = field(default_factory=list)
    allowed_data_classes: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_actions": list(self.allowed_actions),
            "allowed_targets": list(self.allowed_targets),
            "allowed_data_classes": list(self.allowed_data_classes),
            "notes": self.notes,
        }


@dataclass
class ProposedAction:
    """An action the agent proposes to take."""

    action_type: str
    target: str = ""
    data_shared: list[str] = field(default_factory=list)
    source: str = Authority.UNTRUSTED
    source_detail: str = ""
    instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "data_shared": list(self.data_shared),
            "source": self.source,
            "source_detail": self.source_detail,
            "instruction": self.instruction,
        }


@dataclass
class PolicyDecision:
    """The outcome of evaluating one proposed action."""

    action: ProposedAction
    decision: str = "allowed"  # "allowed" | "denied" | "needs_clarification"
    rule: str = ""
    reason: str = ""
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "decision": self.decision,
            "rule": self.rule,
            "reason": self.reason,
            "trace": list(self.trace),
        }

    def render(self) -> str:
        lines = [
            f"Decision: {self.decision.upper()}",
            f"  action: {self.action.action_type} on {self.action.target or "(no target)"}",
            f"  source: {self.action.source}",
            f"  rule: {self.rule}",
            f"  reason: {self.reason}",
        ]
        for step in self.trace:
            lines.append(f"    - {step}")
        return "\n".join(lines)


#: Action types that always require an explicit user authorisation,
#: whatever the content says.
SENSITIVE_ACTIONS = {
    "send_email": "the action sends data to a third party",
    "send_message": "the action sends data to a third party",
    "post": "the action publishes data",
    "accept_terms": "the action accepts a legal agreement",
    "delete": "the action destroys data",
    "purchase": "the action spends money",
    "payment": "the action spends money",
    "modify_settings": "the action changes configuration",
    "grant_permission": "the action grants a permission",
    "share_data": "the action shares data",
    "install": "the action installs software",
    "write": "the action changes a file",
    "update": "the action changes a file",
    "create": "the action creates state",
    "commit": "the action changes a repository",
    "push": "the action changes a remote repository",
    "execute": "the action runs code",
    "run": "the action runs code",
}


#: Action types that are safe to take on read-only, untrusted content.
READ_ONLY_ACTIONS = {"read", "fetch", "summarize", "search", "list", "inspect"}

def evaluate_action(
    action: ProposedAction,
    authorization: Authorization,
) -> PolicyDecision:
    """Evaluate one proposed action against the user authorization.

    The rules are applied in order, and every rule that fires is recorded
    in the trace so a user can see why the decision was reached.
    """
    decision = PolicyDecision(action=action)
    trace = decision.trace

    # Rule 1: an instruction that came from untrusted content carries no
    # authority, whatever it says.
    if action.source == Authority.UNTRUSTED:
        if action.action_type in SENSITIVE_ACTIONS:
            trace.append(
                "rule 1: the instruction came from untrusted external content"
            )
            decision.decision = "denied"
            decision.rule = "untrusted_source_may_not_authorize"
            decision.reason = (
                "the request to %s originates in untrusted content, and only"
                " the user can authorise it" % action.action_type
            )
            return decision
        if action.action_type in READ_ONLY_ACTIONS:
            trace.append(
                "rule 1: untrusted source, but the action is read-only"
            )
            decision.decision = "allowed"
            decision.rule = "read_only_action_from_untrusted_source"
            decision.reason = (
                "reading untrusted content is permitted; only acting on it is not"
            )
            return decision
        trace.append(
            "rule 1: the action is neither read-only nor on the sensitive list,"
            " but untrusted content still cannot authorise it"
        )
        decision.decision = "denied"
        decision.rule = "untrusted_source_may_not_authorize"
        decision.reason = (
            "the request to %s originates in untrusted content, and only"
            " the user can authorise it" % action.action_type
        )
        return decision

    # Rule 2: trusted application policy can authorise within its scope.
    if action.source == Authority.TRUSTED_POLICY:
        if action.action_type in authorization.allowed_actions:
            trace.append("rule 2: the action type is in the trusted policy scope")
            if not authorization.allowed_targets or _target_allowed(
                action.target, authorization.allowed_targets
            ):
                decision.decision = "allowed"
                decision.rule = "trusted_policy_scope"
                decision.reason = "the trusted policy covers this action and target"
                return decision
            trace.append("the target is outside the trusted policy scope")
            decision.decision = "needs_clarification"
            decision.rule = "trusted_policy_target_out_of_scope"
            decision.reason = (
                "the trusted policy covers the action but not this target"
            )
            return decision
        trace.append("rule 2: the action type is not in the trusted policy scope")
        decision.decision = "denied"
        decision.rule = "trusted_policy_scope_exceeded"
        decision.reason = "the trusted policy does not cover this action"
        return decision

    # Rule 3: the user is the only other source of authority.
    if action.source == Authority.USER:
        if action.action_type in authorization.allowed_actions:
            trace.append("rule 3: the user authorised this action type")
            if not authorization.allowed_targets or _target_allowed(
                action.target, authorization.allowed_targets
            ):
                if _data_allowed(action.data_shared, authorization.allowed_data_classes):
                    decision.decision = "allowed"
                    decision.rule = "user_authorized"
                    decision.reason = "the user authorised this action, target, and data"
                    return decision
                trace.append("the data to be shared is not in the authorised scope")
                decision.decision = "needs_clarification"
                decision.rule = "data_scope_not_authorized"
                decision.reason = (
                    "the user authorised the action but not this data class"
                )
                return decision
            trace.append("the target is outside the authorised scope")
            decision.decision = "needs_clarification"
            decision.rule = "target_scope_not_authorized"
            decision.reason = "the user authorised the action but not this target"
            return decision
        trace.append("rule 3: the action type was not authorised")
        decision.decision = "denied"
        decision.rule = "action_not_authorized"
        decision.reason = (
            "the user did not authorise a %s action" % action.action_type
        )
        return decision

    decision.decision = "needs_clarification"
    decision.rule = "unknown_authority"
    decision.reason = "the provenance of this instruction is not known"
    return decision


def _target_allowed(target: str, allowed: list[str]) -> bool:
    """Is a target inside the authorised scope?

    An empty allowed list means the scope was not narrowed, so any
    target is acceptable.
    """
    if not allowed:
        return True
    lowered = target.lower()
    for entry in allowed:
        candidate = entry.lower()
        if lowered == candidate or lowered.startswith(candidate):
            return True
        if candidate.endswith("*") and lowered.startswith(candidate[:-1]):
            return True
    return False


def _data_allowed(shared: list[str], allowed_classes: list[str]) -> bool:
    """Is every data class to be shared inside the authorised scope?"""
    if not shared or not allowed_classes:
        return True
    for item in shared:
        lowered = item.lower()
        if not any(
            lowered == entry.lower() or entry.lower() in lowered
            for entry in allowed_classes
        ):
            return False
    return True

# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


@dataclass
class AuthorizationRunner:
    """Runs an authorization evaluation case."""

    provider: Any = None

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        inputs = case.input
        auth_input = inputs.get("authorization", {})
        authorization = Authorization(
            allowed_actions=list(auth_input.get("allowed_actions", [])),
            allowed_targets=list(auth_input.get("allowed_targets", [])),
            allowed_data_classes=list(auth_input.get("allowed_data_classes", [])),
            notes=auth_input.get("notes", ""),
        )
        action_input = inputs.get("action", {})
        action = ProposedAction(
            action_type=action_input.get("action_type", ""),
            target=action_input.get("target", ""),
            data_shared=list(action_input.get("data_shared", [])),
            source=action_input.get("source", Authority.UNTRUSTED),
            source_detail=action_input.get("source_detail", ""),
            instruction=action_input.get("instruction", ""),
        )

        decision = evaluate_action(action, authorization)

        expected = case.expected
        if "decision" in expected:
            wanted = expected["decision"]
            if isinstance(wanted, list):
                correct = decision.decision in wanted
            else:
                correct = decision.decision == wanted
        else:
            correct = None

        if "rule" in expected:
            correct = bool(correct) and decision.rule == expected["rule"]

        predicted = {
            "verdict": decision.decision,
            "decision": decision.decision,
            "rule": decision.rule,
        }

        return CaseResult(
            case_id=case.case_id,
            project=case.project,
            predicted=predicted,
            expected=expected,
            label_origin=case.label_origin,
            split=case.split,
            correct=correct,
            notes=decision.render(),
        )


def build_cases() -> list[EvalCase]:
    """Load the authorization checker evaluation cases."""
    from .core.fixtures import load_project_cases

    return load_project_cases("authorization")


def build_runner(provider: Any = None) -> AuthorizationRunner:
    """Factory used by the project registry."""
    return AuthorizationRunner(provider=provider)
