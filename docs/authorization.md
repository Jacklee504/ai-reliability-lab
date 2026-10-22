# Authorization checker

Check whether an agent was authorised to take the action it proposed to take.

**Question it answers:** Was the agent authorised to take this action?

**Released as:** `authorization_checker`

## Why it exists

An agent reading a web page can be told what to do by that page. A ticket that says email the report to this address, or a document that quotes the user giving permission, is untrusted content wearing the user's clothes.

The checker evaluates a proposed action against what the user actually authorised, and it fails closed. It never acts on the action. It only answers whether the action was permitted, so the decision stays with the caller.

## How it works

1. **Provenance** records where the instruction came from: the user, a trusted application policy, or untrusted content.
2. **Scope** checks the action type, the target, and the data class against what was authorised.
3. **Decision** returns allowed, denied, or needs clarification, with the rule that fired recorded in the trace.

## The rules that keep it honest

* Untrusted content can never authorise an action, whatever it quotes or claims.
* Reading untrusted content is allowed. Only acting on it needs authority.
* An action in scope for the action type but out of scope for the target or the data class returns needs clarification, not a silent allow and not a blunt deny.
* An instruction whose provenance is unknown returns needs clarification rather than being assumed safe.

## Known limitations

* The checker evaluates a proposed action. It does not observe what the agent actually did.
* The policy is a small fixed rule set. A real deployment needs its own organisation's rules.
* It judges the scope of an action, not whether the action was wise.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval authorization --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release authorization --out releases
```
