# Docs Validation as a Final Agentic-Flow Step

Follow-up to [01-docs-cleanup.md](./01-docs-cleanup.md). That issue is a one-time cleanup of
the current cruft; this one makes it a **standing practice** so it doesn't come back.

Agentic coding loops keep leaving process/self-talk artifacts (references to the numbered
design docs, "closes the loop"-style narration). Rather than periodically re-cleaning, run a
doc-hygiene check as the **final validation step of every agentic flow**.

## Shape

- Mechanical gate (cheap, cheat-proof): `grep -rnE '0[0-9]-[a-z-]+\.md|step [0-9]' src tests`
  returns nothing — wire into `scripts/check.py` so it runs with the other checks.
- Reviewer/agent pass over the flow's diff for tautological comments and build-narrative
  self-talk (not mechanizable; it's the human/agent judgement part).

## Sequencing note

Filed at the end of the list, but it's a *recurring* gate, not a one-shot task — once it
exists it runs at the close of each flow. The one-time cleanup in 01 comes first so the
baseline is clean before the gate turns on.
