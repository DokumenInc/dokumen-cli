---
name: dokumen-test-author
description: Use when creating or revising Dokumen CLI test scaffolds for business SOP adherence, including executor.sops, minimal tool allowlists, and LLM judge success criteria. This replaces the removed dokumen create command.
---

# Dokumen Test Author

Use this workflow to add or revise a Dokumen test scaffold.

## Workflow

1. Read the nearest `dokumen.yaml`, existing `tests/*.test.yaml`, and any local
   `sops/` or `skills/` examples before writing a new scaffold.
2. Prefer the default SOP-test shape: one executor prompted to follow a named
   SOP through `executor.sops`, followed by one or more LLM judges that evaluate
   success criteria.
3. If the requested SOP does not exist, add a concise SOP file under the target
   project's `sops/` directory before writing the test.
4. Write the scaffold under `tests/<kebab-name>.test.yaml`.
5. Keep coordinator mode out of the scaffold unless the user explicitly asks for
   multi-worker execution.
6. Keep the executor tool list narrow. Start with `read_file`; add `glob`,
   `search_file_content`, browser tools, shell, or write tools only when the
   task genuinely needs them.
7. Make judge prompts binary and falsifiable. Ask for JSON only:
   `{"verdict": "PASS" or "FAIL", "reason": "..."}`.
8. Run `dokumen validate` after authoring. Run `dokumen run --dry-run` when an
   offline check is enough, or the scoped real test when credentials are
   available.

## Scaffold Template

```yaml
name: refund-escalation
reason: Verify that the executor follows the refund escalation SOP.

files:
  - path: docs/customer-ticket.md

executor:
  sops:
    - refund-escalation-sop
  tools:
    - read_file
  user_prompt: |
    Follow the refund-escalation-sop while reviewing the referenced customer
    ticket. Report the customer's plan, request amount, refund-window status,
    escalation requirement, and recommended next action.

judges:
  - name: sop-success-criteria
    include_executor_output: true
    system_prompt: |
      Pass only if the executor output proves it followed the refund escalation
      SOP and clearly reports the plan, amount, refund-window status,
      escalation requirement, and recommended next action.
      Return only JSON: {"verdict": "PASS" or "FAIL", "reason": "..."}.
```

The legacy `executor.skills` field still works for existing scaffolds, but new
business-process tests should prefer `executor.sops`.
