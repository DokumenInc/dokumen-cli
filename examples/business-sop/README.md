# Business SOP Example

This example shows the default Dokumen pattern:

1. Put the business procedure in a reusable SOP file.
2. Reference that SOP from the scaffold's `executor.sops`.
3. Prompt the executor to follow the named SOP while handling the case.
4. Let an LLM judge decide whether the executor met the success criteria.

From this directory, set `ANTHROPIC_API_KEY` and run:

```bash
dokumen validate
dokumen run refund-escalation
```

The coordinator is intentionally not involved. This is a single executor
attempt against a business SOP followed by an LLM judge.
