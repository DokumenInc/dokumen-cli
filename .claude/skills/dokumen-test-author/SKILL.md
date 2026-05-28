---
name: dokumen-test-author
description: Use when creating or revising Dokumen CLI test scaffolds for Claude Code-style skills, including executor.skills, minimal tool allowlists, and LLM judge success criteria. This replaces the removed dokumen create command.
---

# Dokumen Test Author

Use this workflow to add or revise a Dokumen test scaffold.

## Workflow

1. Read the nearest `dokumen.yaml`, existing `tests/*.test.yaml`, and any local
   `skills/` examples before writing a new scaffold.
2. Prefer the default skill-use shape: one executor prompted to use a named
   skill, followed by one or more LLM judges that evaluate success criteria.
3. If the requested skill does not exist, add a concise skill file under the
   target project's `skills/` directory before writing the test.
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
name: release-note-skill
reason: Verify that the executor applies the named skill before judging.

files:
  - path: docs/release-notes.md

executor:
  skills:
    - release-note-review
  tools:
    - read_file
  user_prompt: |
    Use the release-note-review skill to inspect the referenced release notes.
    Report the intended audience, changed behavior, required user action, and
    any vague migration language.

judges:
  - name: release-note-success-criteria
    include_executor_output: true
    system_prompt: |
      Pass only if the executor output proves it used the release-note-review
      skill and clearly reports the intended audience, changed behavior,
      required user action, and whether vague migration language exists.
      Return only JSON: {"verdict": "PASS" or "FAIL", "reason": "..."}.
```
