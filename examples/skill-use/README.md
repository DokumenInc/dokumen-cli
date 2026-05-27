# Skill-Use Example

This example shows the default Dokumen pattern:

1. Put reusable instructions in a skill file.
2. Reference that skill from the scaffold's `executor.skills`.
3. Prompt the executor to use the named skill.
4. Let an LLM judge decide whether the executor met the success criteria.

From this directory, set `ANTHROPIC_API_KEY` and run:

```bash
dokumen validate
dokumen run tests/release-note-skill.test.yaml
```

The coordinator is intentionally not involved. This is a single executor skill
attempt followed by an LLM judge.
