# Authentication And Credentials

Dokumen reads provider credentials from environment variables. Do not commit
real tokens to a project repository.

## Provider Credentials

Set the Anthropic API key in the environment used to run agent-backed commands:

```bash
export ANTHROPIC_API_KEY=...
```

The provider and model can be configured in `dokumen.yaml`. Environment
variables are preferred for secrets because they do not enter Git history.

## Secret Hygiene

- Keep real tokens out of documentation, fixtures, and examples.
- Use obvious placeholders in tests, such as `ANTHROPIC_API_KEY=example`.
- Rotate any token that was ever committed, even if it was removed later.
- Prefer short-lived integration-test credentials for automation.
