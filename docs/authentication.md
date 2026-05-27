# Authentication And Credentials

Dokumen reads credentials from environment variables or a secret manager. Do not
commit real tokens to a project repository.

## LLM Providers

Set provider credentials in the environment used to run the CLI:

```bash
export ANTHROPIC_API_KEY=...
```

The provider and model can be configured in `dokumen.yaml`. Environment
variables are preferred for secrets because they do not enter Git history.

## GitLab Integration

Workspace and repository features can use a GitLab personal access token:

```bash
export DOKUMEN_PAT=...
```

For CI, store the token as a masked CI variable. The token should have only the
scopes required by the operation being tested.

## Secret Hygiene

- Keep real tokens out of documentation, fixtures, and examples.
- Use obvious placeholders in tests, such as `glpat-example`.
- Rotate any token that was ever committed, even if it was removed later.
- Prefer short-lived integration-test credentials for automation.
