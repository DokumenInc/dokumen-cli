# Contributing

This project is designed for focused, test-backed changes.

## Workflow

1. Create a branch for the change.
2. Run the smallest relevant test while developing.
3. Add or update tests for behavior changes.
4. Update documentation for user-facing changes.
5. Run the relevant verification commands before opening a pull request.

## Verification

```bash
uv run --extra dev pytest tests -q
uv run --extra dev ruff check dokumen dokumen_schema tests
uv run --extra dev black --check dokumen dokumen_schema tests
uv run --extra dev dokumen validate
```

For CLI changes, include command-level tests for exit codes and output shape.
For tool changes, include success, validation, and failure-path tests.

## Style

- Keep comments useful and specific. Explain why a branch exists, not what a
  simple assignment does.
- Prefer existing module boundaries over new abstractions.
- Keep tool and stage behavior observable in logs and result artifacts.
- Do not commit secrets, `.dokumen-cache/`, virtual environments, or generated
  exploration output.
