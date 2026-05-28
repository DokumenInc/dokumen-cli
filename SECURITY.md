# Security

## Reporting

Report security issues privately to the Dokumen maintainers. Do not open a
public issue with exploit details.

## Credential Handling

Dokumen can use provider API keys, remote Git tokens, browser credentials, and
integration-test secrets. Keep those values in environment variables, CI secret
stores, or a dedicated secret manager.

Never commit:

- provider API keys,
- remote Git access tokens,
- browser session cookies,
- `.env` files,
- `.dokumen-cache/` run output,
- screenshots or traces that contain customer data.

Rotate any credential that was committed, even if it was removed later.

## Tool Safety

Dokumen agents may receive file, shell, web, browser, and write tools. Use
project-level allowlists and blocked tool settings for untrusted repositories.
Shell and write tools should be enabled only for tests that require them.

## CI Safety

Use masked CI variables for credentials. Avoid printing environment variables in
test logs. Debug traces are useful for diagnosis, but they may include prompts,
tool outputs, and generated files, so treat them as sensitive artifacts.
