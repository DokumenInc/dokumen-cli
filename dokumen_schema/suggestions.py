"""Human-readable fixes for common scaffold validation errors."""


def get_suggested_fix(error: str) -> str:
    """Return a concise suggested fix for a validation error."""
    lower = error.lower()
    if "unknown tool" in lower:
        return (
            "Check the tool name with `dokumen list tools` or update dokumen.yaml tool allowlists."
        )
    if "referenced file not found" in lower:
        return "Create the referenced file or update the scaffold `files` path."
    if "browser" in lower and "type" in lower:
        return "Set `type: browser` when using a `browser:` scaffold section."
    if "files" in lower:
        return "Add at least one `files:` entry with a valid `path`."
    if "name" in lower:
        return "Use a kebab-case test name such as `api-authentication-docs`."
    return "Review the scaffold field named in the error and rerun `dokumen validate --verbose`."
