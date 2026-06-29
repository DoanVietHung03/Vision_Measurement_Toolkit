"""Path helpers that keep configs independent from current working directory."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the workspace root for this source layout."""
    return Path(__file__).resolve().parents[3]


def resolve_path(path_value: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    """Resolve a possibly relative path against a base directory or project root."""
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    base = Path(base_dir) if base_dir is not None else project_root()
    return (base / path).resolve()


def require_existing_path(path_value: str | Path | None, label: str, base_dir: str | Path | None = None) -> Path:
    """Resolve and require a path to exist."""
    resolved = resolve_path(path_value, base_dir=base_dir)
    if resolved is None:
        raise ValueError(f"{label} path is not configured")
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved
