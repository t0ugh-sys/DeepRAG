from __future__ import annotations

import os
import re
from pathlib import Path


_WINDOWS_DRIVE_RE = re.compile(r"^[a-zA-Z]:[\\/]")


def normalize_doc_path(path: str) -> str:
    """
    Normalize a logical document path.

    Security goals:
    - Make traversal attempts obvious and reject them upstream.
    - Avoid absolute paths / Windows drive paths.
    - Keep path stable across Windows/Linux by normalizing slashes.
    """
    p = (path or "").strip().replace("\\", "/")

    # Reject absolute or drive paths.
    if p.startswith("/") or _WINDOWS_DRIVE_RE.match(p):
        raise ValueError("Invalid path")

    # Collapse repeated slashes.
    p = re.sub(r"/{2,}", "/", p)

    # Remove leading "./"
    while p.startswith("./"):
        p = p[2:]

    # Reject traversal.
    parts = [seg for seg in p.split("/") if seg]
    if any(seg == ".." for seg in parts):
        raise ValueError("Invalid path")

    normalized = "/".join(parts)
    if not normalized:
        raise ValueError("Invalid path")
    if len(normalized) > 1024:
        raise ValueError("Invalid path")
    return normalized


def enforce_namespace_prefix(path: str, namespace: str) -> str:
    """
    Enforce a simple namespace path prefix convention: '<namespace>/<path>'.

    When enabled, this prevents cross-namespace reads/writes by making the
    namespace part of the document key stored in indexes/metadata.
    """
    ns = (namespace or "").strip()
    if not ns or ns == "default":
        return path
    if path.startswith(f"{ns}/"):
        return path
    return f"{ns}/{path}"


def bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_data_file_path(filepath: str, base_dir: str = "data") -> str:
    """
    Normalize and restrict file path writes to within `base_dir`.
    """
    p = (filepath or "").strip().replace("\\", "/")
    if not p:
        raise ValueError("Invalid filepath")
    if p.startswith("/") or _WINDOWS_DRIVE_RE.match(p):
        raise ValueError("Invalid filepath")
    if ".." in p.split("/"):
        raise ValueError("Invalid filepath")

    base = Path(base_dir).resolve()
    full = (base / p.removeprefix(f"{base_dir}/")).resolve()
    try:
        full.relative_to(base)
    except Exception as exc:  # pragma: no cover
        raise ValueError("Invalid filepath") from exc
    return full.as_posix()
