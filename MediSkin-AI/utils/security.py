"""Password hashing, text sanitising and safe file-path helpers."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

import bcrypt

_BCRYPT_MAX_BYTES = 72


def _password_bytes(password: str) -> bytes:
    # bcrypt ignores bytes beyond 72; truncate explicitly so behaviour is predictable.
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_password_bytes(password), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(value: str, max_length: int) -> str:
    """Strip control characters and surrounding whitespace, then truncate."""
    cleaned = _CONTROL_CHARS.sub("", value or "").strip()
    return cleaned[:max_length]


def unique_filename(prefix: str, extension: str) -> str:
    """Return a collision-free, filesystem-safe filename."""
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]", "", prefix) or "file"
    return f"{safe_prefix}_{uuid.uuid4().hex}.{extension.lstrip('.')}"


def is_within_directory(path: Path | str, directory: Path | str) -> bool:
    """True if ``path`` resolves inside ``directory`` (blocks path traversal)."""
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except ValueError:
        return False
