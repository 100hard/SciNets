"""Utilities for cleaning extracted text before persistence."""

from __future__ import annotations

import re
from typing import Optional

NULL_BYTE_RE = re.compile(r"\x00")
_ALLOWED_CONTROL_CHARS = {"\n", "\t"}


def sanitize_text(value: Optional[str]) -> Optional[str]:
    """Remove NUL bytes and problematic control characters from ``value``.

    The cleaning pipeline mirrors the guidance from production incidents:
    - strip NUL bytes that Postgres rejects outright;
    - re-encode using UTF-8 while dropping undecodable surrogates;
    - filter out non-printable control characters (except newlines / tabs);
    - trim surrounding whitespace.
    """

    if value is None:
        return None

    cleaned = NULL_BYTE_RE.sub("", value)
    cleaned = cleaned.encode("utf-8", "ignore").decode("utf-8", "ignore")
    cleaned = "".join(
        ch for ch in cleaned if ch in _ALLOWED_CONTROL_CHARS or ord(ch) >= 32
    )
    cleaned = cleaned.strip()
    return cleaned if cleaned else None
