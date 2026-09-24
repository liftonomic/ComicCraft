"""Security helpers: safe identifier generation.

A thin, dependency-free wrapper around :mod:`secrets` used to mint
URL-safe identifiers for comics and panels. Kept separate from the scheme
/ stores so it can be swapped for a user-scoped namespace later.
"""

from __future__ import annotations

import secrets
import string

_ALPHABET = string.ascii_lowercase + string.digits


def new_id(length: int = 12) -> str:
    """Return a URL-safe, unpredictable identifier of ``length`` chars."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))