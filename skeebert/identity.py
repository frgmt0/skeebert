"""Turning Discord ids into stable pseudonyms.

This is the single chokepoint for identity. The database never sees a raw
Discord id: users are stored as ``u_<16 hex>`` and the glyph messages Skeebert
posts are remembered as ``m_<16 hex>``, both salted SHA-256 digests.

The salt lives in ``data/.salt``. It is created once, on the very first run,
and must never change or be lost: ``forget`` (the "Delete everything" button
and ``skeebert forget``) finds a person's rows by re-deriving their hash, so a
rotated salt would orphan exactly the rows it needs to delete. The salt never
leaves the machine the bot runs on.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path


class Pseudonymiser:
    """Salted one-way hashes for Discord ids."""

    def __init__(self, salt: str) -> None:
        if not salt:
            raise ValueError("refusing to pseudonymise with an empty salt")
        self._salt = salt.encode("utf-8")

    def user(self, user_id: object) -> str:
        return "u_" + self._digest(f"user:{user_id}")[:16]

    def message(self, message_id: object) -> str:
        return "m_" + self._digest(f"message:{message_id}")[:16]

    def _digest(self, value: str) -> str:
        return hashlib.sha256(self._salt + b"|" + value.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, salt_path: str | os.PathLike, *, create: bool = True) -> "Pseudonymiser":
        """Use the salt in ``salt_path``; create it (once, 0600) only if ``create`` is True.

        Callers pass ``create=False`` whenever data hashed with a salt may
        already exist (a non-empty database): a fresh salt there would make
        every stored row unreachable by ``forget``, so it is refused with
        ``SaltMissing`` instead.
        """
        path = Path(salt_path)
        if path.exists():
            salt = path.read_text(encoding="utf-8").strip()
            if not salt:
                raise ValueError(
                    f"{path.resolve()} exists but is empty. Restore it from backup: a new salt would make "
                    "every stored row unreachable by forget."
                )
            return cls(salt)
        if not create:
            raise SaltMissing(
                f"no hash salt at {path.resolve()}, but data already exists next to it. Restore data/.salt from "
                "backup; creating a new one would orphan every stored row (forget could no longer find them)."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        salt = secrets.token_hex(32)
        # O_EXCL: if two processes race on first start, exactly one salt wins.
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return cls.load(path)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(salt + "\n")
        return cls(salt)


class SaltMissing(RuntimeError):
    """The salt file is gone but hashed data exists: refusing to mint a new salt."""
