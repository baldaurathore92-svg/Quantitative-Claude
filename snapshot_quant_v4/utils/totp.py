"""RFC 6238 TOTP generation using only the standard library.

Angel One requires a time-based one time password for login. The usual
dependency (``pyotp``) is a thin wrapper around ``hmac``; implementing the
30 lines of RFC 4226/6238 directly removes a third-party dependency from the
authentication path, which is desirable for an unattended trading process.

Security note: the configuration stores the *TOTP secret* (the base32 seed),
never a generated code — a code is valid for at most one time step and cannot
be persisted usefully. The secret is registered with the logging redactor at
startup so it cannot leak through a stack trace.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import struct
import time
from typing import Final

_DEFAULT_DIGITS: Final[int] = 6
_DEFAULT_INTERVAL: Final[int] = 30


class TotpError(ValueError):
    """Raised when a TOTP secret cannot be decoded or parameters are invalid."""


def _decode_base32_secret(secret: str) -> bytes:
    """Decode a (possibly human-formatted) base32 secret into bytes.

    Accepts lowercase input and spaces, and repairs missing ``=`` padding,
    because both are common in secrets copied out of an authenticator UI.
    """
    cleaned = secret.strip().replace(" ", "").replace("-", "").upper()
    if not cleaned:
        raise TotpError("TOTP secret is empty")
    padding = (-len(cleaned)) % 8
    try:
        return base64.b32decode(cleaned + ("=" * padding), casefold=True)
    except (binascii.Error, ValueError) as exc:
        raise TotpError("TOTP secret is not valid base32") from exc


def generate_totp(
    secret: str,
    *,
    timestamp: float | None = None,
    digits: int = _DEFAULT_DIGITS,
    interval: int = _DEFAULT_INTERVAL,
    digest: str = "sha1",
) -> str:
    """Return the TOTP code for ``secret``.

    Parameters
    ----------
    secret:
        Base32-encoded shared secret.
    timestamp:
        Unix time to generate for. Defaults to now. Explicit timestamps make
        the function unit-testable against the RFC 6238 vectors.
    digits:
        Code length (Angel One uses 6).
    interval:
        Time step in seconds (Angel One uses 30).
    digest:
        Hash name understood by :mod:`hashlib`.

    Raises
    ------
    TotpError
        On an undecodable secret or out-of-range parameters.
    """
    if digits < 6 or digits > 10:
        raise TotpError(f"digits must be in 6..10, got {digits}")
    if interval <= 0:
        raise TotpError(f"interval must be positive, got {interval}")
    try:
        hash_factory = getattr(hashlib, digest)
    except AttributeError as exc:  # pragma: no cover - guarded by config
        raise TotpError(f"unsupported digest: {digest!r}") from exc

    key = _decode_base32_secret(secret)
    now = time.time() if timestamp is None else timestamp
    counter = int(now // interval)
    mac = hmac.new(key, struct.pack(">Q", counter), hash_factory).digest()
    offset = mac[-1] & 0x0F
    truncated = struct.unpack(">I", mac[offset : offset + 4])[0] & 0x7FFF_FFFF
    return str(truncated % (10**digits)).zfill(digits)


def seconds_until_next_step(
    *, timestamp: float | None = None, interval: int = _DEFAULT_INTERVAL
) -> float:
    """Seconds remaining before the current TOTP code expires.

    The login routine uses this to avoid submitting a code that is about to
    roll over, which is a common cause of intermittent authentication failures.
    """
    if interval <= 0:
        raise TotpError(f"interval must be positive, got {interval}")
    now = time.time() if timestamp is None else timestamp
    return interval - (now % interval)


__all__ = ["TotpError", "generate_totp", "seconds_until_next_step"]
