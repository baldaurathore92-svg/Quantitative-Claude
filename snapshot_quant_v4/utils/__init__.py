"""Shared primitives: constants, data model, clock, maths, logging, TOTP.

This package must never import from :mod:`snapshot_quant_v4.engine`,
:mod:`snapshot_quant_v4.features` or :mod:`snapshot_quant_v4.adapter`. Keeping
the dependency arrow pointing one way is what allows features and buffers to be
unit tested in isolation.
"""

from __future__ import annotations
