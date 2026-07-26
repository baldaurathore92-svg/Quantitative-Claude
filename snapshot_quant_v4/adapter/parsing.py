"""Tolerant parsing of SmartAPI V2 SnapQuote payloads.

Why this module exists separately from the adapter
-------------------------------------------------
Parsing is the part of the integration most likely to break, and it is also the
only part that can be tested without a network. Isolating it means the payload
handling has unit tests, while the adapter is left with nothing but connection
management.

Defensive assumptions
---------------------
The engine makes no assumptions about the exact shape of the callback payload,
because the SmartAPI Python client has shipped several field spellings across
versions and the callback can also deliver a raw JSON string or bytes. The parser
therefore:

*   accepts ``dict``, ``str``, ``bytes``/``bytearray`` and lists of any of those;
*   recognises heartbeat and acknowledgement frames (``pong``, ``ping``, plain
    strings, dictionaries without a token) and reports them as *not an error*;
*   looks each field up through a list of candidate keys
    (:data:`_TOKEN_KEYS` and friends) instead of one hardcoded name;
*   converts the wire price integers to rupees exactly once, using the
    per-segment divisor from :mod:`utils.constants`, and keeps the integer paise
    value for exact price-keyed comparison downstream.

Anything genuinely unparseable raises :class:`PayloadError`, which the adapter
counts and logs. Silence is never an option: a payload shape change that halved
the depth would otherwise degrade every feature invisibly.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from ..utils.constants import (
    DEFAULT_PRICE_DIVISOR,
    PRICE_DIVISOR_BY_EXCHANGE,
    SNAPQUOTE_DEPTH_LEVELS,
)
from ..utils.types import DepthLevel, RawSnapshot


class PayloadError(ValueError):
    """Raised when a payload claims to be market data but cannot be parsed."""


# Candidate key spellings, in priority order.
_TOKEN_KEYS: Final[tuple[str, ...]] = ("token", "tk", "instrument_token")
_EXCHANGE_KEYS: Final[tuple[str, ...]] = ("exchange_type", "exchange", "e")
_TIMESTAMP_KEYS: Final[tuple[str, ...]] = (
    "exchange_timestamp",
    "exchange_feed_time_epoch_millis",
    "exchange_feed_time",
    "ft",
)
_SEQUENCE_KEYS: Final[tuple[str, ...]] = ("sequence_number", "seq", "sequence")
_LTP_KEYS: Final[tuple[str, ...]] = ("last_traded_price", "ltp", "lp")
_LTQ_KEYS: Final[tuple[str, ...]] = ("last_traded_quantity", "ltq")
_ATP_KEYS: Final[tuple[str, ...]] = ("average_traded_price", "avg_traded_price", "ap")
_VOLUME_KEYS: Final[tuple[str, ...]] = (
    "volume_trade_for_the_day",
    "volume_traded_today",
    "volume",
    "v",
)
_TOTAL_BUY_KEYS: Final[tuple[str, ...]] = (
    "total_buy_quantity",
    "total_buy_quant",
    "tbq",
)
_TOTAL_SELL_KEYS: Final[tuple[str, ...]] = (
    "total_sell_quantity",
    "total_sell_quant",
    "tsq",
)
_OPEN_KEYS: Final[tuple[str, ...]] = ("open_price_of_the_day", "open_price", "open", "o")
_HIGH_KEYS: Final[tuple[str, ...]] = ("high_price_of_the_day", "high_price", "high", "h")
_LOW_KEYS: Final[tuple[str, ...]] = ("low_price_of_the_day", "low_price", "low", "l")
_CLOSE_KEYS: Final[tuple[str, ...]] = ("closed_price", "close_price", "close", "c")
_BUY_DEPTH_KEYS: Final[tuple[str, ...]] = ("best_5_buy_data", "best_five_buy", "bids")
_SELL_DEPTH_KEYS: Final[tuple[str, ...]] = ("best_5_sell_data", "best_five_sell", "asks")
_LEVEL_PRICE_KEYS: Final[tuple[str, ...]] = ("price", "p")
_LEVEL_QUANTITY_KEYS: Final[tuple[str, ...]] = ("quantity", "qty", "q")
# The order-count key is spelled with spaces in the SmartAPI client, which is easy
# to miss and is the single most valuable field in the payload after the ladder
# itself.
_LEVEL_ORDERS_KEYS: Final[tuple[str, ...]] = (
    "no of orders",
    "no_of_orders",
    "orders",
    "no_of_order",
)

#: Payload values that identify a keep-alive frame rather than market data.
_HEARTBEAT_TOKENS: Final[frozenset[str]] = frozenset({"pong", "ping", "", "ok"})


def _lookup(payload: Mapping[str, Any], keys: Iterable[str]) -> Any:
    """Return the first present key from ``keys``, or ``None``."""
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _as_int(value: Any, *, default: int = 0) -> int:
    """Coerce a wire value to ``int``, tolerating numeric strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default
    return default


def _as_float(value: Any, *, default: float = 0.0) -> float:
    """Coerce a wire value to ``float``, tolerating numeric strings."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default
    return default


def is_heartbeat(payload: Any) -> bool:
    """Return ``True`` for keep-alive and acknowledgement frames.

    Heartbeats are frequent and completely normal; treating them as parse
    failures would bury the log in noise and hide real problems.
    """
    if payload is None:
        return True
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return False
    if isinstance(payload, str):
        text = payload.strip().strip('"').lower()
        if text in _HEARTBEAT_TOKENS:
            return True
        return not text.startswith(("{", "["))
    if isinstance(payload, Mapping):
        # A mapping with no instrument token is a control or acknowledgement
        # frame rather than market data.
        return _lookup(payload, _TOKEN_KEYS) is None
    return False


def coerce_payloads(payload: Any) -> list[Mapping[str, Any]]:
    """Normalise any callback payload into a list of mappings.

    Handles the four shapes the SmartAPI client is known to deliver: a mapping, a
    JSON string, JSON bytes, and a list of any of those. Heartbeats yield an
    empty list.

    Raises
    ------
    PayloadError
        If the payload is a non-heartbeat string that is not valid JSON, or is of
        a type that cannot contain market data.
    """
    if is_heartbeat(payload):
        return []

    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PayloadError("payload bytes are not valid UTF-8") from exc

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise PayloadError(f"payload is not valid JSON: {exc.msg}") from exc

    if isinstance(payload, Mapping):
        return [payload]

    if isinstance(payload, Sequence):
        items: list[Mapping[str, Any]] = []
        for entry in payload:
            items.extend(coerce_payloads(entry))
        return items

    raise PayloadError(f"unsupported payload type {type(payload).__name__}")


def price_divisor(exchange_type: int) -> int:
    """Return the wire-to-rupee divisor for a segment."""
    return PRICE_DIVISOR_BY_EXCHANGE.get(exchange_type, DEFAULT_PRICE_DIVISOR)


def parse_depth(
    entries: Any, divisor: int, *, descending: bool
) -> tuple[DepthLevel, ...]:
    """Parse one side of the ladder into sorted, non-empty levels.

    Empty levels (price or quantity of zero) are dropped rather than retained as
    placeholders: SmartAPI pads the five-level array for thin books, and a
    zero-priced placeholder would corrupt every distance-weighted calculation.

    The result is explicitly re-sorted, descending for bids and ascending for
    asks, because the wire order is not guaranteed and the validator's
    monotonicity check would otherwise reject a perfectly good book.
    """
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return ()
    levels: list[DepthLevel] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        price_wire = _as_int(_lookup(entry, _LEVEL_PRICE_KEYS))
        quantity = _as_int(_lookup(entry, _LEVEL_QUANTITY_KEYS))
        if price_wire <= 0 or quantity <= 0:
            continue
        orders = _as_int(_lookup(entry, _LEVEL_ORDERS_KEYS))
        price_paise = price_wire if divisor == 100 else round(price_wire * 100 / divisor)
        levels.append(
            DepthLevel(
                price_paise=price_paise,
                price=price_wire / divisor,
                quantity=quantity,
                # A missing order count is reported as one rather than zero: zero
                # means "no orders", which is false when quantity is present, and
                # several ratios divide by it.
                orders=orders if orders > 0 else 1,
            )
        )
        if len(levels) >= SNAPQUOTE_DEPTH_LEVELS:
            break
    levels.sort(key=lambda level: level.price_paise, reverse=descending)
    return tuple(levels)


def parse_snapshot(
    payload: Mapping[str, Any],
    *,
    received_monotonic_ms: float,
    symbol: str = "",
    fallback_exchange_type: int = 1,
) -> RawSnapshot:
    """Build a :class:`RawSnapshot` from one SnapQuote mapping.

    Parameters
    ----------
    payload:
        A single market-data mapping (already extracted by
        :func:`coerce_payloads`).
    received_monotonic_ms:
        Local monotonic receive time, stamped by the adapter as early as
        possible so that queueing delay can be measured later.
    symbol:
        Display name for the token, from the configuration.
    fallback_exchange_type:
        Segment to assume when the payload omits it.

    Raises
    ------
    PayloadError
        If the token is missing, or the payload carries no usable depth on either
        side. Both cases mean the message is not a SnapQuote and must not be
        silently turned into an empty book.
    """
    token_value = _lookup(payload, _TOKEN_KEYS)
    if token_value is None:
        raise PayloadError("payload has no instrument token")
    token = str(token_value).strip()
    if not token:
        raise PayloadError("payload has an empty instrument token")

    exchange_type = _as_int(
        _lookup(payload, _EXCHANGE_KEYS), default=fallback_exchange_type
    )
    if exchange_type <= 0:
        exchange_type = fallback_exchange_type
    divisor = price_divisor(exchange_type)

    bids = parse_depth(_lookup(payload, _BUY_DEPTH_KEYS), divisor, descending=True)
    asks = parse_depth(_lookup(payload, _SELL_DEPTH_KEYS), divisor, descending=False)
    if not bids and not asks:
        raise PayloadError(
            f"token {token}: payload carries no depth; is the subscription in "
            "SnapQuote (mode 3)?"
        )

    return RawSnapshot(
        token=token,
        exchange_type=exchange_type,
        exchange_timestamp_ms=_as_int(_lookup(payload, _TIMESTAMP_KEYS)),
        received_monotonic_ms=received_monotonic_ms,
        sequence_number=_as_int(_lookup(payload, _SEQUENCE_KEYS)),
        last_traded_price=_as_float(_lookup(payload, _LTP_KEYS)) / divisor,
        last_traded_quantity=_as_int(_lookup(payload, _LTQ_KEYS)),
        average_traded_price=_as_float(_lookup(payload, _ATP_KEYS)) / divisor,
        volume_traded_today=_as_int(_lookup(payload, _VOLUME_KEYS)),
        total_buy_quantity=_as_float(_lookup(payload, _TOTAL_BUY_KEYS)),
        total_sell_quantity=_as_float(_lookup(payload, _TOTAL_SELL_KEYS)),
        open_price=_as_float(_lookup(payload, _OPEN_KEYS)) / divisor,
        high_price=_as_float(_lookup(payload, _HIGH_KEYS)) / divisor,
        low_price=_as_float(_lookup(payload, _LOW_KEYS)) / divisor,
        close_price=_as_float(_lookup(payload, _CLOSE_KEYS)) / divisor,
        bids=bids,
        asks=asks,
        symbol=symbol or token,
    )


def snapshot_to_json(snapshot: RawSnapshot) -> str:
    """Serialise a snapshot to one JSONL record.

    Used by the recorder so that a live session can be replayed deterministically
    later. The format is the engine's own normalised form (rupees, sorted
    ladders), not the raw wire format, because replaying the *validated* input is
    what makes a regression reproducible.
    """
    return json.dumps(
        {
            "token": snapshot.token,
            "symbol": snapshot.symbol,
            "exchange_type": snapshot.exchange_type,
            "exchange_timestamp_ms": snapshot.exchange_timestamp_ms,
            "sequence_number": snapshot.sequence_number,
            "last_traded_price": snapshot.last_traded_price,
            "last_traded_quantity": snapshot.last_traded_quantity,
            "average_traded_price": snapshot.average_traded_price,
            "volume_traded_today": snapshot.volume_traded_today,
            "total_buy_quantity": snapshot.total_buy_quantity,
            "total_sell_quantity": snapshot.total_sell_quantity,
            "open_price": snapshot.open_price,
            "high_price": snapshot.high_price,
            "low_price": snapshot.low_price,
            "close_price": snapshot.close_price,
            "bids": [
                {"price": level.price, "quantity": level.quantity, "orders": level.orders}
                for level in snapshot.bids
            ],
            "asks": [
                {"price": level.price, "quantity": level.quantity, "orders": level.orders}
                for level in snapshot.asks
            ],
        },
        separators=(",", ":"),
    )


def snapshot_from_json(
    record: Mapping[str, Any], *, received_monotonic_ms: float
) -> RawSnapshot:
    """Rebuild a snapshot from a JSONL record written by :func:`snapshot_to_json`.

    Raises
    ------
    PayloadError
        If the record is missing a token or has no depth.
    """
    token = str(record.get("token", "")).strip()
    if not token:
        raise PayloadError("replay record has no token")

    def levels(key: str, descending: bool) -> tuple[DepthLevel, ...]:
        entries = record.get(key, ())
        if not isinstance(entries, Sequence):
            return ()
        parsed = [
            DepthLevel(
                price_paise=round(_as_float(entry.get("price")) * 100),
                price=_as_float(entry.get("price")),
                quantity=_as_int(entry.get("quantity")),
                orders=max(1, _as_int(entry.get("orders"), default=1)),
            )
            for entry in entries
            if isinstance(entry, Mapping)
            and _as_float(entry.get("price")) > 0.0
            and _as_int(entry.get("quantity")) > 0
        ]
        parsed.sort(key=lambda level: level.price_paise, reverse=descending)
        return tuple(parsed)

    bids = levels("bids", True)
    asks = levels("asks", False)
    if not bids and not asks:
        raise PayloadError(f"replay record for {token} has no depth")

    return RawSnapshot(
        token=token,
        exchange_type=_as_int(record.get("exchange_type"), default=1),
        exchange_timestamp_ms=_as_int(record.get("exchange_timestamp_ms")),
        received_monotonic_ms=received_monotonic_ms,
        sequence_number=_as_int(record.get("sequence_number")),
        last_traded_price=_as_float(record.get("last_traded_price")),
        last_traded_quantity=_as_int(record.get("last_traded_quantity")),
        average_traded_price=_as_float(record.get("average_traded_price")),
        volume_traded_today=_as_int(record.get("volume_traded_today")),
        total_buy_quantity=_as_float(record.get("total_buy_quantity")),
        total_sell_quantity=_as_float(record.get("total_sell_quantity")),
        open_price=_as_float(record.get("open_price")),
        high_price=_as_float(record.get("high_price")),
        low_price=_as_float(record.get("low_price")),
        close_price=_as_float(record.get("close_price")),
        bids=bids,
        asks=asks,
        symbol=str(record.get("symbol", "") or token),
    )


__all__ = [
    "PayloadError",
    "coerce_payloads",
    "is_heartbeat",
    "parse_depth",
    "parse_snapshot",
    "price_divisor",
    "snapshot_from_json",
    "snapshot_to_json",
]
